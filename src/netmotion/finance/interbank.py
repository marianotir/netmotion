"""The weighted interbank debt network.

Thesis section 5.6.4 (PDF pp. 68-71) and section 2.1 of

    M. Tirado, *Complex Network for a Crisis Contagion on an Interbank System*,
    Int. J. Mod. Phys. C **23**(9), 1250058 (2012).

A community is a country, a node is a bank, and a link is a lending
relationship.  What turns that structure into a financial system is a weight:
how much money flows along each link, and therefore how exposed each bank is.

The weight rule
---------------
Both sources set the weight of a link from the *degrees of its endpoints*;
the thesis computes them once from the finished network (PDF p. 68).
The article's Eq. (1) reads

.. math:: W_{ij} = k_i - k_j \\quad \\text{if } M_{ij} = 1, \\qquad 0 \\text{ otherwise}

with :math:`W_{ij}` the money bank *j* owes bank *i*.  The idea is that a
well-connected bank lends to a less-connected one, and the imbalance in
connectivity is the size of the debt.

The matrix is antisymmetric, so a bank's net position is

.. math:: D_i = \\sum_j W_{ij} = k_i^2 - \\sum_{j \\in \\mathcal{N}(i)} k_j

and, since :math:`W_{ij} = -W_{ji}`, the whole system nets to zero:

.. math:: \\sum_i D_i = 0

which the thesis states explicitly on PDF p. 70 and which
:func:`~netmotion.finance.interbank.InterbankNetwork.check_conservation`
verifies.  :math:`D_i` is what absorbs losses in the cascade of
:mod:`netmotion.finance.contagion`, so its sign decides which banks are
creditors and which are debtors.

Two sign conventions
--------------------
The thesis writes the rule on PDF p. 68 as :math:`W_{ij} = k_j - k_i`, the debt
bank *i* has with bank *j*; the article writes :math:`W_{ij} = k_i - k_j`.
Both are implemented: ``convention="thesis"`` and ``convention="paper"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph import Network
from ..networks.communities import CommunityNetwork
from ..provenance import Origin, sourced

__all__ = ["InterbankNetwork", "debt_matrix", "bank_capital", "too_big_to_fail"]


@sourced(
    Origin.PAPER_2012,
    pdf_pages="3",
    section="2.1 The network model (Eq. 1)",
    document="paper",
    note=(
        "W_ij = k_i - k_j on existing links, 0 elsewhere, computed from the "
        "final degrees of the grown network. Antisymmetric by construction."
    ),
    reading=(
        "The thesis writes W_ij = k_j - k_i (PDF p. 68): convention='thesis'. "
        "convention='paper' is the article's W_ij = k_i - k_j."
    ),
    tags=("sign-convention",),
)
def debt_matrix(network: Network, *, convention: str = "paper") -> np.ndarray:
    """The antisymmetric weight matrix :math:`W`.

    Parameters
    ----------
    convention:
        ``"paper"`` for :math:`W_{ij} = k_i - k_j` (article Eq. 1, the default),
        ``"thesis"`` for :math:`W_{ij} = k_j - k_i` (thesis PDF p. 68).

    Examples
    --------
    The five-node example: nodes 0 and 1 are hubs of degree 4, nodes 2, 3 and 4
    have degree 2.

    >>> from netmotion import Network
    >>> g = Network.from_edges(5, [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)])
    >>> w = debt_matrix(g)
    >>> float(w[0, 2]), float(w[2, 0])
    (2.0, -2.0)

    Antisymmetry is exact, so the system nets to zero:

    >>> float(w.sum())
    0.0
    """
    if convention not in ("paper", "thesis"):
        raise ValueError(f"convention must be 'paper' or 'thesis', got {convention!r}")

    a = network.adjacency
    k = a.sum(axis=1)
    difference = k[:, None] - k[None, :]
    if convention == "thesis":
        difference = -difference
    return difference * a


@sourced(
    Origin.PAPER_2012,
    pdf_pages="3",
    section="2.1 The network model",
    document="paper",
    note=(
        "D_i = sum_j W_ij, the node intensity in the sense of Barrat et al. "
        "It is the bank's net position and the capital that absorbs losses in "
        "the contagion cascade. In the thesis, PDF p. 68."
    ),
)
def bank_capital(network: Network, *, convention: str = "paper") -> np.ndarray:
    """Net position :math:`D_i` of every bank.

    Positive means the bank is owed more than it owes.

    >>> from netmotion import Network
    >>> g = Network.from_edges(5, [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)])
    >>> bank_capital(g).tolist()
    [6.0, 6.0, -4.0, -4.0, -4.0]
    """
    return debt_matrix(network, convention=convention).sum(axis=1)


@sourced(
    Origin.PAPER_2012,
    pdf_pages="4-6",
    section="2.1 Step 4 / 2.2",
    document="paper",
    note=(
        "The Too-Big-To-Fail banks: those whose net position is far above the "
        "rest. The thesis's fig. 84 (PDF p. 75) reports roughly one per "
        "community after a long growth run."
    ),
    reading="A cut-off of sigma standard deviations above the mean marks them.",
)
def too_big_to_fail(capital: np.ndarray, *, sigma: float = 2.0) -> np.ndarray:
    """Indices of the banks whose capital exceeds the mean by *sigma* deviations."""
    capital = np.asarray(capital, dtype=float)
    if capital.size == 0:
        return np.array([], dtype=int)
    cut = capital.mean() + sigma * capital.std()
    return np.flatnonzero(capital > cut)


@sourced(
    Origin.PAPER_2012,
    pdf_pages="4-6",
    section="2.1-2.2 The network model",
    document="paper",
    note=(
        "A grown community network read as an interbank system: community = "
        "country, node = bank, link weight = debt. Holds the network, its "
        "weight matrix W and the net position D of every bank. In the thesis, "
        "PDF p. 68."
    ),
)
@dataclass
class InterbankNetwork:
    """A grown community network read as an interbank system.

    Attributes
    ----------
    communities:
        The underlying network and its community labels.  A community is a
        country; a node is a bank.
    weights:
        The antisymmetric matrix :math:`W`; ``weights[i, j]`` is what bank *j*
        owes bank *i* under the article's convention.
    capital:
        :math:`D_i`, the net position of each bank.
    """

    communities: CommunityNetwork
    weights: np.ndarray
    capital: np.ndarray
    convention: str = "paper"

    @classmethod
    def from_community_network(
        cls, cn: CommunityNetwork, *, convention: str = "paper"
    ) -> InterbankNetwork:
        """Attach debts to a grown community network."""
        w = debt_matrix(cn.network, convention=convention)
        return cls(communities=cn, weights=w, capital=w.sum(axis=1), convention=convention)

    @property
    def network(self) -> Network:
        return self.communities.network

    @property
    def n_banks(self) -> int:
        return self.communities.network.n

    @property
    def country(self) -> np.ndarray:
        """Which country each bank belongs to."""
        return self.communities.community

    def exposure(self, i: int, j: int) -> float:
        """What bank *j* owes bank *i*."""
        return float(self.weights[i, j])

    def creditors(self) -> np.ndarray:
        """Banks with a positive net position."""
        return np.flatnonzero(self.capital > 0)

    def debtors(self) -> np.ndarray:
        """Banks with a negative net position."""
        return np.flatnonzero(self.capital < 0)

    def largest_bank(self) -> int:
        """The bank with the greatest net position: the usual cascade seed."""
        return int(np.argmax(self.capital))

    def too_big_to_fail(self, sigma: float = 2.0) -> np.ndarray:
        return too_big_to_fail(self.capital, sigma=sigma)

    def check_conservation(self, tol: float = 1e-9) -> bool:
        """Verify :math:`\\sum_i D_i = 0`, stated by the thesis on PDF p. 70.

        Antisymmetry makes this exact.
        """
        return bool(abs(self.capital.sum()) < tol)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<InterbankNetwork banks={self.n_banks} "
            f"countries={self.communities.n_communities} "
            f"creditors={self.creditors().size} debtors={self.debtors().size}>"
        )
