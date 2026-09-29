"""Networks made of communities.

Thesis section 5.6 (PDF pp. 64-74) and section 2.1 of the 2012 IJMPC article.

Real networks are not homogeneous.  They come in groups that are dense inside
and sparse between, and the thesis builds a generator in which that structure
is explicit: a fixed number of communities, each growing by preferential
attachment *within* itself, while new nodes also reach *across* to other
communities with a probability that depends on how attractive those communities
are.

The result is scale-free twice over -- inside a community and, through the
exterior links, between them -- and it is the substrate the interbank model of
:mod:`netmotion.finance` is built on, where a community is a country and a node
is a bank.

Two attractiveness kernels
--------------------------
The interior kernel is ordinary preferential attachment, restricted to the
node's own community (thesis p. 66, article Eq. 2):

.. math:: \\Pi_{mi} = \\frac{k_{mi}}{\\sum_{h=1}^{n_m} k_{mh}}

Section 5.6.2 weights a community by its **exterior degree** -- how many links
already leave it:

.. math:: \\Pi C_i = \\frac{KC_i}{\\sum_h KC_h}

while strategy 1 of section 5.6.4 and the article (Eq. 3) weight it by
its **size**:

.. math:: P_{m'} = \\frac{n_m}{\\sum_h n_h}

Both are implemented and ``exterior_kernel`` selects between them; the default
follows section 5.6.2.

Readings taken
--------------
*The r exterior links of step 0.*  ``seed_scope="per_community"`` (the default)
gives every community an exterior link before growth starts, so every community
has a non-zero exterior degree and receives nodes, as in fig. 74.a;
``seed_scope="total"`` places *r* links in total.

*The link counts of step 3.*  Three readings are implemented:

======================  ====================================================
``attachment``          reading
======================  ====================================================
``"biased_interior"``   the *n* interior links are made; each exterior link
                        is a Bernoulli trial against the kernel (default)
``"sweep"``             both are a single bounded sweep
``"until_filled"``      both are retried until filled
======================  ====================================================

The default gives the minimum degree of 2 of fig. 74.b.  The exponent of
fig. 74.c is a least-squares line through the points it plots, *k* = 3 to 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng

__all__ = ["CommunityNetwork", "community_scale_free", "modularity"]

_MAX_REJECTIONS = 100_000


@dataclass
class CommunityNetwork:
    """A network together with the community each node belongs to.

    Attributes
    ----------
    network:
        The network itself.
    community:
        ``community[i]`` is the index of the community node *i* belongs to.
    """

    network: Network
    community: np.ndarray

    @property
    def n_communities(self) -> int:
        return int(self.community.max()) + 1 if self.community.size else 0

    def members(self, c: int) -> np.ndarray:
        """Node indices belonging to community *c*."""
        return np.flatnonzero(self.community == c)

    def sizes(self) -> np.ndarray:
        """Number of nodes in each community."""
        return np.bincount(self.community, minlength=self.n_communities)

    def interior_edges(self) -> int:
        """Links whose two endpoints lie in the same community."""
        return sum(1 for i, j, _ in self.network.edges() if self.community[i] == self.community[j])

    def exterior_edges(self) -> int:
        """Links joining two different communities."""
        return self.network.n_edges - self.interior_edges()

    def exterior_degree(self) -> np.ndarray:
        """:math:`KC_h`: links leaving each community.

        This is the thesis's measure of how attractive a community is.
        """
        counts = np.zeros(self.n_communities, dtype=int)
        for i, j, _ in self.network.edges():
            ci, cj = self.community[i], self.community[j]
            if ci != cj:
                counts[ci] += 1
                counts[cj] += 1
        return counts

    @property
    def density(self) -> float:
        """Links per node, the quantity the thesis calls *d*.

        Reported as 2.06 for the network of PDF p. 66 and 2.46 for the
        seven-strategy network of PDF p. 74.
        """
        return self.network.n_edges / self.network.n if self.network.n else 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<CommunityNetwork N={self.network.n} E={self.network.n_edges} "
            f"communities={self.n_communities} d={self.density:.2f}>"
        )


@sourced(
    Origin.THESIS_2011,
    pdf_pages="65-66",
    section="5.6.2 Red Scale Free de comunidades",
    document="both",
    note=(
        "Growth with two preferential-attachment kernels: interior attachment "
        "within the node's own community, exterior attachment to other "
        "communities chosen by attractiveness. Reproduces the network of PDF "
        "p. 66: Cn=10, n0=3, r=1, n=2, m=1, t=100, giving N=130."
    ),
    reading=(
        "(1) The exterior kernel is the community exterior degree KC in section "
        "5.6.2 and community size in section 5.6.4 and the 2012 article; both "
        "are implemented and selected by exterior_kernel. (2) The n0 nodes of a "
        "community start as a complete graph, as in the article's Step 0. (3) "
        "Attachment against Pi_i is implemented as accept-reject sampling. (4) r is counted per community by default. "
        "(5) Three attachment readings are implemented; 'biased_interior' is the "
        "default."
    ),
    tags=("chapter-5", "interbank-substrate"),
)
def community_scale_free(
    n_communities: int = 10,
    *,
    initial_nodes_per_community: int = 3,
    exterior_seed_links: int = 1,
    interior_links: int = 2,
    exterior_links: int = 1,
    steps: int = 100,
    exterior_kernel: str = "exterior_degree",
    seed_scope: str = "per_community",
    attachment: str = "biased_interior",
    exterior_attempts: int | None = 3,
    rng: SeedLike = None,
    name: str = "",
) -> CommunityNetwork:
    """Grow a scale-free network of communities.

    Parameters follow the thesis's notation: ``n_communities`` is *Cn*,
    ``initial_nodes_per_community`` is :math:`n_0`, ``exterior_seed_links`` is *r*,
    ``interior_links`` is *n*, ``exterior_links`` is *m* and ``steps`` is *t*.
    The defaults are the values used for the network of PDF p. 66.

    Parameters
    ----------
    exterior_kernel:
        ``"exterior_degree"``
            Weight a community by :math:`KC_h`, the number of links already
            leaving it.  Thesis section 5.6.2, and the default.
        ``"size"``
            Weight a community by how many nodes it has.  Thesis section 5.6.4
            and article Eq. 3.
    seed_scope:
        Whether ``exterior_seed_links`` (*r*) is counted ``"per_community"``
        (the default; see the module docstring) or in ``"total"``.
    attachment:
        ``"biased_interior"`` (default), ``"sweep"`` or ``"until_filled"``.
        See the table in the module docstring.
    exterior_attempts:
        How many times an arriving node tries for its exterior link, each try
        drawing a community and accepting it with probability
        :math:`\\Pi C`.  Three, the default, gives the densities the thesis
        reports for this model: 2.06 links per node at t = 100 (PDF p. 66)
        and 2.16 at t = 400 (PDF p. 67).  ``None`` tries once per community.

    Returns
    -------
    CommunityNetwork
        Of ``n_communities * n0 + steps`` nodes.

    Examples
    --------
    The thesis's own parameters give 130 nodes in ten communities:

    >>> cn = community_scale_free(rng=20110930)
    >>> cn.network.n, cn.n_communities
    (130, 10)

    Nine of the ten communities take arrivals during the run; the other one
    is never chosen and keeps its three seed nodes:

    >>> int((cn.sizes() > 3).sum())
    9

    Communities are dense inside relative to between, which is what makes them
    communities:

    >>> cn.interior_edges() > cn.exterior_edges()
    True

    Counting *r* as a total leaves most communities at their initial three
    nodes:

    >>> frozen = community_scale_free(rng=20110930, seed_scope="total")
    >>> int((frozen.sizes() == 3).sum())
    8
    """
    n0 = initial_nodes_per_community
    if exterior_kernel not in ("exterior_degree", "size"):
        raise ValueError(
            f"exterior_kernel must be 'exterior_degree' or 'size', got {exterior_kernel!r}"
        )
    if interior_links >= n0:
        raise ValueError(
            f"interior_links={interior_links} must be smaller than n0={n0}: a new node "
            "cannot attach to more distinct nodes than its community started with"
        )
    if n_communities < 2 and exterior_links:
        raise ValueError("exterior links need at least two communities")

    generator = as_rng(rng)
    total = n_communities * n0 + steps
    matrix = np.zeros((total, total))
    community = np.zeros(total, dtype=int)

    # Step 0: each community starts as a complete graph on n0 nodes.
    for c in range(n_communities):
        members = np.arange(c * n0, (c + 1) * n0)
        community[members] = c
        for a_idx, a in enumerate(members):
            for b in members[a_idx + 1 :]:
                matrix[a, b] = matrix[b, a] = 1.0

    seeded = n_communities * n0

    # Step 0 (continued): the r exterior links. Every community needs a non-zero
    # exterior degree before growth starts, or the kernel locks it out for ever.
    if seed_scope == "per_community":
        pairs = [
            (c, int(generator.choice([h for h in range(n_communities) if h != c])))
            for c in range(n_communities)
            for _ in range(exterior_seed_links)
        ]
    elif seed_scope == "total":
        pairs = [
            tuple(generator.choice(n_communities, size=2, replace=False))
            for _ in range(exterior_seed_links)
        ]
    else:
        raise ValueError(f"seed_scope must be 'per_community' or 'total', got {seed_scope!r}")

    for ca, cb in pairs:
        a = int(generator.choice(np.flatnonzero(community[:seeded] == ca)))
        b = int(generator.choice(np.flatnonzero(community[:seeded] == cb)))
        matrix[a, b] = matrix[b, a] = 1.0

    def exterior_degrees(count: int) -> np.ndarray:
        kc = np.zeros(n_communities)
        rows, cols = np.nonzero(matrix[:count, :count])
        for i, j in zip(rows, cols, strict=True):
            if i < j and community[i] != community[j]:
                kc[community[i]] += 1
                kc[community[j]] += 1
        return kc

    def community_weights(count: int) -> np.ndarray:
        if exterior_kernel == "size":
            weights = np.bincount(community[:count], minlength=n_communities).astype(float)
        else:
            weights = exterior_degrees(count)
        total_weight = weights.sum()
        if total_weight <= 0:
            raise RuntimeError(
                "every community has zero exterior degree, so the rejection "
                "sampler can never accept; increase exterior_seed_links (r)"
            )
        return weights / total_weight

    def accept_reject(
        candidates: np.ndarray,
        probabilities: np.ndarray,
        wanted: int,
        *,
        interior: bool = True,
    ) -> list[int]:
        """Draw uniformly, accept with probability Pi.

        ``attachment="sweep"`` makes one pass over the candidates, so a link is
        *attempted* ``wanted`` times rather than guaranteed ``wanted`` times.
        That is the structure of the attachment sweep of the scale-free
        generator of PDF pp. 27-29.
        """
        chosen: list[int] = []
        bounded = attachment == "sweep" or (attachment == "biased_interior" and not interior)
        limit = candidates.size if bounded else _MAX_REJECTIONS
        for attempt in range(limit):
            if len(chosen) >= wanted:
                break
            if attachment != "sweep" and attempt == _MAX_REJECTIONS - 1:
                raise RuntimeError(
                    "accept-reject sampling failed to find enough partners; the "
                    "attachment probabilities are degenerate"
                )
            pick = int(generator.integers(0, candidates.size))
            node = int(candidates[pick])
            if node in chosen:
                continue
            if generator.random() < probabilities[pick]:
                chosen.append(node)
        return chosen

    for step in range(steps):
        count = seeded + step
        pc = community_weights(count)

        # Step 2: choose a community by rejection sampling against its attractiveness.
        for _attempt in range(_MAX_REJECTIONS):
            candidate = int(generator.integers(0, n_communities))
            if generator.random() < pc[candidate]:
                target_community = candidate
                break
        else:  # pragma: no cover - guarded by community_weights
            raise RuntimeError("could not select a community; attractiveness is degenerate")

        new = count
        community[new] = target_community

        # Step 3: interior attachment, preferential within the community.
        members = np.flatnonzero(community[:count] == target_community)
        degrees = matrix[:count, :count].sum(axis=1)[members]
        interior_pi = degrees / degrees.sum() if degrees.sum() > 0 else np.full(members.size, 1.0)
        for node in accept_reject(
            members, interior_pi, min(interior_links, members.size), interior=True
        ):
            matrix[new, node] = matrix[node, new] = 1.0

        # Step 3 (continued): exterior attachment, preferential over communities
        # and then uniform inside the chosen one.
        attached: set[int] = set()
        bounded_exterior = attachment in ("sweep", "biased_interior")
        limit = n_communities if bounded_exterior else _MAX_REJECTIONS
        if exterior_attempts is not None:
            limit = exterior_attempts
        for _ in range(limit):
            if len(attached) >= exterior_links:
                break
            other = int(generator.integers(0, n_communities))
            if other == target_community:
                continue
            if generator.random() >= pc[other]:
                continue
            pool = np.flatnonzero(community[:count] == other)
            if pool.size == 0:
                continue
            partner = int(generator.choice(pool))
            if partner in attached:
                continue
            attached.add(partner)
            matrix[new, partner] = matrix[partner, new] = 1.0

    net = Network(
        matrix,
        name=name or f"SF-communities Cn={n_communities} t={steps}",
    )
    return CommunityNetwork(network=net, community=community)


@sourced(
    Origin.EXTENSION,
    pdf_pages="64-67",
    section="5.6.1 Definiciones, ejemplos y caracteristicas",
    note=(
        "2026 extension, alongside the dispersion of propagation curves that "
        "section 5.6.3 uses to detect community structure."
    ),
    tags=("modern-comparison",),
)
def modularity(community_network: CommunityNetwork) -> float:
    """Newman-Girvan modularity :math:`Q` of the given partition.

    >>> cn = community_scale_free(rng=20110930)
    >>> bool(modularity(cn) > 0.3)
    True
    """
    cn = community_network
    a = cn.network.adjacency
    m2 = a.sum()
    if m2 == 0:
        return 0.0
    k = a.sum(axis=1)
    same = cn.community[:, None] == cn.community[None, :]
    expected = np.outer(k, k) / m2
    return float(((a - expected) * same).sum() / m2)
