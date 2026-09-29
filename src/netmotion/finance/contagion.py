"""Financial contagion: the crash of one bank travelling along its debts.

Thesis section 5.6.4 (PDF pp. 69-71) and section 2.2 of the 2012 IJMPC article.

The thesis calls it a *virus financiero*.  A bank is attacked -- it bought
something toxic, and does not yet know -- and it fails.  Everyone who was doing
business with it now cannot collect, so each neighbour loses an amount set by
their debt relationship.  A neighbour whose capital cannot absorb that loss
fails in turn, and the failure keeps travelling.

Both the thesis's cascade and the article's are implemented.

The thesis cascade: two states
------------------------------
:math:`S_i(t) \\in \\{+1, -1\\}`, healthy or crashed (PDF p. 69):

.. math::
    D_i(t+1) &= D_i(t) - \\Theta_{ij} \\quad \\text{if } \\exists j / M_{ij} \\neq 0
        \\text{ and } S_j(t) = -1 \\\\
    S_i(t+1) &= -1 \\ \\text{ if } D_i(t+1) < 0 \\text{ and a neighbour is crashed}

with :math:`\\Theta_{ij} = W_{ji} = k_i - k_j` if :math:`k_i > k_j` and
:math:`W_{ij} = k_j - k_i` if :math:`k_j > k_i`, that is
:math:`|k_i - k_j|`.  The bank uses its capital to cushion the fall of its
neighbour.  :func:`two_state_cascade` takes the capital vector from the caller.

The article cascade: four states
--------------------------------
:math:`S_i(t) \\in \\{+2, +1, -1, -2\\}`: unaffected, affected-but-not-reacting,
affected, crashed, with reserves :math:`\\eta` for the net borrowers (Sec. 2.2),
a reaction threshold :math:`th_i = D_i(1) r / 100` (Eq. 9), affected banks
calling in what smaller neighbours owe them, weakest connection first
(Eqs. 6-7), and crashed banks draining their neighbours by :math:`|k_i - k_j|`
(Eq. 8).  :func:`four_state_cascade` takes the signed net position :math:`D_i`, or
the net borrowers through ``borrowers=``.

Readings taken
--------------
A crashed bank drains its neighbours once, when it fails (``drain="once"``,
the default), or at every later step (``drain="every_step"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced

__all__ = [
    "BankState",
    "CascadeRun",
    "two_state_cascade",
    "four_state_cascade",
    "systemic_impact",
]


@sourced(
    Origin.PAPER_2012,
    pdf_pages="6",
    section="2.2 The contagion model",
    document="paper",
    note=(
        "The article's four states. The thesis's two states, +1 and -1 "
        "(thesis PDF p. 69), are UNAFFECTED and CRASHED."
    ),
)
class BankState(IntEnum):
    """The four states of the 2012 article.

    ``UNAFFECTED`` and ``CRASHED`` correspond to the thesis's :math:`+1` and
    :math:`-1`.
    """

    UNAFFECTED = 2
    NOT_REACTED = 1
    AFFECTED = -1
    CRASHED = -2


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="69-71",
    section="5.6.4 Aplicacion de red formada por comunidades. Red de deuda del sistema bancario",
    document="both",
    note=(
        "The record of one cascade. survived_fraction is the quantity the "
        "thesis tabulates per network type on PDF p. 71; capital_history is "
        "what its snapshot figures 82 and 86 are drawn from."
    ),
)
@dataclass
class CascadeRun:
    """The history of one contagion.

    Attributes
    ----------
    first_to_fail:
        The bank that was attacked.
    state:
        Final state of every bank.
    capital:
        Final capital of every bank.
    history:
        ``history[t]`` is the fraction of banks crashed at step *t*.
    capital_history:
        ``capital_history[t]`` is the capital vector at step *t*.
    crashed_history:
        ``crashed_history[t]`` marks the banks crashed at step *t*.
    """

    first_to_fail: int
    state: np.ndarray
    capital: np.ndarray
    initial_capital: np.ndarray
    history: np.ndarray
    capital_history: list[np.ndarray] = field(default_factory=list)
    crashed_history: list[np.ndarray] = field(default_factory=list)

    @property
    def crashed(self) -> np.ndarray:
        """Indices of the banks that failed."""
        return np.flatnonzero(self.state == BankState.CRASHED)

    @property
    def crashed_fraction(self) -> float:
        """The headline number: what fraction of the system went down."""
        return float(self.history[-1])

    @property
    def survived_fraction(self) -> float:
        """The thesis reports this one, per network type, on PDF p. 71."""
        return 1.0 - self.crashed_fraction

    @property
    def steps(self) -> int:
        return int(self.history.size - 1)

    def capital_lost(self) -> float:
        """Total capital destroyed, across every bank."""
        return float(np.sum(self.initial_capital - self.capital))


@sourced(
    Origin.THESIS_2011,
    pdf_pages="69-71",
    section="5.6.4 Aplicacion de red formada por comunidades. Red de deuda del sistema bancario",
    note=(
        "The two-state cascade of PDF p. 69: a bank adjacent to a crashed bank "
        "loses Theta_ij = |k_i - k_j| from its capital and crashes if what "
        "remains is negative."
    ),
    reading=(
        "A crashed bank drains its neighbours once (drain='once', the default) "
        "or at every later step (drain='every_step')."
    ),
    tags=("chapter-5", "cascade", "drain-readings"),
)
def two_state_cascade(
    network: Network,
    first_to_fail: int,
    capital: np.ndarray,
    *,
    drain: str = "once",
    max_steps: int = 1000,
) -> CascadeRun:
    """Run the 2011 two-state contagion from the failure of the bank *first_to_fail*.

    Parameters
    ----------
    capital:
        Starting capital :math:`D_i` of every bank, from
        :func:`~netmotion.finance.interbank.bank_capital`.
    drain:
        ``"once"`` or ``"every_step"``; see the module docstring.

    Examples
    --------
    A hub next to three small banks. The hub crashes; each small bank loses
    :math:`|k_i - k_j|` and fails if that takes it below zero.

    >>> from netmotion import Network
    >>> from netmotion.finance.interbank import bank_capital
    >>> star = Network.from_edges(4, [(0,1),(0,2),(0,3)])
    >>> run = two_state_cascade(star, 0, bank_capital(star))
    >>> run.crashed.tolist()
    [0, 1, 2, 3]
    """
    seed = first_to_fail
    if drain not in ("once", "every_step"):
        raise ValueError(f"drain must be 'once' or 'every_step', got {drain!r}")

    n = network.n
    a = network.adjacency
    k = a.sum(axis=1)
    theta = np.abs(k[:, None] - k[None, :]) * a

    d = np.asarray(capital, dtype=float).copy()
    initial = d.copy()
    state = np.full(n, int(BankState.UNAFFECTED), dtype=int)
    state[seed] = int(BankState.CRASHED)

    history = [1.0 / n]
    capital_history = [d.copy()]
    crashed_history = [state == int(BankState.CRASHED)]
    already_drained = np.zeros(n, dtype=bool)

    for _ in range(max_steps):
        crashed = state == int(BankState.CRASHED)
        sources = crashed if drain == "every_step" else (crashed & ~already_drained)
        if not sources.any():
            break

        # Every solvent bank adjacent to a draining failure loses Theta.  A bank
        # that has already crashed has nothing further to lose.
        loss = theta[:, sources].sum(axis=1)
        loss[crashed] = 0.0
        d = d - loss
        already_drained |= sources

        # The condition of PDF p. 69 is D < 0 *and* a crashed neighbour: a bank
        # left with exactly zero capital has absorbed the loss, and a bank in
        # debt falls only once the contagion reaches it.
        touched = (a[:, crashed] != 0).any(axis=1)
        newly = (d < 0) & touched & ~crashed
        if not (loss > 0).any() and not newly.any():
            break
        state[(loss > 0) & ~crashed & (state == int(BankState.UNAFFECTED))] = int(
            BankState.AFFECTED
        )
        state[newly] = int(BankState.CRASHED)

        capital_history.append(d.copy())
        crashed_history.append(state == int(BankState.CRASHED))
        history.append(float((state == int(BankState.CRASHED)).sum() / n))

        # Under the one-shot reading the cascade advances only through fresh
        # failures, so no new failure means it is over.
        if drain == "once" and not newly.any():
            break

    return CascadeRun(
        first_to_fail=seed,
        state=state,
        capital=d,
        initial_capital=initial,
        history=np.asarray(history, dtype=float),
        capital_history=capital_history,
        crashed_history=crashed_history,
    )


@sourced(
    Origin.PAPER_2012,
    pdf_pages="6-8",
    section="2.2 The contagion model",
    document="paper",
    note=(
        "The four-state cascade: reserves eta applied to the net borrowers "
        "(Sec. 2.2), reaction threshold r (Eq. 9), partial recovery by affected "
        "banks calling in what smaller neighbours owe them, weakest connection "
        "first (Eqs. 6-7), and the crashed-bank drain of Eq. 8."
    ),
    reading=(
        "A crashed bank drains its neighbours once (drain='once', the default) "
        "or at every later step. Affected banks act in node order within a "
        "step. Where no T_ij is supplied, a link dates from the later of its "
        "two endpoints, T_ij = n - max(i, j); connection_age= overrides it."
    ),
    tags=("cascade", "drain-readings", "reserves-apply-to-borrowers"),
)
def four_state_cascade(
    network: Network,
    first_to_fail: int,
    capital: np.ndarray,
    *,
    reserves: float = 100.0,
    borrowers: np.ndarray | None = None,
    reaction_threshold: float = 0.0,
    threshold_base: str = "initial",
    drain: str = "once",
    connection_age: np.ndarray | None = None,
    max_steps: int = 1000,
) -> CascadeRun:
    """Run the 2012 four-state contagion from the failure of the bank *first_to_fail*.

    Parameters
    ----------
    capital:
        The *signed* net position :math:`D_i` of every bank, from
        :func:`~netmotion.finance.interbank.bank_capital`.  The sign is read,
        not ignored: it is what decides who the reserves rule applies to.  A
        caller holding only :math:`|D_i|` must say who the borrowers are with
        *borrowers*.
    reserves:
        :math:`\\eta`, a percentage.  A bank with :math:`D_i < 0` -- a net
        borrower -- enters the cascade with :math:`D_i(1) = \\eta |D_i| / 100`;
        a bank with :math:`D_i > 0` is a creditor and enters with :math:`|D_i|`
        in full (Sec. 2.2).  At :math:`\\eta = 100` the two coincide and every
        bank holds the absolute value of its net position.
    borrowers:
        Optional boolean mask naming the net borrowers, for capital vectors
        whose sign has already been discarded.  Overrides the sign of *capital*.
    reaction_threshold:
        :math:`r`, a percentage.  A bank reacts only once its losses exceed
        :math:`th_i = D_i(1) r / 100` (Eq. 9).  At :math:`r = 0` every bank
        reacts to any loss at all.
    threshold_base:
        Which :math:`D_i` Eq. 9 is read against.  ``"initial"`` (the default)
        is the equation as printed, :math:`D_i(1)`, so a borrower's threshold
        is scaled by :math:`\\eta` along with its money.  ``"gross"`` takes the
        unscaled :math:`|D_i|` instead, the reading on which :math:`r` measures
        loss against the whole position.
    connection_age:
        :math:`T_{ij}`, how long banks *i* and *j* have been connected.  An
        affected bank calls in its weakest connection first, which is its
        *smallest* :math:`T_{ij}` and so its newest link.  Defaults to arrival
        order: a link dates from the later of its endpoints, giving
        :math:`T_{ij} = n - \\max(i, j)`.

    Examples
    --------
    With full reserves and no reaction threshold, the hub's crash damages its
    smaller neighbours without necessarily destroying them:

    >>> from netmotion import Network
    >>> from netmotion.finance.interbank import bank_capital
    >>> g = Network.from_edges(5, [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)])
    >>> run = four_state_cascade(g, 0, bank_capital(g))
    >>> run.crashed.tolist()
    [0]
    >>> int((run.state == BankState.AFFECTED).sum())
    3

    The reserves haircut falls on the borrowers alone.  Banks 0 and 1 are the
    creditors here, so at :math:`\\eta = 20` they still start at 6, while the
    three borrowers start at 20% of 4:

    >>> run = four_state_cascade(g, 0, bank_capital(g), reserves=20.0)
    >>> run.initial_capital.tolist()
    [6.0, 6.0, 0.8, 0.8, 0.8]
    """
    seed = first_to_fail
    if drain not in ("once", "every_step"):
        raise ValueError(f"drain must be 'once' or 'every_step', got {drain!r}")
    if threshold_base not in ("initial", "gross"):
        raise ValueError(f"threshold_base must be 'initial' or 'gross', got {threshold_base!r}")

    n = network.n
    a = network.adjacency
    k = a.sum(axis=1)

    # Sec. 2.2: eta is a haircut on the net borrowers, the banks with D_i < 0.
    # A creditor is not a borrower and brings |D_i| to the cascade untouched.
    signed = np.asarray(capital, dtype=float)
    gross = np.abs(signed)
    if borrowers is None:
        if reserves != 100.0 and not np.any(signed < 0.0):
            raise ValueError(
                "four_state_cascade needs the signed net position D_i: the reserves "
                "rule of Sec. 2.2 applies only to banks with D_i < 0, and no "
                "entry of this capital vector is negative. Pass the signed "
                "vector from bank_capital(), or name the net borrowers with "
                "borrowers=."
            )
        is_borrower = signed < 0.0
    else:
        is_borrower = np.asarray(borrowers, dtype=bool)
        if is_borrower.shape != signed.shape:
            raise ValueError(f"borrowers has shape {is_borrower.shape}, capital has {signed.shape}")

    d1 = gross.copy()
    d1[is_borrower] *= reserves / 100.0
    d = d1.copy()

    # Eq. 9, against D_i(1); threshold_base='gross' uses |D_i|.
    base = d1 if threshold_base == "initial" else gross
    threshold = base * reaction_threshold / 100.0

    if connection_age is not None:
        age = np.asarray(connection_age, dtype=float)
    else:
        arrival = np.arange(n, dtype=float)
        age = float(n) - np.maximum(arrival[:, None], arrival[None, :])

    state = np.full(n, int(BankState.UNAFFECTED), dtype=int)
    state[seed] = int(BankState.CRASHED)
    d[seed] = 0.0

    history = [1.0 / n]
    capital_history = [d.copy()]
    crashed_history = [state == int(BankState.CRASHED)]
    already_drained = np.zeros(n, dtype=bool)
    # What each debtor has already repaid on each loan: a loan is called in once,
    # not again at every instant the lender's loss lasts.
    repaid = np.zeros((n, n))

    for _ in range(max_steps):
        previous_state = state.copy()
        previous_capital = d.copy()

        # Eq. 8: a crashed bank drains every neighbour by |k_i - k_j|.
        crashed = state == int(BankState.CRASHED)
        sources = crashed if drain == "every_step" else (crashed & ~already_drained)
        if sources.any():
            d = d - (np.abs(k[:, None] - k[None, :]) * a)[:, sources].sum(axis=1)
            already_drained |= sources

        # Eqs. 6-7: an affected bank calls in what smaller neighbours owe it,
        # recovering part of its own loss and passing the damage on.  It asks
        # the weakest connection first -- the smallest T_ij, i.e. the link it
        # has held for the shortest time.  Affected banks act in node order.
        for j in np.flatnonzero(state == int(BankState.AFFECTED)):
            shortfall = d1[j] - d[j]
            if shortfall <= 0:
                continue
            # Sec. 2.2: a crashed bank is not available to repay its debts.
            targets = np.flatnonzero((a[j] > 0) & (k[j] > k) & (state != int(BankState.CRASHED)))
            for i in targets[np.argsort(age[j, targets], kind="stable")]:
                claim = k[j] - k[i] - repaid[j, i]
                paid = min(shortfall, claim)
                if paid <= 0:
                    continue
                d[i] -= paid
                d[j] += paid
                repaid[j, i] += paid
                shortfall -= paid
                if shortfall <= 0:
                    break

        # The state rules of Sec. 2.2: reclassify by how much capital each bank has left.
        loss = d1 - d
        new_state = state.copy()
        unresolved = state != int(BankState.CRASHED)
        # A bank crashes when the contagion has taken its money to zero; a bank
        # the contagion has not reached keeps its state, even one that entered
        # with nothing (D_i = 0, or a borrower at eta = 0).
        new_state[unresolved & (d <= 0) & (loss > 0)] = int(BankState.CRASHED)
        still = unresolved & ~((d <= 0) & (loss > 0))
        new_state[still & (loss <= 0)] = int(BankState.UNAFFECTED)
        new_state[still & (loss > 0) & (loss < threshold)] = int(BankState.NOT_REACTED)
        new_state[still & (loss > 0) & (loss >= threshold)] = int(BankState.AFFECTED)
        state = new_state

        capital_history.append(d.copy())
        crashed_history.append(state == int(BankState.CRASHED))
        history.append(float((state == int(BankState.CRASHED)).sum() / n))

        settled = np.array_equal(state, previous_state) and np.allclose(d, previous_capital)
        if settled:
            break

    return CascadeRun(
        first_to_fail=seed,
        state=state,
        capital=d,
        initial_capital=d1,
        history=np.asarray(history, dtype=float),
        capital_history=capital_history,
        crashed_history=crashed_history,
    )


@sourced(
    Origin.PAPER_2012,
    pdf_pages="11",
    section="3.2 The contagion model results (Eq. 10)",
    document="paper",
    note=(
        "I_c = D_c, the capital of the bank that was attacked, and "
        "R = sum over every other bank. Their ratio is how large the shock was "
        "relative to the system that had to absorb it."
    ),
)
def systemic_impact(capital: np.ndarray, first_to_fail: int) -> tuple[float, float, float]:
    """Return ``(I_c, R, I_c / R)`` for a cascade started by *first_to_fail*.

    >>> import numpy as np
    >>> systemic_impact(np.array([6.0, 6.0, 4.0, 4.0, 4.0]), 0)
    (6.0, 18.0, 0.3333333333333333)
    """
    seed = first_to_fail
    capital = np.asarray(capital, dtype=float)
    impact = float(capital[seed])
    rest = float(capital.sum() - capital[seed])
    return impact, rest, (impact / rest if rest else float("inf"))
