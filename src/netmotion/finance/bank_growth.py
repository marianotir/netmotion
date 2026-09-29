"""How banks choose whom to lend to: the growth strategies.

Thesis section 5.6.4 (PDF pp. 72-77) and section 2.1 of the 2012 IJMPC article.

The community generator of :mod:`netmotion.networks.communities` grows every
node the same way.  In the model of growth by strategies each arriving bank is
given a **strategy**, drawn at random, and the strategy decides how many links
it makes and how it picks its partners.  Seven strategies in the thesis, three
in the article.  The model produces the banks known as too big to fail.

The seven strategies of the thesis
-----------------------------------
Each is described on PDF pp. 72-73 in terms of the same quantities: the
interior attachment kernel :math:`\\Pi_i = k_i / \\sum_{h \\in c} k_h`, the
community-size kernel :math:`P_m`, the community-attractiveness kernel
:math:`\\Pi C_h = KC_h / \\sum_g KC_g`, and averages of those over a community
or over the whole network.

=========================  ==================================================
:attr:`Strategy.SAFE`      Attach to the largest bank in the community, plus
                           two more chosen preferentially; abroad, attach in
                           the largest community.  The rule of a bank that
                           wants a partner who will not fail.
:attr:`Strategy.IMITATE`   Do what the average bank in the community does:
                           accept a partner with probability
                           :math:`\\langle \\Pi \\rangle`, and make
                           :math:`\\langle k \\rangle` links.
:attr:`Strategy.CAUTIOUS`  As IMITATE, but two links fewer.
:attr:`Strategy.RISKY`     Two links *more*, and accept with probability
                           :math:`1 - \\langle \\Pi \\rangle`.
:attr:`Strategy.GLOBAL`    As IMITATE, but averaged over the whole network
                           rather than over the bank's own community.
:attr:`Strategy.RANDOM`    Two interior and one exterior partner, uniformly,
                           with no kernel at all.
:attr:`Strategy.DOMESTIC`  Interior links only; never lends abroad.
=========================  ==================================================

Restricted growth
-----------------
Thesis PDF pp. 76-77 adds one condition: a node can only make connections with
exterior nodes *that have positive debt*, :math:`D_i > 0`.  Here :math:`D_i` is
computed from the degrees at that moment.  With this restriction the contagion reaches about 10%
of the network, against about 80% without it, and only one community is
affected.  :func:`strategy_growth` implements it as ``restricted=True``, and
:func:`confidence_strategy_growth` carries the flag too, for the article's own
statement of the restriction on p. 15.

A refused attempt
-----------------
Two readings of what an arriving bank does when the foreign bank it approached
has :math:`D_i \\le 0`:

``on_refusal="retry"``
    The condition constrains only the *choice*: the arrival keeps looking, in
    another community or at another bank, until it finds a solvent partner.
    The number of cross-border links is unchanged; only their endpoints move.
``on_refusal="drop"``
    The condition constrains the *link*: the approach is made as it would have
    been without the restriction, and if the bank approached has
    :math:`D_i \\le 0` the attempt is spent.  The arrival makes its interior
    links and goes without the foreign one.  Cross-border links then become
    scarce.

PDF p. 76 builds the restricted network with ``t = 1000`` because "la
restriccion impuesta para las conexiones exteriores dificulta el crecimiento de
las ramas entre comunidades".  Under ``"drop"`` the links between communities
grow more slowly, as that sentence says; ``"drop"`` is the default.

Readings taken
--------------
The readings of the seven rules are recorded in the ``reading`` field of
:func:`strategy_growth`.
"""

from __future__ import annotations

import enum

import numpy as np

from ..graph import Network
from ..networks.communities import CommunityNetwork
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng

__all__ = [
    "Strategy",
    "STRATEGY_DESCRIPTIONS",
    "strategy_growth",
    "confidence_strategy_growth",
    "net_position_from_degrees",
]

_MAX_TRIES = 20_000


class Strategy(enum.IntEnum):
    """The seven growth strategies of thesis PDF pp. 72-73, in its order."""

    SAFE = 1
    IMITATE = 2
    CAUTIOUS = 3
    RISKY = 4
    GLOBAL = 5
    RANDOM = 6
    DOMESTIC = 7


# The rules given plain integer counts: strategies 1 and 6 as the thesis states
# them ("a dos nodos interiores", "a dos nodos ... a un nodo"), and strategy 7
# ("union solo interior"), which is given the two uniform interior links of
# strategy 6 here -- as against the four that derive a count from a local mean.
_EXPLICIT_COUNTS = frozenset({Strategy.SAFE, Strategy.RANDOM, Strategy.DOMESTIC})


STRATEGY_DESCRIPTIONS: dict[Strategy, str] = {
    Strategy.SAFE: "attach to the largest bank, and in the largest community",
    Strategy.IMITATE: "do what the average bank in the community does",
    Strategy.CAUTIOUS: "as IMITATE, but two links fewer",
    Strategy.RISKY: "two links more, accepted with probability 1 - <Pi>",
    Strategy.GLOBAL: "as IMITATE, averaged over the whole network",
    Strategy.RANDOM: "two interior and one exterior partner, uniformly",
    Strategy.DOMESTIC: "interior links only; never lends abroad",
}


@sourced(
    Origin.PAPER_2012,
    pdf_pages="3",
    section="2.1 The network model",
    document="paper",
    note=(
        "D_i = k_i^2 - sum of neighbour degrees, the closed form of the node "
        "intensity. Computed from the degrees as they stand, which is what the "
        "restricted-growth condition of thesis PDF p. 76 requires."
    ),
)
def net_position_from_degrees(matrix: np.ndarray, count: int) -> np.ndarray:
    """Net position of the first *count* banks, from the current adjacency."""
    a = (matrix[:count, :count] != 0).astype(float)
    k = a.sum(axis=1)
    return k * k - a @ k


def _exterior_pools(
    community: np.ndarray,
    count: int,
    n_communities: int,
    target: int,
    exterior_p: np.ndarray,
    solvent: np.ndarray | None,
) -> dict[int, np.ndarray]:
    """The banks abroad that this arrival is allowed to lend to, by community.

    A community is excluded when it is the arrival's own, when the strategy
    gives it zero attachment probability, or when the restriction of thesis PDF
    p. 76 / article p. 15 leaves it with no solvent member.
    """
    pools: dict[int, np.ndarray] = {}
    for other in range(n_communities):
        if other == target or exterior_p[other] <= 0:
            continue
        pool = np.flatnonzero(community[:count] == other)
        if solvent is not None:
            pool = pool[solvent[pool]]
        if pool.size:
            pools[other] = pool
    return pools


@sourced(
    Origin.THESIS_2011,
    pdf_pages="72-77",
    section="5.6.4 Aplicacion de red formada por comunidades. Red de deuda del sistema bancario",
    note=(
        "Growth in which every arriving bank is given one of seven strategies "
        "at random, each deciding how many links it makes and how it chooses "
        "partners. restricted=True adds the condition of PDF p. 76: a foreign "
        "bank may be lent to only if its net position is positive."
    ),
    reading=(
        "(1) Strategy 2's exterior link count is round(<KC>), as for strategies "
        "3-5. (2) The count given under strategy 3's exterior link, <KC> - 2, "
        "sets its exterior links. (3) A "
        "number of communities is a number of exterior links, one per community. "
        "(4) Counts are at least 1 and at most 4, the bounds stated for "
        "strategies 2-5. (5) Strategy 1 joins the node with most links and, "
        "with probability Pi_i, two interior nodes, giving three interior links. "
        "(6) The candidates are swept once at the attachment probability and the "
        "sweep is topped up to the rule's floor -- the whole count for "
        "strategies 1, 6 and 7, and one link for strategies 2-5, 'o como minimo "
        "igual a 1'. (7) Strategy 1 joins 'la comunidad que tenga mas nodos' "
        "among the other communities. (8) Strategy 5's <Pi> is the mean over "
        "all N nodes of the kernel taken over the whole network, and its "
        "exterior link is accepted with <PiC>, as printed. (9) Under restricted=True an attempt at a "
        "bank with D_i <= 0 is spent (on_refusal='drop', the default) or "
        "repeated until a solvent bank is found (on_refusal='retry')."
    ),
    tags=("chapter-5", "interbank", "seven-strategies"),
)
def strategy_growth(
    n_communities: int = 10,
    *,
    founding_banks: int = 3,
    exterior_seed_links: int = 1,
    steps: int = 100,
    strategies: tuple[int, ...] | None = None,
    restricted: bool = False,
    on_refusal: str = "drop",
    restriction_sides: str = "lender",
    rng: SeedLike = None,
    name: str = "",
) -> tuple[CommunityNetwork, np.ndarray]:
    """Grow a community network where each arrival picks a strategy.

    Parameters follow the thesis's notation, and the defaults are the values of
    PDF p. 73: ``Cn = 10``, ``n0 = 3``, ``r = 1``, ``t = 100``.

    Parameters
    ----------
    strategies:
        Which of the seven to draw from.  Defaults to all seven, drawn at
        random as in the thesis (a uniform draw here).
    restricted:
        Apply the condition of PDF p. 76 -- an exterior link may only be made
        to a bank whose net position is currently positive.
    on_refusal:
        What an arrival does when the restriction turns it down; read only when
        *restricted* is true.  ``"drop"`` (the default) spends the attempt and
        the arrival goes without that foreign link, which is what makes the
        branches between communities hard to grow, as PDF p. 76 states;
        ``"retry"`` keeps looking for a solvent partner, so the link is made
        anyway.  See the module docstring.
    restriction_sides:
        Read only when *restricted* is true.  `"lender"` checks the bank
        lent to abroad; `"both"` also requires the arriving bank itself to
        have D > 0, after its interior links, so that every link between
        countries joins two banks in credit -- PDF p. 73: the contagion can
        pass to another community only through nodes with D > 0.

    Returns
    -------
    (CommunityNetwork, np.ndarray)
        The grown network, and the strategy each node was given (``0`` for the
        seed nodes, which arrived before any strategy was assigned).

    Examples
    --------
    The thesis's own parameters give the same 130 banks as the uniform
    generator, but a denser network, because several strategies make more than
    three links:

    >>> cn, assigned = strategy_growth(rng=20110930)
    >>> cn.network.n
    130
    >>> bool(cn.density > 2.4)
    True

    The restriction of PDF p. 76 lets a bank lend abroad only to a bank in
    credit; a refused approach is spent, so the network still reaches the same
    size but carries far fewer links across its borders:

    >>> free, _ = strategy_growth(rng=7)
    >>> tied, _ = strategy_growth(rng=7, restricted=True)
    >>> tied.network.n == free.network.n == 130
    True
    >>> 0 < tied.exterior_edges() < free.exterior_edges()
    True

    With ``on_refusal="retry"`` an arrival that is refused looks elsewhere:

    >>> retried, _ = strategy_growth(rng=7, restricted=True, on_refusal="retry")
    >>> retried.exterior_edges() > tied.exterior_edges()
    True
    """
    n0 = founding_banks
    if on_refusal not in ("retry", "drop"):
        raise ValueError(f"on_refusal must be 'retry' or 'drop', got {on_refusal!r}")
    if restriction_sides not in ("lender", "both"):
        raise ValueError(f"restriction_sides must be 'lender' or 'both', got {restriction_sides!r}")

    generator = as_rng(rng)
    choices = tuple(strategies) if strategies else tuple(int(s) for s in Strategy)

    total = n_communities * n0 + steps
    matrix = np.zeros((total, total))
    community = np.zeros(total, dtype=int)
    assigned = np.zeros(total, dtype=int)

    for c in range(n_communities):
        members = np.arange(c * n0, (c + 1) * n0)
        community[members] = c
        for idx, a in enumerate(members):
            for b in members[idx + 1 :]:
                matrix[a, b] = matrix[b, a] = 1.0

    seeded = n_communities * n0

    # One seed link per community, so that no community starts with zero
    # exterior degree and becomes permanently unreachable; see the module
    # docstring of netmotion.networks.communities.
    for c in range(n_communities):
        for _ in range(exterior_seed_links):
            other = int(generator.choice([h for h in range(n_communities) if h != c]))
            a = int(generator.choice(np.flatnonzero(community[:seeded] == c)))
            b = int(generator.choice(np.flatnonzero(community[:seeded] == other)))
            matrix[a, b] = matrix[b, a] = 1.0

    def exterior_degrees(count: int) -> np.ndarray:
        kc = np.zeros(n_communities)
        rows, cols = np.nonzero(matrix[:count, :count])
        for i, j in zip(rows, cols, strict=True):
            if i < j and community[i] != community[j]:
                kc[community[i]] += 1
                kc[community[j]] += 1
        return kc

    def clamp(value: float) -> int:
        """Counts are at least one link and at most four."""
        return int(np.clip(round(value), 1, 4))

    for step in range(steps):
        count = seeded + step
        new = count

        degrees = (matrix[:count, :count] != 0).sum(axis=1).astype(float)
        sizes = np.bincount(community[:count], minlength=n_communities).astype(float)
        kc = exterior_degrees(count)

        pc_size = sizes / sizes.sum()
        pc_exterior = kc / kc.sum() if kc.sum() > 0 else np.full(n_communities, 1.0 / n_communities)

        # Step 2: choose a community by rejection sampling on its size.
        target = 0
        for _ in range(_MAX_TRIES):
            candidate = int(generator.integers(0, n_communities))
            if generator.random() < pc_size[candidate]:
                target = candidate
                break
        community[new] = target

        strategy = Strategy(int(generator.choice(choices)))
        assigned[new] = int(strategy)

        members = np.flatnonzero(community[:count] == target)
        member_degrees = degrees[members]
        pi = (
            member_degrees / member_degrees.sum()
            if member_degrees.sum() > 0
            else np.full(members.size, 1.0 / max(members.size, 1))
        )
        pi_mean = float(pi.mean()) if pi.size else 0.0
        # Strategy 5's <Pi> = sum_{i=1}^{N} Pi_i / N, "el valor medio de todos
        # los valores ... que tiene cada nodo de la red": the kernel taken over
        # the whole network.
        pi_global = float((degrees / degrees.sum()).mean()) if degrees.sum() > 0 else 0.0
        k_community = float(member_degrees.mean()) if member_degrees.size else 0.0
        k_global = float(degrees.mean()) if degrees.size else 0.0
        kc_mean = float(kc.mean())
        pic_mean = float(pc_exterior.mean())

        # --- interior attachment -------------------------------------------
        if strategy is Strategy.SAFE:
            # Attach to the largest bank in the community, then two more
            # preferentially.
            largest = int(members[int(np.argmax(member_degrees))])
            matrix[new, largest] = matrix[largest, new] = 1.0
            interior_n, interior_p = 2, pi
        elif strategy is Strategy.IMITATE:
            interior_n, interior_p = clamp(k_community), np.full(members.size, pi_mean)
        elif strategy is Strategy.CAUTIOUS:
            interior_n, interior_p = clamp(k_community - 2), np.full(members.size, pi_mean)
        elif strategy is Strategy.RISKY:
            interior_n, interior_p = clamp(k_community + 2), np.full(members.size, 1.0 - pi_mean)
        elif strategy is Strategy.GLOBAL:
            interior_n, interior_p = clamp(k_global), np.full(members.size, pi_global)
        else:  # RANDOM and DOMESTIC both take two interior partners, uniformly
            interior_n, interior_p = 2, np.ones(members.size)

        # The kernel says *which* partners; the strategy says how many. One
        # sweep over the candidates fills the quota where the kernel is
        # generous, and is then topped up to the rule's floor where it is not.
        # Strategies 1, 6 and 7 state their counts as explicit integers ("union
        # ... a dos nodos interiores"), so for them the floor is the whole
        # count; the rest derive a count from a mean and state a floor beside
        # it ("... o como minimo igual a 1"), so for them the floor is one link.
        available = int((matrix[new, members] == 0).sum())
        interior_quota = min(interior_n, available)
        interior_floor = interior_quota if strategy in _EXPLICIT_COUNTS else min(1, interior_quota)
        chosen: set[int] = set()
        tries = 0
        while len(chosen) < interior_quota and (
            tries < members.size or (len(chosen) < interior_floor and tries < _MAX_TRIES)
        ):
            tries += 1
            slot = int(generator.integers(0, members.size))
            node = int(members[slot])
            if node in chosen or matrix[new, node] != 0:
                continue
            if generator.random() < interior_p[slot]:
                chosen.add(node)
                matrix[new, node] = matrix[node, new] = 1.0

        if strategy is Strategy.DOMESTIC:
            continue

        # --- exterior attachment -------------------------------------------
        if strategy is Strategy.SAFE:
            # "Union a la comunidad que tenga mas nodos": the largest community
            # the arrival can actually reach, which is the largest one other
            # than its own.
            abroad = [c for c in np.argsort(sizes)[::-1] if int(c) != target]
            exterior_n = 1
            exterior_p = np.zeros(n_communities)
            if abroad:
                exterior_p[int(abroad[0])] = pc_size[int(abroad[0])]
        elif strategy is Strategy.IMITATE:
            exterior_n, exterior_p = clamp(kc_mean), np.full(n_communities, pic_mean)
        elif strategy is Strategy.CAUTIOUS:
            exterior_n, exterior_p = clamp(kc_mean - 2), np.full(n_communities, pic_mean)
        elif strategy is Strategy.RISKY:
            exterior_n, exterior_p = clamp(kc_mean + 2), np.full(n_communities, 1.0 - pic_mean)
        elif strategy is Strategy.GLOBAL:
            exterior_n, exterior_p = clamp(kc_mean), np.full(n_communities, pic_mean)
        else:  # RANDOM
            exterior_n, exterior_p = 1, np.ones(n_communities)

        # PDF p. 76: lend abroad only to a bank whose net position is currently
        # positive. Under "retry" the condition narrows the pool the arrival
        # draws from, so it keeps drawing until it lands on a solvent bank;
        # under "drop" it draws from the whole of the abroad, as an unrestricted
        # arrival would, and the attempt is spent whether or not the bank it
        # approached can accept it.
        solvent = net_position_from_degrees(matrix, count) > 0 if restricted else None
        narrows_pool = solvent if (restricted and on_refusal == "retry") else None
        refuses = ~solvent if (restricted and on_refusal == "drop") else None
        if restricted and restriction_sides == "both":
            if net_position_from_degrees(matrix, count + 1)[new] <= 0:
                continue  # an arrival in debt may not lend abroad

        pools = _exterior_pools(community, count, n_communities, target, exterior_p, narrows_pool)
        capacity = sum(int(pool.size) for pool in pools.values())
        exterior_quota = min(exterior_n, capacity)
        exterior_floor = exterior_quota if strategy in _EXPLICIT_COUNTS else min(1, exterior_quota)

        attached: set[int] = set()
        tries = 0
        while len(attached) < exterior_quota and (
            tries < n_communities or (len(attached) < exterior_floor and tries < _MAX_TRIES)
        ):
            tries += 1
            other = int(generator.integers(0, n_communities))
            if other not in pools or generator.random() >= exterior_p[other]:
                continue
            partner = int(generator.choice(pools[other]))
            if partner in attached:
                continue
            attached.add(partner)
            if refuses is not None and refuses[partner]:
                # The bank approached is in deficit and will not take the loan.
                # The attempt counts against the arrival's quota, so the arrival
                # goes without this foreign link rather than looking elsewhere.
                continue
            matrix[new, partner] = matrix[partner, new] = 1.0

    label = name or (
        f"7-strategy {'restricted ' if restricted else ''}"
        f"{'drop ' if restricted and on_refusal == 'drop' else ''}"
        f"Cn={n_communities} t={steps}"
    )
    return CommunityNetwork(Network(matrix, name=label), community), assigned


@sourced(
    Origin.PAPER_2012,
    pdf_pages="4-5",
    section="2.1 The network model, Steps 0-4",
    document="paper",
    note=(
        "The article's three strategies, which move both the link counts and "
        "the attachment probabilities by the confidence factor alpha: strategy "
        "1 at d and Pi, strategy 2 at d/alpha and Pi/alpha, strategy 3 at "
        "alpha*d and alpha*Pi. Step 4, the explicit disequilibrium that grows "
        "the Too-Big-To-Fail banks, runs inside the growth loop."
    ),
    reading=(
        "(1) Strategy 3 accepts with alpha*Pi_mi and alpha*P_m', as printed on "
        "article p. 5. (2) 'c_min <= d nodes' is "
        "read as a count bounded below by c_min and above by d: the candidates "
        "are swept once at the attachment probability, and the sweep is topped "
        "up to c_min. (3) The defaults are the article's own values, from the "
        "caption of its Fig. 1: C = 10, e = 1, n0 = 4, t = 200, "
        "d_in = 4, d_ex = 2, c_min = 1, C_min = 0, alpha = 2, and the Set 1 "
        "strategy mix [75, 20, 5] read from Fig. 2a. (4) Step 4 chooses one strategy-3 node 'randomly' and "
        "connects it 'with probability Pi_m'n''; the choice is read as uniform "
        "and Pi as the acceptance probability of that one connection."
    ),
    tags=("interbank", "confidence-factor"),
)
def confidence_strategy_growth(
    n_communities: int = 10,
    *,
    founding_banks: int = 4,
    exterior_seed_links: int = 1,
    steps: int = 200,
    interior_max: int = 4,
    exterior_max: int = 2,
    interior_min: int = 1,
    exterior_min: int = 0,
    confidence: float = 2.0,
    mix: tuple[float, float, float] = (75.0, 20.0, 5.0),
    restricted: bool = False,
    on_refusal: str = "drop",
    too_big_to_fail: int = 0,
    rng: SeedLike = None,
    name: str = "",
) -> tuple[CommunityNetwork, np.ndarray]:
    """Grow the article's network: three strategies and a confidence factor.

    The three strategies differ by how the *confidence factor* :math:`\\alpha`
    moves their appetite (article pp. 4-5).  Strategy 1 is plain preferential
    attachment: :math:`d_{in}` interior links at :math:`\\Pi_{mi}`, and
    :math:`d_{ex}` exterior ones at :math:`P_{m'}`.  Strategy 2 is the bank
    that finds partners hard to come by, and divides both by :math:`\\alpha`.
    Strategy 3 is the bank that is *prone* to connect, and multiplies both by
    :math:`\\alpha`: it is the rule that grows the too-big-to-fail banks, and
    the article's Fig. 2 shows it providing the most edges per node.

    Parameters
    ----------
    interior_max, exterior_max:
        :math:`d_{in}` and :math:`d_{ex}`, the number of interior and exterior
        connections defined for strategy 1.
    interior_min, exterior_min:
        :math:`c_{min}` and :math:`C_{min}`, the minima below which no strategy
        goes.
    confidence:
        :math:`\\alpha`.  Must be at least 1; at exactly 1 the three strategies
        coincide.
    mix:
        Percentages of the three strategies, in order.
    restricted:
        The restriction of article p. 15, used for its Fig. 8: a node may only
        connect to a node *i* of another community if :math:`D_i > 0`,
        evaluated from the degrees as they stand at that moment.
    on_refusal:
        What an arrival does when that restriction turns it down -- ``"drop"``
        (the default) spends the attempt, ``"retry"`` looks for another
        partner.  See the module docstring.
    too_big_to_fail:
        An optional post-hoc rewiring: after growth
        this many of the largest banks are grown further.  It is *not* the
        article's Step 4, which always runs and lives inside the growth loop.

    Examples
    --------
    >>> cn, assigned = confidence_strategy_growth(rng=3)
    >>> cn.network.n
    240
    >>> sorted(set(assigned[40:].tolist()))
    [1, 2, 3]

    Strategy 3 is the one that grows the big banks:

    >>> degrees = cn.network.degrees()
    >>> float(degrees[assigned == 3].mean()) > float(degrees[assigned == 2].mean())
    True
    """
    n0 = founding_banks
    if on_refusal not in ("retry", "drop"):
        raise ValueError(f"on_refusal must be 'retry' or 'drop', got {on_refusal!r}")
    if confidence < 1.0:
        raise ValueError(f"confidence factor alpha must be at least 1, got {confidence}")
    if len(mix) != 3 or sum(mix) <= 0:
        raise ValueError("mix must be three positive percentages")

    generator = as_rng(rng)
    weights = np.asarray(mix, dtype=float) / sum(mix)

    total = n_communities * n0 + steps
    matrix = np.zeros((total, total))
    community = np.zeros(total, dtype=int)
    assigned = np.zeros(total, dtype=int)

    for c in range(n_communities):
        members = np.arange(c * n0, (c + 1) * n0)
        community[members] = c
        for idx, a in enumerate(members):
            for b in members[idx + 1 :]:
                matrix[a, b] = matrix[b, a] = 1.0

    seeded = n_communities * n0
    for c in range(n_communities):
        for _ in range(exterior_seed_links):
            other = int(generator.choice([h for h in range(n_communities) if h != c]))
            a = int(generator.choice(np.flatnonzero(community[:seeded] == c)))
            b = int(generator.choice(np.flatnonzero(community[:seeded] == other)))
            matrix[a, b] = matrix[b, a] = 1.0

    for step in range(steps):
        count = seeded + step
        new = count

        degrees = (matrix[:count, :count] != 0).sum(axis=1).astype(float)
        sizes = np.bincount(community[:count], minlength=n_communities).astype(float)
        pc_size = sizes / sizes.sum()

        target = 0
        for _ in range(_MAX_TRIES):
            candidate = int(generator.integers(0, n_communities))
            if generator.random() < pc_size[candidate]:
                target = candidate
                break
        community[new] = target

        # Step 3: the arrival adopts one of the three strategies.
        strategy = int(generator.choice([1, 2, 3], p=weights))
        assigned[new] = strategy

        # Strategy 1 lends at its defined appetite, strategy 2 at 1/alpha of
        # it, strategy 3 at alpha times it -- the counts and the attachment
        # probabilities together. Strategy 3 is the "prone to perform
        # connections" rule, so it is the most connective of the three, and the
        # high-degree region 20 <= k <= 55 of the article's Fig. 3 comes from it.
        appetite = 1.0 if strategy == 1 else (1.0 / confidence if strategy == 2 else confidence)

        members = np.flatnonzero(community[:count] == target)
        member_degrees = degrees[members]
        pi = (
            member_degrees / member_degrees.sum()
            if member_degrees.sum() > 0
            else np.full(members.size, 1.0 / max(members.size, 1))
        )
        interior_p = np.minimum(pi * appetite, 1.0)
        exterior_p = np.minimum(pc_size * appetite, 1.0)

        interior_n = max(interior_min, int(round(interior_max * appetite)))
        exterior_n = max(exterior_min, int(round(exterior_max * appetite)))

        # "Connection to c_min <= d_in nodes i ... with the probability Pi_mi":
        # a count bounded below by c_min and above by d_in, with the kernel
        # deciding where between them it falls. The community is swept once at
        # the attachment probability, capped at the strategy's count and topped
        # up to c_min. Pi_mi sums to one over the community, so a sweep yields
        # about one link for strategy 1, 1/alpha for strategy 2 and alpha for
        # strategy 3 -- which is what makes alpha change how many links are
        # made and not merely how fast they are found.
        interior_quota = min(interior_n, members.size)
        interior_floor = min(interior_min, interior_quota)
        chosen: set[int] = set()
        tries = 0
        while len(chosen) < interior_quota and (
            tries < members.size or (len(chosen) < interior_floor and tries < _MAX_TRIES)
        ):
            tries += 1
            slot = int(generator.integers(0, members.size))
            node = int(members[slot])
            if node in chosen:
                continue
            if generator.random() < interior_p[slot]:
                chosen.add(node)
                matrix[new, node] = matrix[node, new] = 1.0

        # Article p. 15: a node may connect to a node of another community only
        # if that node's net position is currently positive. "retry" narrows the
        # pool and keeps drawing; "drop" spends the attempt on whoever was
        # approached. See the module docstring.
        solvent = net_position_from_degrees(matrix, count) > 0 if restricted else None
        narrows_pool = solvent if (restricted and on_refusal == "retry") else None
        refuses = ~solvent if (restricted and on_refusal == "drop") else None

        pools = _exterior_pools(community, count, n_communities, target, exterior_p, narrows_pool)
        capacity = sum(int(pool.size) for pool in pools.values())
        exterior_quota = min(exterior_n, capacity)
        exterior_floor = min(exterior_min, exterior_quota)
        attached: set[int] = set()
        tries = 0
        while len(attached) < exterior_quota and (
            tries < n_communities or (len(attached) < exterior_floor and tries < _MAX_TRIES)
        ):
            tries += 1
            other = int(generator.integers(0, n_communities))
            if other not in pools or generator.random() >= exterior_p[other]:
                continue
            partner = int(generator.choice(pools[other]))
            if partner in attached:
                continue
            attached.add(partner)
            if refuses is not None and refuses[partner]:
                continue
            matrix[new, partner] = matrix[partner, new] = 1.0

        # Step 4: "In order to produce a disequilibrium in the growth process,
        # it is considered that the nodes with strategy 3 are more active than
        # the other nodes. Therefore, at each step, one of the nodes n'
        # generated in earlier iterations and defined by the strategy 3 is
        # randomly chosen and connected to the node added at this iteration
        # with probability Pi_{m'n'}."  One veteran is chosen uniformly, and
        # the connection is made with probability Pi, Eq. (2) read in the
        # veteran's own community from the degrees of Step 1.
        active = np.flatnonzero(assigned[:count] == 3)
        if active.size:
            veteran = int(generator.choice(active))
            members = community[:count] == community[veteran]
            total = degrees[members].sum()
            if total > 0 and generator.random() < degrees[veteran] / total:
                matrix[new, veteran] = matrix[veteran, new] = 1.0

    # An optional post-hoc rewiring: grow a
    # handful of the largest banks further once growth has finished. The
    # article's own Step 4 is inside the loop above.
    if too_big_to_fail:
        degrees = (matrix != 0).sum(axis=1)
        for bank in np.argsort(degrees)[::-1][:too_big_to_fail]:
            pool = np.flatnonzero(matrix[bank] == 0)
            pool = pool[pool != bank]
            extra = min(pool.size, int(round(2.5 * degrees.mean())))
            if extra <= 0:
                continue
            for partner in generator.choice(pool, size=extra, replace=False):
                matrix[bank, partner] = matrix[partner, bank] = 1.0

    label = name or (
        f"paper 3-strategy {'restricted ' if restricted else ''}"
        f"alpha={confidence:g} Cn={n_communities} t={steps}"
    )
    return CommunityNetwork(Network(matrix, name=label), community), assigned
