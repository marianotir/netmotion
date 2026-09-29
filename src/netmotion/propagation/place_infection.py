"""Infection through *places*: the city model of thesis section 5.4.3.

Thesis section 5.4.3, *Infecciones: Otra perspectiva* (PDF pp. 51-53, printed
pp. 48-50).

Section 5.4.1 propagates an infection over the network itself: a node is a
person, a link is a relationship, and the disease crosses the link.  Section
5.4.3 throws that away and starts again from a different premise, stated in the
thesis in one sentence: *"la persona infectada y la contagiada no han de
conocerse o tener relacion alguna mas alla del hecho de haber por ejemplo
hablado por telefono en la misma cabina o haberse hecho fotos en el mismo
fotomaton"*.  Contagion happens at a **place**, not along an acquaintance.

So the model has two populations, not one:

* **places** -- schools, cafes, offices, restaurants, shopping centres, public
  transport, and the *casas*, one per habitant;
* **habitants** -- each of whom owns a home and walks a fixed random closed
  route out of that home, through three to five places, and back.

Nodes are the places, and a link is drawn between two places when some habitant
walks from one to the other (figs. 48-50).  The dynamics, however, do **not**
run on that link structure.  They run on the *incidence* between habitants and
places, which is a bipartite relation:

.. math::

    \\Psi_{t+1} = \\Psi_t \\cup \\{h : \\mathcal{V}(h) \\cap \\Phi_t \\neq
    \\emptyset\\}, \\qquad
    \\Phi_{t+1} = \\Phi_t \\cup \\bigcup_{h \\in \\Psi_{t+1}} \\mathcal{V}(h)

where :math:`\\Phi_t` is the set of contaminated places, :math:`\\Psi_t` the set
of infected habitants and :math:`\\mathcal{V}(h)` the set of places on the route
of *h*.  The thesis states it in words (PDF p. 52): an infected person *"dejara
restos de su infeccion en todos los lugares por los que pase"*, and then *"todos
los habitantes que pasan por esos lugares quedaran infectados"*, whose own
routes are contaminated in turn.

An infected habitant contaminates every place on his route *at once*, including
places two or three steps apart along the drawn place-graph, so the process is
not :func:`~netmotion.propagation.propagate` run on :meth:`City.place_graph`,
which gives the network drawn in figs. 48-50.

The reported run
----------------
The city of figs. 48-53 has 100 habitants and ten types of place with counts
``[16, 19, 15, 11, 20, 13, 15, 14, 20, 100]`` -- 243 places, of which the last
100 are the houses, one per habitant -- and per-type contagion probabilities
``[92, 60, 99, 61, 88, 84, 56, 72, 98, 97]`` (percent).  One habitant is
infected at random and the infection covers the whole population in five
instants: fig. 52 plots healthy and infected counts against :math:`t = 0 \\dots
5`, crossing between :math:`t = 2` and :math:`t = 3`, and fig. 53 plots the
infected fraction rising from 0.04 at :math:`t = 1` to 1 at :math:`t = 5`.

Both figures saturate at 100, the habitant count.  :class:`PlaceInfectionRun`
reports the habitants and the places separately.

Readings taken
--------------
The default follows the described mechanism, in which everybody who passes
through a contaminated place is infected; ``use_probabilities=True`` applies
the per-place contagion probabilities.  Other readings are recorded in the
``reading`` field of the objects below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng

__all__ = [
    "THESIS_PLACE_COUNTS",
    "THESIS_CONTAGION_PCT",
    "City",
    "build_city",
    "PlaceInfectionRun",
    "propagate_places",
    "place_infection_curves",
]

#: Places per type in the city of figs. 48-53 (PDF p. 51).  The last type is
#: the *casas*: one house per habitant, hence exactly 100 of them.
THESIS_PLACE_COUNTS: tuple[int, ...] = (16, 19, 15, 11, 20, 13, 15, 14, 20, 100)

#: Per-type contagion probability, in percent, as printed on PDF p. 51.
THESIS_CONTAGION_PCT: tuple[int, ...] = (92, 60, 99, 61, 88, 84, 56, 72, 98, 97)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="51-52",
    section="5.4.3 Infecciones: Otra perspectiva",
    note=(
        "The city: places typed 0..K-1, one house per habitant, and one fixed "
        "closed route per habitant. The habitant-place incidence is the object "
        "the dynamics run on; the place-graph is the drawing of figs. 48-50."
    ),
    reading=(
        "The links of figs. 48-50 join consecutive places on each route, "
        "following 'las ramas representaran los caminos que siguen los "
        "individuos entre esos lugares'. The dynamics run on the incidence."
    ),
    tags=("chapter-5", "bipartite"),
)
@dataclass(eq=False)
class City:
    """Places, habitants, and the closed route each habitant walks.

    Parameters
    ----------
    place_type:
        ``(P,)`` integer array; ``place_type[u]`` is the type index of place
        *u*, in ``0 .. K-1``.
    home_of:
        ``(H,)`` integer array; ``home_of[h]`` is the place that is habitant
        *h*'s house.  Houses are not shared, so this is injective.
    paths:
        ``paths[h]`` is the closed walk ``[home, u_1, ..., u_m, home]``.
    contagion_prob:
        ``(K,)`` array of per-type contagion probabilities in ``[0, 1]``, i.e.
        the printed percentages divided by 100.
    name:
        Free-form label.

    Notes
    -----
    The route is stored as a *walk*, keeping the order in which the places are
    visited, because the drawn network of figs. 48-50 needs the consecutive
    pairs.  The dynamics only ever need :meth:`visited`, the unordered set.
    """

    place_type: np.ndarray
    home_of: np.ndarray
    paths: list[list[int]]
    contagion_prob: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.place_type = np.asarray(self.place_type, dtype=int)
        self.home_of = np.asarray(self.home_of, dtype=int)
        self.contagion_prob = np.asarray(self.contagion_prob, dtype=float)
        if self.place_type.ndim != 1 or self.home_of.ndim != 1:
            raise ValueError("place_type and home_of must be one-dimensional")
        if len(self.paths) != self.home_of.size:
            raise ValueError(
                f"{len(self.paths)} routes for {self.home_of.size} habitants; "
                "every habitant walks exactly one route"
            )
        n_places = self.place_type.size
        if self.place_type.size and self.place_type.max() >= self.contagion_prob.size:
            raise ValueError(
                f"place types run up to {int(self.place_type.max())} but only "
                f"{self.contagion_prob.size} contagion probabilities were given"
            )
        if np.any((self.contagion_prob < 0.0) | (self.contagion_prob > 1.0)):
            raise ValueError("contagion probabilities must lie in [0, 1] (percent / 100)")
        if self.home_of.size and np.unique(self.home_of).size != self.home_of.size:
            raise ValueError("two habitants share a house; home_of must be injective")
        for h, path in enumerate(self.paths):
            if len(path) < 2 or path[0] != path[-1] or path[0] != int(self.home_of[h]):
                raise ValueError(
                    f"route of habitant {h} must start and end at his house "
                    f"{int(self.home_of[h])}, got {path}"
                )
            if any(not 0 <= u < n_places for u in path):
                raise ValueError(f"route of habitant {h} leaves the city: {path}")

    # ------------------------------------------------------------------
    # basic structure
    # ------------------------------------------------------------------

    @property
    def n_places(self) -> int:
        """Number of places, :math:`P` (243 in the thesis's city)."""
        return int(self.place_type.size)

    @property
    def n_habitants(self) -> int:
        """Number of habitants, :math:`H` (100 in the thesis's city)."""
        return int(self.home_of.size)

    @property
    def n_types(self) -> int:
        """Number of place types, :math:`K` (10 in the thesis's city)."""
        return int(self.contagion_prob.size)

    def visited(self, h: int) -> np.ndarray:
        """:math:`\\mathcal{V}(h)`: the distinct places on habitant *h*'s route.

        Sorted and de-duplicated, so the home appears once even though the walk
        begins and ends there.
        """
        return np.unique(np.asarray(self.paths[h], dtype=int))

    def incidence(self) -> np.ndarray:
        """The ``(H, P)`` boolean habitant-place incidence the model runs on.

        Row *h* is the indicator of :math:`\\mathcal{V}(h)`.  This matrix, not
        an adjacency matrix, is the structure of section 5.4.3.
        """
        m = np.zeros((self.n_habitants, self.n_places), dtype=bool)
        for h in range(self.n_habitants):
            m[h, self.visited(h)] = True
        return m

    def place_edges(self) -> list[tuple[int, int]]:
        """The consecutive pairs of every route, de-duplicated, ``i < j``."""
        edges: set[tuple[int, int]] = set()
        for path in self.paths:
            for a, b in zip(path[:-1], path[1:], strict=True):
                if a != b:
                    edges.add((a, b) if a < b else (b, a))
        return sorted(edges)

    @sourced(
        Origin.THESIS_2011,
        pdf_pages="52",
        section="5.4.3 Infecciones: Otra perspectiva",
        note=(
            "Figs. 48-50: the network drawn from the city, nodes = places, "
            "links = the steps habitants take between them. The propagation of "
            "5.4.3 runs on the incidence."
        ),
    )
    def place_graph(self) -> Network:
        """The place-graph of figs. 48-50.

        Nodes are places and a link joins two places that some habitant walks
        between; each step is one link.  The propagation of section 5.4.3 runs
        on :meth:`incidence`: an infected habitant contaminates his whole route
        in one instant, however far apart its places are on this graph.
        """
        return Network.from_edges(
            self.n_places, self.place_edges(), name=self.name or "city places"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        label = f" {self.name!r}" if self.name else ""
        return (
            f"<City{label} places={self.n_places} habitants={self.n_habitants} "
            f"types={self.n_types}>"
        )


@sourced(
    Origin.THESIS_2011,
    pdf_pages="51",
    section="5.4.3 Infecciones: Otra perspectiva",
    note=(
        "The city of figs. 48-53: counts [16,19,15,11,20,13,15,14,20,100], "
        "100 habitants, one house each from the last (100-strong) type, one "
        "fixed random closed route per habitant through 3 to 5 further places, "
        "and the per-type contagion probabilities [92,60,99,61,88,84,56,72,"
        "98,97] percent. Those are the defaults, so build_city(rng=...) is the "
        "thesis's own city."
    ),
    reading=(
        "'un camino aleatorio que pase por como minimo 3 lugares y como maximo "
        "por 5' is read as 3 to 5 places besides the house, drawn uniformly "
        "without replacement from every place except the walker's own house. "
        "home_counts_toward_length=True and allow_homes_on_paths=False give "
        "other readings. The houses are the last type, whose count equals the "
        "number of habitants."
    ),
    tags=("chapter-5", "bipartite"),
)
def build_city(
    place_counts: Sequence[int] = THESIS_PLACE_COUNTS,
    n_habitants: int = 100,
    *,
    min_places: int = 3,
    max_places: int = 5,
    contagion_percentages: Sequence[float] | None = None,
    allow_homes_on_paths: bool = True,
    home_counts_toward_length: bool = False,
    rng: SeedLike = None,
    name: str = "",
) -> City:
    """Build a city and draw one closed route for each habitant.

    Parameters
    ----------
    place_counts:
        Places per type.  The last type holds the houses and must have at least
        ``n_habitants`` places in it.
    n_habitants:
        Number of habitants, one per house.
    min_places, max_places:
        Bounds on the number of places a route passes through, inclusive.
    contagion_percentages:
        Per-type contagion probabilities *in percent*.  Defaults to the
        thesis's list.  Only used when
        :func:`propagate_places` is called with ``use_probabilities=True``.
    allow_homes_on_paths:
        Whether other habitants' houses may appear on a route.  ``True``, the
        default, draws the visited places uniformly from the whole city; a
        walker's own house is excluded either way, since the route already
        starts and ends there.
    home_counts_toward_length:
        ``False`` (the default) reads "3 to 5 places" as places *besides* the
        house; ``True`` counts the house as one of them.

    Examples
    --------
    With no arguments but a seed, this is the city of figs. 48-53 (PDF p. 51):

    >>> city = build_city(rng=2011)
    >>> city.n_places, city.n_habitants, city.n_types
    (243, 100, 10)

    Every habitant owns exactly one of the 100 houses, which are the places of
    the last type:

    >>> houses = sorted(city.home_of.tolist())
    >>> houses == list(range(143, 243))
    True
    >>> sorted(set(city.place_type[houses].tolist()))
    [9]

    Every route is a closed walk out of its owner's house through three to five
    further places:

    >>> route = city.paths[0]
    >>> route[0] == route[-1] == int(city.home_of[0])
    True
    >>> sorted({len(city.visited(h)) - 1 for h in range(city.n_habitants)})
    [3, 4, 5]

    The contagion probabilities are the printed percentages, as fractions:

    >>> city.contagion_prob.tolist()
    [0.92, 0.6, 0.99, 0.61, 0.88, 0.84, 0.56, 0.72, 0.98, 0.97]
    """
    counts, contagion_pct = place_counts, contagion_percentages
    counts_arr = np.asarray(counts, dtype=int)
    if counts_arr.ndim != 1 or counts_arr.size == 0:
        raise ValueError("counts must be a non-empty sequence of places per type")
    if np.any(counts_arr < 0):
        raise ValueError(f"place counts must be non-negative, got {counts_arr.tolist()}")
    if n_habitants < 0:
        raise ValueError(f"n_habitants must be non-negative, got {n_habitants}")
    if not 1 <= min_places <= max_places:
        raise ValueError(f"need 1 <= min_places <= max_places, got {min_places}, {max_places}")
    if counts_arr[-1] < n_habitants:
        raise ValueError(
            f"the last place type holds the houses and has {int(counts_arr[-1])} "
            f"places, too few for {n_habitants} habitants"
        )

    pct = THESIS_CONTAGION_PCT if contagion_pct is None else contagion_pct
    pct_arr = np.asarray(pct, dtype=float)
    if pct_arr.size != counts_arr.size:
        raise ValueError(
            f"{pct_arr.size} contagion probabilities for {counts_arr.size} place types"
        )

    generator = as_rng(rng)
    place_type = np.repeat(np.arange(counts_arr.size), counts_arr)
    n_places = int(counts_arr.sum())

    # The houses are the last block of the last type, so home_of is a bijection
    # onto them whenever counts[-1] == n_habitants, as in the thesis.
    first_house = n_places - int(counts_arr[-1])
    home_of = np.arange(first_house, first_house + n_habitants)

    n_visited_lo = min_places - 1 if home_counts_toward_length else min_places
    n_visited_hi = max_places - 1 if home_counts_toward_length else max_places
    if n_visited_lo < 1:
        raise ValueError(
            "with home_counts_toward_length=True a route needs min_places >= 2 "
            "so that at least one place other than the house is visited"
        )

    candidates = np.arange(n_places)
    if not allow_homes_on_paths:
        candidates = np.setdiff1d(candidates, home_of)

    paths: list[list[int]] = []
    for h in range(n_habitants):
        home = int(home_of[h])
        pool = candidates[candidates != home]
        n_visited = int(generator.integers(n_visited_lo, n_visited_hi + 1))
        if n_visited > pool.size:
            raise ValueError(
                f"a route needs {n_visited} places but only {pool.size} are "
                "available; the city is too small"
            )
        visited = generator.choice(pool, size=n_visited, replace=False)
        paths.append([home, *(int(u) for u in visited), home])

    return City(
        place_type=place_type,
        home_of=home_of,
        paths=paths,
        contagion_prob=pct_arr / 100.0,
        name=name,
    )


@sourced(
    Origin.THESIS_2011,
    pdf_pages="52-53",
    section="5.4.3 Infecciones: Otra perspectiva",
    note=(
        "The outcome of one run of the place model: the instant each habitant "
        "was infected and each place contaminated, and the two counting curves "
        "of figs. 52 and 53."
    ),
    reading=(
        "Instants are indexed as in fig. 52, from t = 0: instant 0 is the "
        "seeded habitant with his route contaminated, and one instant is one "
        "place-to-habitant-to-place round."
    ),
    tags=("chapter-5", "bipartite"),
)
@dataclass
class PlaceInfectionRun:
    """One propagation through the city, from one seeded habitant.

    Attributes
    ----------
    first_infected:
        The habitant infected at :math:`t = 0`.
    infected_at:
        ``(H,)`` array; the instant each habitant was infected, ``np.inf`` if
        never.  The seed has ``0.0``.
    contaminated_at:
        ``(P,)`` array; likewise for the places.  Every place on the seed's own
        route is contaminated at ``0.0``.
    people_counts, place_counts:
        Infected habitants and contaminated places at each instant, the two
        quantities of fig. 52.  ``people_counts[k]`` is the count at
        :math:`t = k`.
    """

    first_infected: int
    infected_at: np.ndarray
    contaminated_at: np.ndarray
    people_counts: np.ndarray
    place_counts: np.ndarray

    @property
    def n_habitants(self) -> int:
        return int(self.infected_at.size)

    @property
    def n_places(self) -> int:
        return int(self.contaminated_at.size)

    @property
    def people_curve(self) -> np.ndarray:
        """Infected fraction of the population -- the curve of fig. 53."""
        return self.people_counts / self.n_habitants

    @property
    def places_curve(self) -> np.ndarray:
        """Contaminated fraction of the places."""
        return self.place_counts / self.n_places

    @property
    def healthy_counts(self) -> np.ndarray:
        """The *sano* curve of fig. 52: ``H`` minus the infected count."""
        return self.n_habitants - self.people_counts

    @property
    def steps(self) -> int:
        """Instants after the seed, i.e. the last :math:`t` of the run."""
        return int(self.people_counts.size - 1)

    @property
    def complete(self) -> bool:
        """Whether every habitant was eventually infected."""
        return bool(np.isfinite(self.infected_at).all())

    def infected_people(self, t: int) -> np.ndarray:
        """Indices of the habitants infected at instant *t*."""
        return np.flatnonzero(self.infected_at <= t)

    def contaminated_places(self, t: int) -> np.ndarray:
        """Indices of the places contaminated at instant *t*."""
        return np.flatnonzero(self.contaminated_at <= t)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="52-53",
    section="5.4.3 Infecciones: Otra perspectiva",
    note=(
        "Psi_{t+1} = Psi_t + {h : V(h) meets Phi_t}, Phi_{t+1} = Phi_t + union "
        "of V(h) over the newly infected. One habitant is seeded at random and "
        "his whole route is contaminated at t = 0. Neither set ever shrinks: "
        "the model has no recovery; PDF p. 53 names 'el tiempo que un lugar "
        "puede estar infectado' as a factor for variable routes."
    ),
    reading=(
        "The default is the described mechanism, 'todos los habitantes que "
        "pasan por esos lugares quedaran infectados'. use_probabilities=True "
        "applies the per-place percentages, one independent draw per "
        "contaminated place a susceptible habitant walks through, per instant. "
        "A habitant's own house is one of the places on his route."
    ),
    tags=("chapter-5", "bipartite", "original-behaviour"),
)
def propagate_places(
    city: City,
    first_infected: int | None = None,
    *,
    use_probabilities: bool = False,
    max_steps: int | None = None,
    rng: SeedLike = None,
) -> PlaceInfectionRun:
    """Infect one habitant and let the city carry the infection.

    Parameters
    ----------
    first_infected:
        The habitant infected at :math:`t = 0`.  ``None`` draws one uniformly
        at random, which is the thesis's *"una persona al azar"*.
    use_probabilities:
        Apply the per-type contagion probabilities instead of infecting
        everyone who steps on a contaminated place.  See the reading note.
    max_steps:
        Stop after this many instants even if the infection is still
        spreading.  ``None`` runs to the absorbing state, which the
        deterministic rule reaches in at most :math:`H` instants.

    Examples
    --------
    The three-habitant city: houses 3, 4, 5 and shared places 0, 1, 2, with
    routes ``3-0-1-3``, ``4-1-2-4`` and ``5-2-0-5``.

    >>> import numpy as np
    >>> city = City(
    ...     place_type=np.array([0, 0, 0, 1, 1, 1]),
    ...     home_of=np.array([3, 4, 5]),
    ...     paths=[[3, 0, 1, 3], [4, 1, 2, 4], [5, 2, 0, 5]],
    ...     contagion_prob=np.array([1.0, 1.0]),
    ... )
    >>> run = propagate_places(city, 0)

    At :math:`t = 0` the seed alone is infected and his three places are
    contaminated; at :math:`t = 1` both other habitants have walked through one
    of them, so the whole city is infected:

    >>> run.infected_people(0).tolist(), run.contaminated_places(0).tolist()
    ([0], [0, 1, 3])
    >>> run.people_counts.tolist(), run.place_counts.tolist()
    ([1, 3], [3, 6])
    >>> run.steps, run.complete
    (1, True)

    Contagion probabilities of zero stop the infection at the seed, whatever
    the routes:

    >>> harmless = City(city.place_type, city.home_of, city.paths, np.zeros(2))
    >>> stalled = propagate_places(harmless, 0, use_probabilities=True, rng=0)
    >>> stalled.people_counts.tolist(), stalled.complete
    ([1], False)
    """
    n_habitants, n_places = city.n_habitants, city.n_places
    if n_habitants == 0:
        raise ValueError("an empty city has nobody to infect")

    generator = as_rng(rng)
    if first_infected is None:
        first_infected = int(generator.integers(n_habitants))
    if not 0 <= first_infected < n_habitants:
        raise IndexError(f"first infected habitant {first_infected} outside 0..{n_habitants - 1}")

    incidence = city.incidence()
    place_prob = city.contagion_prob[city.place_type]

    infected = np.zeros(n_habitants, dtype=bool)
    contaminated = np.zeros(n_places, dtype=bool)
    infected_at = np.full(n_habitants, np.inf)
    contaminated_at = np.full(n_places, np.inf)

    infected[first_infected] = True
    infected_at[first_infected] = 0.0
    contaminated[city.visited(first_infected)] = True
    contaminated_at[contaminated] = 0.0

    people_counts = [int(infected.sum())]
    place_counts = [int(contaminated.sum())]

    t = 0
    while max_steps is None or t < max_steps:
        # A habitant is exposed when his route meets the contaminated set. In
        # the probabilistic reading a place that can never infect anybody is no
        # exposure at all, and excluding it is what makes the absorbing state
        # reachable rather than merely likely.
        reachable = contaminated & (place_prob > 0.0) if use_probabilities else contaminated
        exposed = (incidence & reachable).any(axis=1) & ~infected
        if not exposed.any():
            break

        if use_probabilities:
            newly = np.zeros(n_habitants, dtype=bool)
            for h in np.flatnonzero(exposed):
                met = incidence[h] & contaminated
                draws = generator.random(int(met.sum()))
                newly[h] = bool((draws < place_prob[met]).any())
        else:
            newly = exposed

        t += 1
        if newly.any():
            infected |= newly
            infected_at[newly] = float(t)
            spread = incidence[newly].any(axis=0) & ~contaminated
            contaminated |= spread
            contaminated_at[spread] = float(t)
        people_counts.append(int(infected.sum()))
        place_counts.append(int(contaminated.sum()))

    return PlaceInfectionRun(
        first_infected=int(first_infected),
        infected_at=infected_at,
        contaminated_at=contaminated_at,
        people_counts=np.asarray(people_counts, dtype=int),
        place_counts=np.asarray(place_counts, dtype=int),
    )


@sourced(
    Origin.THESIS_2011,
    pdf_pages="53",
    section="5.4.3 Infecciones: Otra perspectiva",
    note=(
        "The two curves the section reports: fig. 52 counts healthy and "
        "infected habitants against t, fig. 53 plots the infected fraction. "
        "The place curve is returned as well."
    ),
)
def place_infection_curves(
    city: City, first_infected: int | None = None, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """``(people, places)``: the two fractions against :math:`t = 0, 1, \\dots`.

    Examples
    --------
    >>> import numpy as np
    >>> city = City(
    ...     place_type=np.array([0, 0, 0, 1, 1, 1]),
    ...     home_of=np.array([3, 4, 5]),
    ...     paths=[[3, 0, 1, 3], [4, 1, 2, 4], [5, 2, 0, 5]],
    ...     contagion_prob=np.array([1.0, 1.0]),
    ... )
    >>> people, places = place_infection_curves(city, 0)
    >>> [round(float(x), 4) for x in people]
    [0.3333, 1.0]
    >>> places.tolist()
    [0.5, 1.0]
    """
    run = propagate_places(city, first_infected, **kwargs)
    return run.people_curve, run.places_curve
