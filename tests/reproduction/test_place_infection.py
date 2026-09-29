"""Section 5.4.3, *Infecciones: Otra perspectiva*, as assertions.

The city is fully specified -- counts, houses, route lengths, contagion
percentages -- and those are equalities.  The propagation is stochastic, so its
numbers are asserted as relations.

Every test names the PDF page it is checking; printed page = PDF page - 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from netmotion.propagation.infection import propagate
from netmotion.propagation.place_infection import (
    THESIS_CONTAGION_PCT,
    THESIS_PLACE_COUNTS,
    City,
    PlaceInfectionRun,
    build_city,
    place_infection_curves,
    propagate_places,
)

pytestmark = pytest.mark.reproduction


@pytest.fixture
def thesis_city() -> City:
    """The city of figs. 48-53, drawn from a recorded seed."""
    return build_city(rng=20110930)


def _three_habitant_city() -> City:
    """Houses 3, 4, 5; shared places 0, 1, 2; routes 3-0-1-3, 4-1-2-4, 5-2-0-5."""
    return City(
        place_type=np.array([0, 0, 0, 1, 1, 1]),
        home_of=np.array([3, 4, 5]),
        paths=[[3, 0, 1, 3], [4, 1, 2, 4], [5, 2, 0, 5]],
        contagion_prob=np.array([1.0, 1.0]),
    )


# ---------------------------------------------------------------------------
# The city (PDF p. 51)
# ---------------------------------------------------------------------------


def test_the_printed_place_counts_give_243_places_and_100_houses():
    """PDF p. 51: '10 tipos diferentes de lugares' with counts

        [16 19 15 11 20 13 15 14 20 100]

    and '100 habitantes', to whom 'tambien se les ha asignado un lugar el cual
    sera la casa que es origen y final de los caminos'.  The tenth count being
    exactly the number of habitants is what identifies that type as the houses.
    """
    assert THESIS_PLACE_COUNTS == (16, 19, 15, 11, 20, 13, 15, 14, 20, 100)
    assert sum(THESIS_PLACE_COUNTS) == 243
    assert THESIS_PLACE_COUNTS[-1] == 100

    city = build_city(rng=0)
    assert (city.n_places, city.n_habitants, city.n_types) == (243, 100, 10)
    assert np.array_equal(np.bincount(city.place_type), np.array(THESIS_PLACE_COUNTS))


def test_each_habitant_owns_exactly_one_house_of_the_last_type():
    """PDF p. 51: one house per habitant, which makes home_of a bijection onto
    the hundred places of the tenth type -- the light-green nodes drawn apart
    from the rest in fig. 50 (PDF p. 52)."""
    city = build_city(rng=7)
    houses = city.home_of
    assert houses.size == 100
    assert np.unique(houses).size == 100
    assert set(city.place_type[houses].tolist()) == {9}
    assert sorted(houses.tolist()) == list(range(143, 243))


def test_the_contagion_probabilities_are_the_printed_percentages():
    """PDF p. 51: 'a cada lugar se le ha asociado una probabilidad de contagio
    aleatoria [92 60 99 61 88 84 56 72 98 97]', one per place *type*."""
    assert THESIS_CONTAGION_PCT == (92, 60, 99, 61, 88, 84, 56, 72, 98, 97)
    city = build_city(rng=0)
    assert city.contagion_prob == pytest.approx(np.array(THESIS_CONTAGION_PCT) / 100.0)
    assert city.contagion_prob.size == city.n_types


def test_every_route_is_a_closed_walk_through_three_to_five_further_places():
    """PDF p. 51: 'a cada habitante un camino aleatorio que pase por como
    minimo 3 lugares y como maximo por 5', starting and ending at his house.

    The default reads them as places besides the house; the other reading is
    checked too.
    """
    city = build_city(rng=3)
    lengths = set()
    for h, route in enumerate(city.paths):
        home = int(city.home_of[h])
        assert route[0] == home and route[-1] == home
        assert route.count(home) == 2, "the house is the origin and the end, not a stop"
        assert len(set(route)) == len(route) - 1, "a route visits distinct places"
        lengths.add(len(city.visited(h)) - 1)
    assert lengths == {3, 4, 5}

    counted = build_city(rng=3, home_counts_toward_length=True)
    assert {len(counted.visited(h)) for h in range(counted.n_habitants)} == {3, 4, 5}


def test_houses_can_be_kept_off_the_routes():
    """allow_homes_on_paths=False keeps other people's houses off the routes."""
    city = build_city(rng=11, allow_homes_on_paths=False)
    houses = set(city.home_of.tolist())
    for h, route in enumerate(city.paths):
        assert set(route) & houses == {int(city.home_of[h])}


# ---------------------------------------------------------------------------
# The drawn network (figs. 48-50, PDF p. 52)
# ---------------------------------------------------------------------------


def test_the_place_graph_joins_consecutive_stops_of_every_route(thesis_city):
    """PDF p. 52, fig. 48: 'Los nodos representan lugares y las ramas los
    caminos seguidos por los habitantes de la red'."""
    graph = thesis_city.place_graph()
    assert graph.n == thesis_city.n_places

    for route in thesis_city.paths:
        for a, b in zip(route[:-1], route[1:], strict=True):
            assert graph.has_edge(a, b)

    drawn = {(min(i, j), max(i, j)) for i, j, _ in graph.edges()}
    assert drawn == set(thesis_city.place_edges())


def test_isolated_places_exist_because_nobody_walks_through_them(thesis_city):
    """100 routes do not cover 243 places, so some places have no link."""
    graph = thesis_city.place_graph()
    assert (graph.degrees() == 0).sum() > 0


# ---------------------------------------------------------------------------
# The dynamics: the bipartite rule, not adjacency (PDF p. 52)
# ---------------------------------------------------------------------------


def test_worked_three_habitant_city():
    """The mechanism of PDF p. 52 in six places.

    'Dejara restos de su infeccion en todos los lugares por los que pase',
    so seeding habitant 0 contaminates his whole route at once; then 'todos los
    habitantes que pasan por esos lugares quedaran infectados', which is both
    of the others, since each shares one place with him.
    """
    run = propagate_places(_three_habitant_city(), 0)

    assert run.infected_people(0).tolist() == [0]
    assert run.contaminated_places(0).tolist() == [0, 1, 3]
    assert run.infected_people(1).tolist() == [0, 1, 2]
    assert run.contaminated_places(1).tolist() == [0, 1, 2, 3, 4, 5]
    assert run.steps == 1 and run.complete

    people, places = place_infection_curves(_three_habitant_city(), 0)
    assert people == pytest.approx([1 / 3, 1.0])
    assert places == pytest.approx([0.5, 1.0])


def test_the_place_model_is_not_the_node_model_on_the_place_graph():
    """PDF pp. 51-52.

    Section 5.4.1 moves the infection one branch per instant.  Here an infected
    habitant contaminates his entire route in one instant, however far apart its
    places lie on the drawn graph, because the contagion travels through the
    person and not along the branch.

    Two habitants, routes 0-2-3-0 and 1-4-3-5-1, sharing only place 3.  Place 1
    -- the second habitant's house -- is three branches away from place 0 on the
    place-graph, so the node model of 5.4.1 needs three iterations to reach it,
    while the place model contaminates it at the very first instant.
    """
    city = City(
        place_type=np.zeros(7, dtype=int),
        home_of=np.array([0, 1]),
        paths=[[0, 2, 3, 0], [1, 4, 3, 5, 1]],
        contagion_prob=np.array([1.0]),
    )

    run = propagate_places(city, 0)
    assert run.infected_at.tolist() == [0.0, 1.0]
    assert float(run.contaminated_at[1]) == 1.0

    node_model = propagate(city.place_graph(), 0)
    assert float(node_model.infected_at[1]) == 4.0, "instant 4, i.e. three iterations"
    assert float(node_model.infected_at[3]) == 2.0

    place_seven_is_on_no_route = 6
    assert not np.isfinite(run.contaminated_at[place_seven_is_on_no_route])


def test_the_infection_never_recedes(thesis_city):
    """The model has no recovery, so both sets only grow.  PDF p. 53 names 'el
    tiempo que un lugar puede estar infectado antes de que la infeccion muera' as
    a factor for a model with variable routes."""
    for seed in (0, 17, 55, 99):
        run = propagate_places(thesis_city, seed)
        assert np.all(np.diff(run.people_counts) >= 0)
        assert np.all(np.diff(run.place_counts) >= 0)
        for t in range(run.steps):
            assert set(run.infected_people(t).tolist()) <= set(run.infected_people(t + 1).tolist())
            assert set(run.contaminated_places(t).tolist()) <= set(
                run.contaminated_places(t + 1).tolist()
            )


def test_every_infected_habitant_has_contaminated_his_whole_route():
    """PDF p. 52: the closure that couples the two populations.  At every
    instant the contaminated set contains the union of the routes of everyone
    infected so far."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        city = build_city(rng=rng)
        run = propagate_places(city, rng=rng)
        for t in range(run.steps + 1):
            union: set[int] = set()
            for h in run.infected_people(t):
                union |= set(city.visited(int(h)).tolist())
            assert union <= set(run.contaminated_places(t).tolist())


def test_the_process_terminates_in_at_most_one_step_per_habitant():
    """Each instant that changes anything infects at least one habitant, so the
    run cannot outlast the population (fig. 52 ends at t = 5)."""
    rng = np.random.default_rng(99)
    for _ in range(50):
        city = build_city(rng=rng)
        run = propagate_places(city, rng=rng)
        assert run.steps <= city.n_habitants


# ---------------------------------------------------------------------------
# The reported run (figs. 51-53, PDF pp. 52-53)
# ---------------------------------------------------------------------------


def test_healthy_and_infected_always_sum_to_the_population(thesis_city):
    """PDF p. 53, fig. 52: 'cantidad de nodos' sanos e infectados.  Fig. 52 plots
    counts up to 100, the number of habitants; the habitants are counted."""
    run = propagate_places(thesis_city, 0)
    assert np.all(run.healthy_counts + run.people_counts == 100)
    assert run.people_counts[0] == 1
    assert run.healthy_counts[0] == 99


def test_the_infection_reaches_practically_the_whole_population():
    """PDF p. 53, fig. 52: the *sano* curve reaches 0 at t = 5 and the
    *infectado* curve reaches 100.

    In a random city a habitant occasionally walks a route that meets nobody
    else's; over 100 cities the mean final infected fraction is above 0.99.
    """
    rng = np.random.default_rng(20110930)
    finals = []
    for _ in range(100):
        city = build_city(rng=rng)
        finals.append(float(propagate_places(city, rng=rng).people_curve[-1]))
    assert np.mean(finals) > 0.99
    assert np.mean([f == 1.0 for f in finals]) > 0.9


def test_saturation_takes_a_handful_of_instants():
    """PDF p. 53, fig. 52: the run is over by t = 5, having started from one
    infected habitant at t = 0.

    Asserted: a single habitant infects the city within a handful of
    instants.
    """
    rng = np.random.default_rng(2011)
    steps = [propagate_places(build_city(rng=rng), rng=rng).steps for _ in range(60)]
    assert min(steps) >= 3
    assert max(steps) <= 8
    assert 3 <= float(np.mean(steps)) <= 6


def test_half_the_population_is_infected_around_the_third_instant():
    """PDF p. 53, fig. 52: the *infectado* curve crosses the *sano* curve
    between t = 2 and t = 3.  Asserted: half the city is infected within the
    first three instants and not at the first one.
    """
    rng = np.random.default_rng(1234)
    crossings = []
    for _ in range(60):
        city = build_city(rng=rng)
        counts = propagate_places(city, rng=rng).people_counts
        crossings.append(int(np.argmax(counts >= 50)))
    assert all(c >= 2 for c in crossings), "the first instant never infects half the city"
    assert np.mean([c in (2, 3) for c in crossings]) > 0.9
    assert 2.0 <= float(np.mean(crossings)) <= 3.5


def test_the_first_instant_infects_only_a_small_minority():
    """PDF p. 53, figs. 52 and 53: four habitants are infected at t = 1, i.e.
    'el porcentaje de nodos infectados' is 0.04 there.

    Asserted: the first instant infects a small minority of the city, an order
    of magnitude below saturation.
    """
    rng = np.random.default_rng(5150)
    first = []
    for _ in range(60):
        city = build_city(rng=rng)
        first.append(int(propagate_places(city, rng=rng).people_counts[1]))
    assert all(1 < f <= 30 for f in first)
    assert float(np.median(first)) < 20


def test_the_infected_fraction_curve_rises_from_near_zero_to_one():
    """PDF p. 53, fig. 53: '% de nodos infectados' against t, from 0.04 at
    t = 1 to 1 at t = 5.  The shape -- strictly rising until saturation, then
    flat -- is what is asserted."""
    city = build_city(rng=20110930)
    people, places = place_infection_curves(city, 0)
    assert people[0] == pytest.approx(0.01)
    assert people[-1] == pytest.approx(1.0)
    assert np.all(np.diff(people) > 0)
    assert np.all(np.diff(places) >= 0)
    assert places[-1] < 1.0, "no route passes through every place in the city"


# ---------------------------------------------------------------------------
# The two readings of the contagion probabilities (PDF pp. 51-52)
# ---------------------------------------------------------------------------


def test_the_default_rule_is_deterministic_and_ignores_the_random_state(thesis_city):
    """PDF p. 52: 'todos los habitantes que pasan por esos lugares quedaran
    infectados' -- the default run is a function of the city alone."""
    a = propagate_places(thesis_city, 0, rng=1)
    b = propagate_places(thesis_city, 0, rng=999)
    assert np.array_equal(a.people_counts, b.people_counts)
    assert np.array_equal(a.place_counts, b.place_counts)
    assert np.array_equal(a.infected_at, b.infected_at)


def test_zero_probabilities_freeze_the_infection_at_the_seed():
    """PDF p. 51 introduces the per-place probabilities as the handle on
    'lugares los cuales en caso de ser saneados dificultaran notablemente la
    propagacion'.  Sanitising every place is the limit of that: nothing spreads.
    """
    city = _three_habitant_city()
    harmless = City(city.place_type, city.home_of, city.paths, np.zeros(2))
    run = propagate_places(harmless, 0, use_probabilities=True, rng=3)
    assert run.people_counts.tolist() == [1]
    assert run.contaminated_places(run.steps).tolist() == [0, 1, 3]
    assert not run.complete


def test_probabilities_can_only_slow_the_infection_down(thesis_city):
    """The probabilistic reading of PDF p. 51 draws from the same exposed set
    as the deterministic reading of PDF p. 52 and keeps a subset of it, so no
    habitant is ever infected earlier and none is contaminated sooner.

    Deterministic relation, not a statistical one: it holds run by run.
    """
    reference = propagate_places(thesis_city, 0)
    for seed in (0, 1, 2, 3, 4):
        run = propagate_places(thesis_city, 0, use_probabilities=True, rng=seed)
        assert np.all(run.infected_at >= reference.infected_at)
        assert np.all(run.contaminated_at >= reference.contaminated_at)
        assert run.people_counts[1] <= reference.people_counts[1]


def test_a_random_habitant_is_seeded_when_none_is_named(thesis_city):
    """PDF p. 52: 'supongamos que una persona al azar es infectada'."""
    seeds = {propagate_places(thesis_city, rng=s).first_infected for s in range(30)}
    assert len(seeds) > 5
    assert all(0 <= s < thesis_city.n_habitants for s in seeds)


def test_max_steps_truncates_the_run(thesis_city):
    run = propagate_places(thesis_city, 0, max_steps=2)
    assert run.steps == 2
    assert not run.complete


# ---------------------------------------------------------------------------
# Provenance of the readings
# ---------------------------------------------------------------------------


def test_the_readings_of_the_section_are_recorded():
    """The readings that change the numbers are declared on the objects that
    take them."""
    assert "camino aleatorio" in build_city.source.reading
    assert "last type" in build_city.source.reading
    assert "use_probabilities" in propagate_places.source.reading
    assert "t = 0" in PlaceInfectionRun.source.reading
    assert City.place_graph.source.pdf_pages == "52"
