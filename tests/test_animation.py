"""The animation module: frames, the four animations, and saving."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

from netmotion import Network  # noqa: E402
from netmotion.dynamics import random_initial_states, run_hopfield  # noqa: E402
from netmotion.finance import InterbankNetwork, bank_capital, four_state_cascade  # noqa: E402
from netmotion.finance.bank_growth import strategy_growth  # noqa: E402
from netmotion.networks import barabasi_albert, watts_strogatz  # noqa: E402
from netmotion.propagation import propagate  # noqa: E402
from netmotion.visualization import (  # noqa: E402
    animate_cascade,
    animate_frames,
    animate_growth,
    animate_propagation,
    animate_states,
    cascade_frames,
    community_layout,
    growth_frames,
    propagation_frames,
    save_animation,
    to_html,
)
from netmotion.visualization.animation import close  # noqa: E402


def _plays(anim: FuncAnimation) -> None:
    """Every frame renders: the player is built from all of them."""
    html = anim.to_jshtml()
    close(anim)
    assert "<img" in html or "frames" in html


@pytest.fixture(scope="module")
def banks():
    cn, _ = strategy_growth(n_communities=4, steps=40, rng=3)
    return InterbankNetwork.from_community_network(cn)


def test_growth_frames_add_one_node_a_frame():
    frames = growth_frames(6, initial_nodes=3)
    assert frames.shape == (4, 6)
    assert frames.sum(axis=1).tolist() == [3, 4, 5, 6]


def test_propagation_frames_follow_the_infection_times():
    g = watts_strogatz(30, 4, 0.1, rng=2)
    run = propagate(g, 5)
    frames = propagation_frames(run)
    assert frames.shape == (run.curve.size, g.n)
    # the fraction infected in each frame is the curve the book plots
    assert np.allclose((frames > 0).mean(axis=1), run.curve)
    assert frames[0, 5] == 2 and (frames[-1] > 0).all()


def test_cascade_frames_end_on_the_fallen_banks(banks):
    run = four_state_cascade(banks.network, banks.largest_bank(), bank_capital(banks.network),
                        reserves=20.0, reaction_threshold=30.0)
    frames = cascade_frames(run)
    assert frames.shape == (run.steps + 1, banks.n_banks)
    assert np.flatnonzero(frames[-1]).tolist() == run.crashed.tolist()
    assert (np.diff(frames.sum(axis=1)) >= 0).all()


def test_community_layout_keeps_each_community_together():
    community = np.repeat([0, 1, 2], 5)
    pos = community_layout(community)
    inside = np.linalg.norm(pos[0] - pos[4])
    across = min(np.linalg.norm(pos[0] - pos[j]) for j in range(5, 15))
    assert inside < across


def test_animate_frames_checks_its_colours():
    g = Network.from_edges(3, [(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="one column per node"):
        animate_frames(g, np.full((2, 4), "red", dtype=object), layout="circular")


def test_the_four_animations_play(banks):
    g = barabasi_albert(25, links_per_new_node=2, rng=1)
    anim = animate_growth(g, initial_nodes=3, layout="circular")
    assert anim.frame_count == 23
    _plays(anim)

    anim = animate_propagation(watts_strogatz(20, 4, 0.1, rng=1), 0, layout="circular")
    _plays(anim)

    ring = watts_strogatz(20, 4, 0.1, rng=1)
    run = run_hopfield(ring, random_initial_states(20, 8, rng=2), steps=5)
    anim = animate_states(ring, run)
    assert anim.frame_count == run.states.shape[0]
    _plays(anim)

    run = four_state_cascade(banks.network, banks.largest_bank(), bank_capital(banks.network),
                        reserves=20.0, reaction_threshold=30.0)
    anim = animate_cascade(banks, run)
    assert anim.frame_count == run.steps + 1
    _plays(anim)


def test_save_animation_writes_a_gif(tmp_path):
    g = Network.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    anim = animate_propagation(g, 0, layout="circular")
    path = save_animation(anim, tmp_path / "infection.gif", fps=4)
    close(anim)
    assert path.exists() and path.stat().st_size > 0
    with pytest.raises(ValueError, match="gif"):
        save_animation(anim, tmp_path / "infection.png")


def test_to_html_works_without_ipython(monkeypatch):
    # a plain script, where IPython is not installed: the player comes back as HTML text
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    anim = animate_propagation(Network.from_edges(3, [(0, 1), (1, 2)]), 0, layout="circular")
    html = to_html(anim)
    assert isinstance(html, str) and "<script" in html
