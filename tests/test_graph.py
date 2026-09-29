"""Unit tests for the adjacency-matrix representation (thesis section 2.2)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from netmotion import Network

# A square: 0-1-2-3-0.  Every node has degree 2; four links.
SQUARE = Network.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)], name="square")

# A path 0-1-2 with a pendant 3 on node 1.
STAR_PATH = Network.from_edges(4, [(0, 1), (1, 2), (1, 3)], name="star-path")


def test_size_and_edge_count():
    assert SQUARE.n == 4
    assert len(SQUARE) == 4
    assert SQUARE.n_edges == 4
    assert STAR_PATH.n_edges == 3


def test_degrees_are_row_sums():
    assert np.array_equal(SQUARE.degrees(), [2, 2, 2, 2])
    assert np.array_equal(STAR_PATH.degrees(), [1, 3, 1, 1])


def test_matrix_is_symmetric_with_zero_diagonal():
    """The convention of thesis section 2.2."""
    assert np.allclose(SQUARE.matrix, SQUARE.matrix.T)
    assert np.all(np.diag(SQUARE.matrix) == 0)


def test_asymmetric_matrix_is_rejected_unless_directed():
    m = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        Network(m)
    assert Network(m, directed=True).n_edges == 1


def test_self_links_are_rejected():
    with pytest.raises(ValueError, match="self-links"):
        Network(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="self-links"):
        SQUARE.copy().add_edge(0, 0)


def test_non_square_matrix_is_rejected():
    with pytest.raises(ValueError, match="square"):
        Network(np.zeros((2, 3)))


def test_neighbors_and_has_edge():
    assert np.array_equal(STAR_PATH.neighbors(1), [0, 2, 3])
    assert STAR_PATH.has_edge(1, 3)
    assert not STAR_PATH.has_edge(0, 3)


def test_edges_yields_each_undirected_link_once():
    assert list(SQUARE.edges()) == [(0, 1, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]


def test_add_and_remove_edge_keep_symmetry():
    g = SQUARE.copy()
    g.add_edge(0, 2)
    assert g.n_edges == 5
    assert g.matrix[2, 0] == 1.0
    g.remove_edge(0, 2)
    assert g.n_edges == 4
    assert g.matrix[2, 0] == 0.0


def test_weighted_flag_and_strengths():
    w = Network.from_edges(3, [(0, 1, 2.5), (1, 2, 0.5)])
    assert w.weighted
    assert not SQUARE.weighted
    assert np.allclose(w.strengths(), [2.5, 3.0, 0.5])
    assert np.array_equal(w.degrees(), [1, 2, 1])


def test_adjacency_is_the_binary_support_of_a_weighted_matrix():
    w = Network.from_edges(3, [(0, 1, 2.5), (1, 2, 0.5)])
    assert set(np.unique(w.adjacency)) <= {0.0, 1.0}
    assert np.array_equal(w.adjacency.sum(axis=1), [1, 2, 1])


def test_subgraph_reindexes_from_zero():
    sub = SQUARE.subgraph([0, 1, 2])
    assert sub.n == 3
    assert sub.n_edges == 2
    assert np.array_equal(sub.degrees(), [1, 2, 1])


def test_without_nodes_isolates_but_keeps_indices():
    """Node-removal vulnerability (section 5.5.1.1) needs stable indices."""
    cut = SQUARE.without_nodes([0])
    assert cut.n == 4
    assert cut.n_edges == 2
    assert np.array_equal(cut.degrees(), [0, 1, 2, 1])
    assert SQUARE.n_edges == 4, "without_nodes must not mutate the original"


@pytest.mark.crossvalidation
def test_roundtrip_through_networkx():
    g = SQUARE.to_networkx()
    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 4
    back = Network.from_networkx(g)
    assert np.allclose(back.matrix, SQUARE.matrix)


@pytest.mark.crossvalidation
def test_degrees_agree_with_networkx():
    g = nx.gnp_random_graph(60, 0.1, seed=7)
    net = Network.from_networkx(g)
    expected = np.array([d for _, d in sorted(g.degree())])
    assert np.array_equal(net.degrees(), expected)
    assert net.n_edges == g.number_of_edges()


@pytest.mark.crossvalidation
def test_directed_roundtrip_preserves_asymmetry():
    """A directed network keeps its asymmetric matrix through a round trip."""
    m = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 0.0]])
    net = Network(m, directed=True)
    assert net.n_edges == 3
    back = Network.from_networkx(net.to_networkx(), weight="weight")
    assert np.allclose(back.matrix, m)
