"""Adjacency-matrix network representation.

Section 2.2 of the thesis (*Matriz Adyacente*, PDF p. 6) takes the adjacency
matrix as *the* object of study: a network of :math:`N` nodes is the
:math:`N \\times N` matrix :math:`A` with :math:`A_{ij}=1` when nodes *i* and
*j* are linked and :math:`0` otherwise, symmetric because the networks are
undirected, and with a zero diagonal because a node is not linked to itself.
The later chapters build on that matrix directly -- distances by a Dijkstra-type
sweep of the matrix, degree as a row sum, propagation by testing the neighbours in each row.

So ``netmotion`` keeps the adjacency matrix as the primary representation
rather than an edge list or a NetworkX graph: the thesis's algorithms are written
in the language of matrix operations, and they are implemented as written.
:meth:`Network.to_networkx` exists for cross-validation and for layout, never as
the implementation of a thesis algorithm.

Weighted networks (sections 5.3 and 5.6, opinion weights and interbank
exposures) use the same class: :attr:`Network.matrix` then holds real weights
and :attr:`Network.adjacency` gives the 0/1 support.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .provenance import Origin, sourced

__all__ = ["Network"]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="6",
    section="2.2 Matriz Adyacente",
    note=(
        "Representation only. The thesis's example matrix is symmetric with a "
        "null diagonal for undirected, unweighted networks; the weighted variants of "
        "sections 5.3 and 5.6 reuse the same matrix with real entries."
    ),
)
@dataclass(eq=False)
class Network:
    """An undirected network stored as a dense adjacency/weight matrix.

    Parameters
    ----------
    matrix:
        ``(N, N)`` array.  Non-zero entry ``matrix[i, j]`` means *i* and *j*
        are linked; the value is the link weight (1.0 for an unweighted
        network).  The array is copied and cast to ``float``.
    directed:
        ``False`` (the default) enforces symmetry and a zero diagonal, which is
        the convention of the thesis.  Directed networks are supported with
        ``True``; the interbank model of section 5.6.4 keeps :math:`M`
        undirected and puts the direction in the antisymmetric :math:`W`.
    name:
        Free-form label.

    Notes
    -----
    Dense storage is deliberate.  The networks of the thesis are small enough
    that a dense float matrix costs little, and the
    matrix-product formulations of the original algorithms run at BLAS speed.
    Sparse storage would be faster asymptotically but would obscure the
    correspondence with the printed equations, which is the point of this
    package.
    """

    matrix: np.ndarray
    directed: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(f"adjacency matrix must be square, got shape {m.shape}")
        if not self.directed:
            if not np.allclose(m, m.T):
                raise ValueError(
                    "undirected Network requires a symmetric matrix; "
                    "pass directed=True for a directed network"
                )
        if np.any(np.diag(m) != 0):
            raise ValueError("self-links are not part of the thesis convention (A_ii = 0)")
        self.matrix = m

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, n: int, *, directed: bool = False, name: str = "") -> Network:
        """A network of *n* isolated nodes."""
        return cls(np.zeros((n, n)), directed=directed, name=name)

    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        edges: Sequence[tuple[int, int]] | Sequence[tuple[int, int, float]],
        *,
        directed: bool = False,
        name: str = "",
    ) -> Network:
        """Build from an edge list, optionally ``(i, j, weight)`` triples."""
        n = n_nodes
        m = np.zeros((n, n))
        for edge in edges:
            if len(edge) == 3:
                i, j, w = edge
            else:
                (i, j), w = edge, 1.0
            m[i, j] = w
            if not directed:
                m[j, i] = w
        return cls(m, directed=directed, name=name)

    @classmethod
    def from_networkx(cls, graph, *, weight: str | None = None, name: str = "") -> Network:
        """Convert a NetworkX graph.  Used for cross-validation, not for models."""
        import networkx as nx

        nodes = sorted(graph.nodes())
        m = nx.to_numpy_array(graph, nodelist=nodes, weight=weight)
        return cls(m, directed=graph.is_directed(), name=name or graph.name)

    # ------------------------------------------------------------------
    # basic structure
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of nodes, :math:`N` (the same as :attr:`n_nodes`)."""
        return self.matrix.shape[0]

    @property
    def n_nodes(self) -> int:
        """Number of nodes, :math:`N`, named as :attr:`n_edges` is."""
        return self.matrix.shape[0]

    def __len__(self) -> int:
        return self.n

    @property
    def adjacency(self) -> np.ndarray:
        """The 0/1 support of :attr:`matrix`, as ``float``."""
        return (self.matrix != 0).astype(float)

    @property
    def weighted(self) -> bool:
        """True when any link weight differs from 1."""
        nz = self.matrix[self.matrix != 0]
        return bool(nz.size) and not np.allclose(nz, 1.0)

    @property
    def n_edges(self) -> int:
        """Number of links.  Undirected links are counted once."""
        count = int(np.count_nonzero(self.matrix))
        return count if self.directed else count // 2

    def degrees(self) -> np.ndarray:
        """Degree of every node, :math:`k_i = \\sum_j A_{ij}` (integer array)."""
        return self.adjacency.sum(axis=1).astype(int)

    def strengths(self) -> np.ndarray:
        """Weighted degree :math:`s_i = \\sum_j w_{ij}` (float array)."""
        return self.matrix.sum(axis=1)

    def neighbors(self, i: int) -> np.ndarray:
        """Indices of the nodes linked to *i*."""
        return np.flatnonzero(self.matrix[i])

    def has_edge(self, i: int, j: int) -> bool:
        return bool(self.matrix[i, j] != 0)

    def add_edge(self, i: int, j: int, weight: float = 1.0) -> None:
        """Add or re-weight a link, in place."""
        if i == j:
            raise ValueError("self-links are not part of the thesis convention")
        self.matrix[i, j] = weight
        if not self.directed:
            self.matrix[j, i] = weight

    def remove_edge(self, i: int, j: int) -> None:
        """Remove a link, in place."""
        self.matrix[i, j] = 0.0
        if not self.directed:
            self.matrix[j, i] = 0.0

    def edges(self) -> Iterator[tuple[int, int, float]]:
        """Yield ``(i, j, weight)``; for undirected networks only ``i < j``."""
        rows, cols = np.nonzero(self.matrix)
        for i, j in zip(rows, cols, strict=True):
            if self.directed or i < j:
                yield int(i), int(j), float(self.matrix[i, j])

    # ------------------------------------------------------------------
    # derived networks
    # ------------------------------------------------------------------

    def subgraph(self, nodes: Sequence[int]) -> Network:
        """The network induced on *nodes*, re-indexed from 0."""
        idx = np.asarray(nodes, dtype=int)
        return Network(
            self.matrix[np.ix_(idx, idx)].copy(),
            directed=self.directed,
            name=f"{self.name} subgraph" if self.name else "",
        )

    def without_nodes(self, nodes: Sequence[int]) -> Network:
        """A copy with *nodes* isolated (links cut, indices preserved).

        Node indices are kept so that a vulnerability score computed on the
        reduced network stays comparable with the full one, which is what the
        node-removal method of section 5.5.1.1 needs.
        """
        m = self.matrix.copy()
        idx = np.asarray(nodes, dtype=int)
        m[idx, :] = 0.0
        m[:, idx] = 0.0
        return Network(m, directed=self.directed, name=self.name)

    def copy(self) -> Network:
        return Network(self.matrix.copy(), directed=self.directed, name=self.name)

    def to_networkx(self):
        """A NetworkX view, for cross-validation and for drawing only."""
        import networkx as nx

        creator = nx.DiGraph if self.directed else nx.Graph
        g = nx.from_numpy_array(self.matrix, create_using=creator())
        g.name = self.name
        return g

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "directed" if self.directed else "undirected"
        w = ", weighted" if self.weighted else ""
        label = f" {self.name!r}" if self.name else ""
        return f"<Network{label} N={self.n} E={self.n_edges} {kind}{w}>"
