from __future__ import annotations

import random
from typing import List


AdjacencyMatrix = List[List[int]]


def line_topology(n: int) -> AdjacencyMatrix:
    """
    Create a line (path) topology with n agents:
    0 -- 1 -- 2 -- ... -- (n-1)

    Undirected edges are modeled as two directed edges (i <-> j).
    """
    if n <= 0:
        raise ValueError("n must be positive")

    adj = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n - 1):
        adj[i][i + 1] = 1
        adj[i + 1][i] = 1

    return adj


def ring_topology(n: int) -> AdjacencyMatrix:
    """
    Create a ring topology with n agents:
    0 -- 1 -- 2 -- ... -- (n-1) -- 0
    """
    if n <= 1:
        raise ValueError("n must be greater than 1")

    adj = line_topology(n)
    # connect end to start
    adj[0][n - 1] = 1
    adj[n - 1][0] = 1
    return adj


def complete_topology(n: int) -> AdjacencyMatrix:
    """
    Create a complete graph topology: every node connected to every other.
    No self-loops.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    adj = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i][j] = 1
    return adj


def random_connected_topology(n: int, extra_edges: int = 0, seed: int | None = None) -> AdjacencyMatrix:
    """
    Create a random connected undirected graph on n nodes.

    Construction:
      1. Build a random spanning tree to guarantee connectivity.
      2. Add `extra_edges` random undirected edges.

    Args:
        n: number of nodes (agents).
        extra_edges: number of additional random edges to add.
        seed: optional random seed for reproducibility.

    Returns:
        Adjacency matrix (n x n) with 0/1 entries.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    if seed is not None:
        random.seed(seed)

    # Start with no edges
    adj = [[0 for _ in range(n)] for _ in range(n)]

    # Step 1: random spanning tree
    # We connect each new node k to a random previous node in [0, k-1]
    for k in range(1, n):
        j = random.randint(0, k - 1)
        adj[k][j] = 1
        adj[j][k] = 1

    # Step 2: add extra undirected edges
    possible_edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if adj[i][j] == 0
    ]
    random.shuffle(possible_edges)

    for _ in range(extra_edges):
        if not possible_edges:
            break
        i, j = possible_edges.pop()
        adj[i][j] = 1
        adj[j][i] = 1

    return adj


def print_adjacency(adj: AdjacencyMatrix) -> None:
    """Pretty-print an adjacency matrix."""
    for row in adj:
        print(row)


if __name__ == "__main__":
    # Simple manual test when you run:
    #   python topology_utils.py
    n = 5

    print("Line topology:")
    print_adjacency(line_topology(n))
    print()

    print("Ring topology:")
    print_adjacency(ring_topology(n))
    print()

    print("Complete topology:")
    print_adjacency(complete_topology(n))
    print()

    print("Random connected topology (with 2 extra edges):")
    print_adjacency(random_connected_topology(n, extra_edges=2, seed=0))