from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Callable, FrozenSet, Iterable, Iterator, Optional, Set


def _neighbors(node: int, grid_size: int) -> Iterator[int]:
    r, c = (node - 1) // grid_size, (node - 1) % grid_size
    if r > 0:
        yield node - grid_size
    if r < grid_size - 1:
        yield node + grid_size
    if c > 0:
        yield node - 1
    if c < grid_size - 1:
        yield node + 1


def _manhattan_distance(n1: int, n2: int, grid_size: int) -> int:
    r1, c1 = (n1 - 1) // grid_size, (n1 - 1) % grid_size
    r2, c2 = (n2 - 1) // grid_size, (n2 - 1) % grid_size
    return abs(r1 - r2) + abs(c1 - c2)


def _calculate_distance_uncached(
    n1: int,
    n2: int,
    *,
    grid_size: int,
    barriers: FrozenSet[int],
) -> int:
    if n1 == n2:
        return 0

    max_node = grid_size * grid_size
    if not (1 <= n1 <= max_node and 1 <= n2 <= max_node):
        raise ValueError(f"Node out of bounds for grid {grid_size}x{grid_size}: {n1}, {n2}")

    if not barriers:
        return _manhattan_distance(n1, n2, grid_size)

    if n1 in barriers or n2 in barriers:
        raise ValueError(f"No path: start/end is a barrier (start={n1}, end={n2})")

    q: deque[int] = deque([n1])
    dist: dict[int, int] = {n1: 0}

    while q:
        cur = q.popleft()
        d = dist[cur]
        for nb in _neighbors(cur, grid_size):
            if nb in barriers or nb in dist:
                continue
            nd = d + 1
            if nb == n2:
                return nd
            dist[nb] = nd
            q.append(nb)

    raise ValueError(f"No path found from {n1} to {n2} with barriers={sorted(barriers)}")


def make_calculate_distance(
    default_grid_size: int,
    default_barriers: Iterable[int] | Set[int] | FrozenSet[int],
) -> Callable[[int, int, int, Optional[Iterable[int]]], int]:

    default_barriers_frozen: FrozenSet[int] = frozenset(int(b) for b in default_barriers)

    @lru_cache(maxsize=None)
    def _cached(n1: int, n2: int) -> int:
        return _calculate_distance_uncached(
            int(n1), int(n2), grid_size=int(default_grid_size), barriers=default_barriers_frozen
        )

    def calculate_distance(
        node1: int,
        node2: int,
        grid_size: int = default_grid_size,
        barriers: Optional[Iterable[int]] = None,
    ) -> int:
        n1 = int(node1)
        n2 = int(node2)

        if int(grid_size) != int(default_grid_size):
            barriers_set = frozenset(int(b) for b in (barriers or ()))
            return _calculate_distance_uncached(n1, n2, grid_size=int(grid_size), barriers=barriers_set)

        if barriers is None:
            return _cached(n1, n2)

        barriers_set = frozenset(int(b) for b in barriers)
        if barriers_set == default_barriers_frozen:
            return _cached(n1, n2)

        return _calculate_distance_uncached(n1, n2, grid_size=int(grid_size), barriers=barriers_set)

    return calculate_distance
