"""Exact BFS solver and finite-horizon random walk Markov chain analyzer."""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple, FrozenSet

from remnant_core.config import (
    GRID_HEIGHT,
    GRID_WIDTH,
    START_POS,
    WALLS,
    ITEM_BAIT,
    ITEM_KEY,
    BAIT_POS,
    KEY_POS,
    GATE_POS,
    EXIT_POS,
    ACTIONS,
)

State = Tuple[Tuple[int, int], FrozenSet[int]]


def solve_shortest_escape(
    has_key_allowed: bool = True,
    visit_bait_first: bool = False,
    start_pos: Tuple[int, int] = START_POS,
) -> Tuple[Optional[int], Optional[List[Tuple[int, int]]], Optional[List[int]]]:
    """Find the shortest escape path using exact Breadth-First Search.
    
    Args:
        has_key_allowed: If False, the key cannot be acquired (testing structural necessity).
        visit_bait_first: If True, path must visit the bait before escaping.
        start_pos: Initial agent position.
        
    Returns:
        (min_steps, path_positions, actions) or (None, None, None) if unreachable.
    """
    initial_inventory: FrozenSet[int] = frozenset()
    initial_state: State = (start_pos, initial_inventory)

    # Queue holds: (state, path_positions, actions)
    q: deque = deque([(initial_state, [start_pos], [])])
    visited: Set[State] = {initial_state}

    while q:
        (pos, inv), path, acts = q.popleft()

        # Goal check
        if pos == EXIT_POS:
            if not visit_bait_first or (ITEM_BAIT in inv):
                return len(acts), path, acts

        for a_idx, (dr, dc) in enumerate(ACTIONS):
            cand = (pos[0] + dr, pos[1] + dc)
            in_bounds = (0 <= cand[0] < GRID_HEIGHT) and (0 <= cand[1] < GRID_WIDTH)

            if in_bounds and (cand not in WALLS):
                if cand == GATE_POS and (ITEM_KEY not in inv):
                    nxt_pos = pos
                else:
                    nxt_pos = cand
            else:
                nxt_pos = pos

            nxt_inv = set(inv)
            if nxt_pos == KEY_POS and has_key_allowed:
                nxt_inv.add(ITEM_KEY)
            if nxt_pos == BAIT_POS:
                nxt_inv.add(ITEM_BAIT)

            nxt_state: State = (nxt_pos, frozenset(nxt_inv))

            if nxt_state not in visited:
                visited.add(nxt_state)
                q.append((nxt_state, path + [nxt_pos], acts + [a_idx]))

    return None, None, None


def compute_random_walk_escape_probabilities(max_steps: int = 18) -> Dict[int, float]:
    """Compute exact escape probability distribution of an unbiased random walk.
    
    Under uniform random action selection (p=0.25 per cardinal action), invalid moves
    (walls, boundaries, locked gate) leave the agent in place. The exit is absorbing.
    
    Returns:
        Dict mapping step index (1..max_steps) to cumulative escape probability.
    """
    # Enumerate all valid states
    open_cells = [
        (r, c)
        for r in range(GRID_HEIGHT)
        for c in range(GRID_WIDTH)
        if (r, c) not in WALLS
    ]
    possible_inventories = [
        frozenset(),
        frozenset({ITEM_BAIT}),
        frozenset({ITEM_KEY}),
        frozenset({ITEM_BAIT, ITEM_KEY}),
    ]
    all_states: List[State] = [
        (pos, inv) for pos in open_cells for inv in possible_inventories
    ]

    # Current probability distribution over states
    dist: Dict[State, float] = {s: 0.0 for s in all_states}
    dist[(START_POS, frozenset())] = 1.0

    cumulative_escape_probs: Dict[int, float] = {}

    for step in range(1, max_steps + 1):
        next_dist: Dict[State, float] = {s: 0.0 for s in all_states}

        for (pos, inv), prob in dist.items():
            if prob == 0.0:
                continue

            # Absorbing exit portal: stays at exit
            if pos == EXIT_POS:
                next_dist[(pos, inv)] += prob
                continue

            for dr, dc in ACTIONS:
                cand = (pos[0] + dr, pos[1] + dc)
                in_bounds = (0 <= cand[0] < GRID_HEIGHT) and (0 <= cand[1] < GRID_WIDTH)

                if in_bounds and (cand not in WALLS):
                    if cand == GATE_POS and (ITEM_KEY not in inv):
                        nxt_pos = pos
                    else:
                        nxt_pos = cand
                else:
                    nxt_pos = pos

                nxt_inv = set(inv)
                if nxt_pos == KEY_POS:
                    nxt_inv.add(ITEM_KEY)
                if nxt_pos == BAIT_POS:
                    nxt_inv.add(ITEM_BAIT)

                next_dist[(nxt_pos, frozenset(nxt_inv))] += prob * 0.25

        dist = next_dist
        # Sum probability across all states at EXIT_POS
        p_escape = sum(prob for (pos, inv), prob in dist.items() if pos == EXIT_POS)
        cumulative_escape_probs[step] = p_escape

    return cumulative_escape_probs
