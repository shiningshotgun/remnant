"""Exact solver and Markov chain random walk analysis for the Reach of Vesper.

Derives:
1. Shortest escape path = 42 strides (with actions).
2. Unreachability without the Lamp-Key.
3. Shortest escape collecting pearls first = 62 strides.
4. Finite-horizon uniform random walk escape probabilities within 54 strides.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Set, Any
from remnant_core.worlds.reach import (
    START_POS,
    TIDEGATE_POS,
    REACH_LOOP_LIMIT,
    PhysicalState,
    physical_transition,
)


def get_physical_successors(state: PhysicalState, has_key_allowed: bool = True) -> List[Tuple[int, PhysicalState]]:
    """Side-effect-free physical transitions from state under actions 0..3 using shared physical_transition."""
    successors = []
    for a in range(4):
        next_state, reward, terminal, invalid_reason, pearls_picked, key_picked = physical_transition(
            state, a, has_key_allowed=has_key_allowed
        )
        successors.append((a, next_state))
    return successors


def solve_reach_shortest_escape(
    has_key_allowed: bool = True,
    collect_pearls_first: bool = False,
) -> Tuple[Optional[int], Optional[List[Tuple[int, int]]], Optional[List[int]]]:
    """BFS over physical state space to find shortest escape path to Tidegate."""
    start_state: PhysicalState = (START_POS, False, False)
    queue = deque([(start_state, [START_POS], [])])
    visited: Set[PhysicalState] = {start_state}

    while queue:
        curr_state, path, actions = queue.popleft()
        pos, has_key, pearls = curr_state

        if pos == TIDEGATE_POS:
            if collect_pearls_first and not pearls:
                continue
            return len(actions), path, actions

        for a, next_state in get_physical_successors(curr_state, has_key_allowed=has_key_allowed):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state[0]], actions + [a]))

    return None, None, None


def propagate_random_walk_probabilities(
    initial_state: Any,
    transition_fn: Any,
    is_terminal_fn: Any,
    max_steps: int,
    action_count: int = 4,
) -> Dict[int, float]:
    """Generic finite-horizon uniform random walk probability propagation.
    
    Args:
        initial_state: Starting state
        transition_fn: Callable(state, action) -> next_state
        is_terminal_fn: Callable(state) -> bool
        max_steps: Maximum propagation horizon
        action_count: Number of equiprobable actions (default 4)
        
    Returns:
        Dict mapping step index (1..max_steps) to cumulative absorption probability.
    """
    # Enumerate reachable states
    reachable: List[Any] = []
    queue = deque([initial_state])
    seen: Set[Any] = {initial_state}

    while queue:
        s = queue.popleft()
        reachable.append(s)
        if is_terminal_fn(s):
            continue  # Terminal / absorbing
        for a in range(action_count):
            sn = transition_fn(s, a)
            if sn not in seen:
                seen.add(sn)
                queue.append(sn)

    state_to_idx = {s: i for i, s in enumerate(reachable)}
    n_states = len(reachable)

    dist = [0.0] * n_states
    dist[state_to_idx[initial_state]] = 1.0

    prob_per_action = 1.0 / action_count
    step_escape_prob: Dict[int, float] = {}

    for step in range(1, max_steps + 1):
        next_dist = [0.0] * n_states
        for idx, s in enumerate(reachable):
            p = dist[idx]
            if p == 0.0:
                continue
            if is_terminal_fn(s):
                next_dist[idx] += p
                continue

            for a in range(action_count):
                sn = transition_fn(s, a)
                next_dist[state_to_idx[sn]] += p * prob_per_action

        dist = next_dist
        escape_p = sum(dist[idx] for idx, s in enumerate(reachable) if is_terminal_fn(s))
        step_escape_prob[step] = escape_p

    return step_escape_prob


def compute_reach_random_walk_escape_probabilities(
    max_steps: int = REACH_LOOP_LIMIT,
) -> Dict[int, float]:
    """Compute exact discrete-time distribution of time-to-escape under uniform random walk on ReachWorld."""
    start_state: PhysicalState = (START_POS, False, False)

    def transition_fn(state: PhysicalState, action: int) -> PhysicalState:
        next_state, _, _, _, _, _ = physical_transition(state, action, has_key_allowed=True)
        return next_state

    def is_terminal_fn(state: PhysicalState) -> bool:
        return state[0] == TIDEGATE_POS

    return propagate_random_walk_probabilities(
        initial_state=start_state,
        transition_fn=transition_fn,
        is_terminal_fn=is_terminal_fn,
        max_steps=max_steps,
        action_count=4,
    )
