"""Fixed discrete loop world for Remnant.

Implements the single frozen 5x5 topology, locked gate, item mechanics,
and exact reset semantics.
"""

from typing import Tuple, FrozenSet
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
    STEP_COST,
    BAIT_REWARD,
    KEY_REWARD,
    EXIT_REWARD,
    LOOP_LIMIT_L,
    ACTIONS,
)

State = Tuple[Tuple[int, int], FrozenSet[int]]


class LoopWorld:
    """Fixed 5x5 grid world subject to external time-loop resets."""

    def __init__(self, max_steps: int = LOOP_LIMIT_L):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.walls = WALLS
        self.max_steps = max_steps
        self.pos: Tuple[int, int] = START_POS
        self.inventory: FrozenSet[int] = frozenset()
        self.step_count: int = 0
        self.escaped: bool = False

    def reset(self, start: Tuple[int, int] = START_POS) -> State:
        """Physical loop reset: world returns to start position and re-arms items.
        
        The agent receives no explicit loop marker or timer; it strictly observes s_0.
        """
        self.pos = start
        self.inventory = frozenset()
        self.step_count = 0
        self.escaped = False
        return self.state()

    def state(self) -> State:
        """Return the Markov observation tuple: ((r, c), frozenset(items))."""
        return (self.pos, self.inventory)

    def step(self, action: int) -> Tuple[State, float, bool, bool]:
        """Execute one environment step.
        
        Returns:
            next_state: Tuple[Tuple[int, int], FrozenSet[int]]
            reward: float (scalar reward received)
            terminal: bool (True only on escape at exit portal)
            truncated: bool (True when loop step limit L is reached without escape)
        """
        if self.escaped:
            raise RuntimeError("Cannot step in an escaped world; must call reset()")

        dr, dc = ACTIONS[action]
        cand = (self.pos[0] + dr, self.pos[1] + dc)

        # Movement physics: boundaries, walls, and locked gate
        in_bounds = (0 <= cand[0] < self.height) and (0 <= cand[1] < self.width)
        if in_bounds and (cand not in self.walls):
            if cand == GATE_POS and (ITEM_KEY not in self.inventory):
                # Locked gate leaves agent in place
                pass
            else:
                self.pos = cand
        # Invalid moves leave the agent in place

        reward = STEP_COST
        terminal = False

        # Interactive item collection (re-armed each loop)
        if self.pos == BAIT_POS and (ITEM_BAIT not in self.inventory):
            reward += BAIT_REWARD
            self.inventory = self.inventory | {ITEM_BAIT}

        if self.pos == KEY_POS and (ITEM_KEY not in self.inventory):
            reward += KEY_REWARD
            self.inventory = self.inventory | {ITEM_KEY}

        # Terminal exit portal
        if self.pos == EXIT_POS:
            reward += EXIT_REWARD
            self.escaped = True
            terminal = True

        self.step_count += 1
        truncated = (not terminal) and (self.step_count >= self.max_steps)

        return self.state(), reward, terminal, truncated
