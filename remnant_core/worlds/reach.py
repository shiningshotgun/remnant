"""The Reach of Vesper world environment.

Implements the 72-node waypoint graph on the island of Vesper Reach.
Adheres strictly to the Remnant world contract:
- reset() -> observation
- state() -> observation
- step(action) -> (observation, reward, terminal, truncated)
- max_steps: int
"""

import copy
from typing import Dict, List, Tuple, FrozenSet, Optional, Any
from remnant_core.characters import CharacterManager

# Item IDs
ITEM_PEARLS = 0     # Internal to world only; never visible in observation
ITEM_LAMP_KEY = 1   # Visible in observation when held

# Rewards and limits
STRIDE_COST = -0.01
PEARLS_REWARD = 1.0     # Gross reward at Drowned Chapel
LAMP_KEY_REWARD = 0.1   # Gross reward at Net Loft
TIDEGATE_REWARD = 10.0  # Gross reward at Tidegate (terminal)
REACH_LOOP_LIMIT = 54   # Maximum strides before Drowning Toll

# Named junction coordinates
JUNCTIONS = {
    "Tidegate": (0, 14),
    "Harbor Lock": (6, 14),
    "Drowned Chapel": (10, 4),
    "Wrecksmoor Landing": (20, 0),
    "Needle Cut": (20, 10),
    "Net Loft": (24, 14),
}

START_POS = JUNCTIONS["Wrecksmoor Landing"]
CHAPEL_POS = JUNCTIONS["Drowned Chapel"]
NET_LOFT_POS = JUNCTIONS["Net Loft"]
HARBOR_LOCK_POS = JUNCTIONS["Harbor Lock"]
TIDEGATE_POS = JUNCTIONS["Tidegate"]

# Route specifications: (edge_id, endpoint_A, endpoint_B, strides, path_order)
EDGES = [
    (0, "Wrecksmoor Landing", "Needle Cut", 10, "col-first"),
    (1, "Wrecksmoor Landing", "Drowned Chapel", 14, "row-first"),
    (2, "Drowned Chapel", "Needle Cut", 16, "col-first"),
    (3, "Needle Cut", "Net Loft", 8, "row-first"),
    (4, "Net Loft", "Harbor Lock", 18, "row-first"),
    (5, "Harbor Lock", "Tidegate", 6, "row-first"),
]


def generate_chain(coord_a: Tuple[int, int], coord_b: Tuple[int, int], strides: int, path_order: str) -> List[Tuple[int, int]]:
    """Generate the exact deterministic sequence of lattice nodes along an edge."""
    r_a, c_a = coord_a
    r_b, c_b = coord_b
    nodes = [coord_a]
    curr_r, curr_c = r_a, c_a
    if path_order == "row-first":
        step_r = 1 if r_b > r_a else (-1 if r_b < r_a else 0)
        while curr_r != r_b:
            curr_r += step_r
            nodes.append((curr_r, curr_c))
        step_c = 1 if c_b > c_a else (-1 if c_b < c_a else 0)
        while curr_c != c_b:
            curr_c += step_c
            nodes.append((curr_r, curr_c))
    elif path_order == "col-first":
        step_c = 1 if c_b > c_a else (-1 if c_b < c_a else 0)
        while curr_c != c_b:
            curr_c += step_c
            nodes.append((curr_r, curr_c))
        step_r = 1 if r_b > r_a else (-1 if r_b < r_a else 0)
        while curr_r != r_b:
            curr_r += step_r
            nodes.append((curr_r, curr_c))
    else:
        raise ValueError(f"Unknown path order: {path_order}")
    if len(nodes) != strides + 1 or nodes[-1] != coord_b:
        raise RuntimeError(f"Chain generation mismatch for {coord_a} -> {coord_b}")
    return nodes


def build_reach_graph():
    """Build the node set, edge chains, and arc mappings for ReachWorld."""
    edge_chains: Dict[int, List[Tuple[int, int]]] = {}
    node_arcs: Dict[Tuple[int, int], List[Tuple[int, str, Tuple[int, int]]]] = {}

    for eid, a_name, b_name, strides, path_order in EDGES:
        chain = generate_chain(JUNCTIONS[a_name], JUNCTIONS[b_name], strides, path_order)
        edge_chains[eid] = chain
        for i in range(len(chain)):
            u = chain[i]
            if u not in node_arcs:
                node_arcs[u] = []
            if i > 0:
                node_arcs[u].append((eid, "toward_A", chain[i - 1]))
            if i < len(chain) - 1:
                node_arcs[u].append((eid, "toward_B", chain[i + 1]))

    # Sort arcs at each node: by edge_id ascending, then toward_A before toward_B
    for u in node_arcs:
        node_arcs[u].sort(key=lambda x: (x[0], 0 if x[1] == "toward_A" else 1))

    # The locked boundary: Harbor Lock to the first interior node of edge 5 (chain[1])
    lock_boundary = (HARBOR_LOCK_POS, edge_chains[5][1])

    return edge_chains, node_arcs, lock_boundary


GRAPH_CHAINS, GRAPH_ARCS, LOCK_BOUNDARY = build_reach_graph()
ALL_NODES = frozenset(GRAPH_ARCS.keys())

PhysicalState = Tuple[Tuple[int, int], bool, bool]  # (pos, has_key, pearls_collected)


def physical_transition(
    state: PhysicalState,
    action: int,
    has_key_allowed: bool = True,
) -> Tuple[PhysicalState, float, bool, Optional[str], bool, bool]:
    """Pure physical transition function shared between live world and solver.
    
    Args:
        state: (pos, has_key, pearls_collected)
        action: action index (0..3)
        has_key_allowed: whether key pickup at Net Loft is allowed (False for no-key search)
        
    Returns:
        next_state: (next_pos, next_has_key, next_pearls)
        reward: scalar transition reward
        terminal: whether Tidegate was reached
        invalid_reason: reason string if stayed in place, else None
        pearls_picked: True if pearls were collected on this transition
        key_picked: True if key was collected on this transition
    """
    pos, has_key, pearls = state
    reward = STRIDE_COST
    arcs = GRAPH_ARCS.get(pos, [])
    target_node = pos
    invalid_reason = None

    if 0 <= action < len(arcs):
        eid, direction, neighbor = arcs[action]
        if (pos, neighbor) == LOCK_BOUNDARY and not has_key:
            target_node = pos
            invalid_reason = "locked_gate_no_key"
        else:
            target_node = neighbor
    else:
        target_node = pos
        invalid_reason = "no_incident_arc"

    next_pos = target_node
    next_has_key = has_key
    next_pearls = pearls
    pearls_picked = False
    key_picked = False

    if next_pos == CHAPEL_POS and not pearls:
        next_pearls = True
        reward += PEARLS_REWARD
        pearls_picked = True

    if next_pos == NET_LOFT_POS and not has_key and has_key_allowed:
        next_has_key = True
        reward += LAMP_KEY_REWARD
        key_picked = True

    terminal = False
    if next_pos == TIDEGATE_POS:
        reward += TIDEGATE_REWARD
        terminal = True

    return (next_pos, next_has_key, next_pearls), reward, terminal, invalid_reason, pearls_picked, key_picked


class ReachWorld:
    """The Reach of Vesper: an agent in a resetting tidal world."""

    def __init__(self, max_steps: int = REACH_LOOP_LIMIT):
        self.max_steps = max_steps
        self.pos: Tuple[int, int] = START_POS
        self.has_key: bool = False
        self.pearls_collected: bool = False
        self.stride_count: int = 0
        self.escaped: bool = False
        self.characters = CharacterManager()
        self._last_telemetry: Optional[Dict[str, Any]] = None

    def reset(self) -> Tuple[Tuple[int, int], FrozenSet[int]]:
        """Physical loop reset to dawn state."""
        self.pos = START_POS
        self.has_key = False
        self.pearls_collected = False
        self.stride_count = 0
        self.escaped = False
        self.characters.reset()
        self._last_telemetry = None
        return self.state()

    def state(self) -> Tuple[Tuple[int, int], FrozenSet[int]]:
        """Observation of the world: ((row, col), frozenset({ITEM_LAMP_KEY})) if key held."""
        items = frozenset({ITEM_LAMP_KEY}) if self.has_key else frozenset()
        return (self.pos, items)

    def step(self, action: int) -> Tuple[Tuple[Tuple[int, int], FrozenSet[int]], float, bool, bool]:
        """Execute one action step (0 <= action <= 3)."""
        if self.escaped:
            raise RuntimeError("Cannot step in an escaped world; must call reset()")
        if self.stride_count >= self.max_steps:
            raise RuntimeError("Cannot step in a truncated world; must call reset()")

        if not isinstance(action, int) or action < 0 or action > 3:
            raise ValueError(f"Action must be an integer in 0..3, got {action}")

        self.stride_count += 1

        curr_physical_state = (self.pos, self.has_key, self.pearls_collected)
        next_physical_state, reward, terminal, invalid_reason, pearls_picked, key_picked = physical_transition(
            curr_physical_state, action, has_key_allowed=True
        )

        self.pos, self.has_key, self.pearls_collected = next_physical_state
        if terminal:
            self.escaped = True

        truncated = False
        # Truncation check (terminal takes precedence at stride 54)
        if not terminal and self.stride_count >= self.max_steps:
            truncated = True

        # Update autonomous characters once after physical transition
        self.characters.step(self.pos, self.has_key)

        # Record immutable post-step telemetry snapshot
        self._last_telemetry = {
            "stride": self.stride_count,
            "pos": self.pos,
            "has_key": self.has_key,
            "pearls_collected": self.pearls_collected,
            "pearls_picked": pearls_picked,
            "key_picked": key_picked,
            "reward": round(reward, 4),
            "terminal": terminal,
            "truncated": truncated,
            "invalid_reason": invalid_reason,
            "npc_states": self.characters.npc_states(),
        }

        return self.state(), reward, terminal, truncated

    @property
    def last_telemetry(self) -> Optional[Dict[str, Any]]:
        """Return a defensive deep copy of last telemetry snapshot."""
        if self._last_telemetry is None:
            return None
        return copy.deepcopy(self._last_telemetry)

    def npc_states(self) -> Dict[str, Dict[str, Any]]:
        """Return the current snapshot of NPC states."""
        return self.characters.npc_states()
