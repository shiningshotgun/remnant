"""Frozen configuration parameters for Remnant: the time-loop experiment."""

# World dimensions and topology
GRID_HEIGHT = 5
GRID_WIDTH = 5
START_POS = (4, 0)
WALLS = frozenset({(0, 1), (1, 1), (2, 1), (1, 3), (1, 4)})

# Interactive objects and landmarks
ITEM_BAIT = 0
ITEM_KEY = 1

BAIT_POS = (0, 0)
KEY_POS = (4, 4)
GATE_POS = (0, 3)
EXIT_POS = (0, 4)

# Reward structure
STEP_COST = -0.01
BAIT_REWARD = 1.0       # Gross reward; net on collection = 0.99
KEY_REWARD = 0.1        # Gross reward; net on collection = 0.09
EXIT_REWARD = 10.0      # Gross reward; net on collection = 9.99

# Time-loop limits
LOOP_LIMIT_L = 18       # Maximum environment steps per loop
HORIZON_K_MAX = 100     # Maximum loop iterations per run

# Cardinal actions: 0=NORTH, 1=SOUTH, 2=WEST, 3=EAST
NORTH, SOUTH, WEST, EAST = 0, 1, 2, 3
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = {0: "NORTH", 1: "SOUTH", 2: "WEST", 3: "EAST"}

# Agent learning hyperparameters
DEFAULT_LR = 0.3
DEFAULT_GAMMA = 0.95
DEFAULT_EPS = 0.2
DEFAULT_MEMORY_BLEND = 0.5

# Episodic memory hyperparameters
DEFAULT_MEMORY_TEMPERATURE = 0.1
DEFAULT_MEMORY_LOOKUP_SAMPLE = 400

