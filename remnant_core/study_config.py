"""Configuration and protocol constants for the Reach of Vesper study."""

from remnant_core.worlds.reach import (
    START_POS,
    JUNCTIONS,
    REACH_LOOP_LIMIT,
    STRIDE_COST,
    PEARLS_REWARD,
    LAMP_KEY_REWARD,
    TIDEGATE_REWARD,
)

# Study protocol parameters
STUDY_SEEDS = list(range(101, 131))  # 30 seeds: 101..130
STUDY_HORIZON_K_MAX = 100            # Maximum tides (loops) per seed
SCHEMA_VERSION = "2.0.0"
SCHEMA_VERSION_BASELINE = "1.0.0"
SCHEMA_VERSION_ROUTE_B = "2.0.0"
SUPPORTED_SCHEMAS = {"1.0.0", "2.0.0"}

# Conditions
CONDITION_BASELINE = "baseline_hidden_pearl_v1"
CONDITION_ROUTE_B = "tally_current_tide_pearl_v1"
CONDITION_EXPLORATION = "cross_tide_exploration_memory_v1"
SUPPORTED_CONDITIONS = {CONDITION_BASELINE, CONDITION_ROUTE_B, CONDITION_EXPLORATION}

# Study hyperparameters
STUDY_LR = 0.3
STUDY_GAMMA = 0.95
STUDY_EPS = 0.2
STUDY_MEMORY_BLEND = 0.5
STUDY_MEMORY_TEMPERATURE = 0.1
STUDY_MEMORY_LOOKUP_SAMPLE = 400
