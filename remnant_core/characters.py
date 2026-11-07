"""Autonomous character policies for the Reach of Vesper.

Characters are responsive goal-directed agents evaluated once after each Wick
physical transition (including invalid stays).
Each character has explicit percepts, ordered decision rules, selected activity,
and local state that resets each tide.
No modulo-clock cycles, fixed schedules, RNG calls, language models, or learner controls.
"""

from typing import Dict, Any, Optional, Tuple

State = Tuple[Tuple[int, int], Any]


class Gleaner:
    """The Gleaner, tender of the Drowned Chapel.
    
    Seeks company and cares for the chapel.
    - When Wick arrives: warm greeting
    - While Wick remains: invitation to stay
    - When Wick absent: chapel tending
    
    Tracks prior local presence to distinguish arrival from continued company.
    Never blocks, moves, withholds, consumes, or adds pearls.
    """
    def __init__(self, home_pos: Tuple[int, int] = (10, 4)):
        self.home_pos = home_pos
        self.wick_present_prior: bool = False
        self.current_activity: str = "tending the chapel"

    def reset(self) -> None:
        self.wick_present_prior = False
        self.current_activity = "tending the chapel"

    def step(self, wick_pos: Tuple[int, int]) -> str:
        wick_here = (wick_pos == self.home_pos)
        if wick_here:
            if not self.wick_present_prior:
                self.current_activity = "greeting Wick warmly"
            else:
                self.current_activity = "inviting Wick to stay"
            self.wick_present_prior = True
        else:
            self.current_activity = "tending the chapel"
            self.wick_present_prior = False
        return self.current_activity

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": "The Gleaner",
            "home": self.home_pos,
            "goal": "seeks company and cares for the chapel",
            "activity": self.current_activity,
            "local_state": {"wick_present_prior": self.wick_present_prior},
        }


class Brannoch:
    """Brannoch, the net-mender at the Net Loft.
    
    Seeks progress on his unfinished net and familiar storytelling.
    - When Wick arrives (present): choose his recurring story
    - When Wick absent: choose mending and increment local work counter
    
    Net never completes within 54 strides.
    Never reads key possession or reacts to its disappearance.
    Changing the key flag alone must not change his decision.
    """
    def __init__(self, home_pos: Tuple[int, int] = (24, 14)):
        self.home_pos = home_pos
        self.work_counter: int = 0
        self.current_activity: str = "mending the unfinished net"

    def reset(self) -> None:
        self.work_counter = 0
        self.current_activity = "mending the unfinished net"

    def step(self, wick_pos: Tuple[int, int]) -> str:
        wick_here = (wick_pos == self.home_pos)
        if wick_here:
            self.current_activity = "telling a recurring story"
        else:
            self.work_counter += 1
            self.current_activity = "mending the unfinished net"
        return self.current_activity

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": "Brannoch",
            "home": self.home_pos,
            "goal": "seeks net progress and storytelling",
            "activity": self.current_activity,
            "local_state": {"work_counter": self.work_counter},
        }


class Dov:
    """Dov Harrowgate, the lockkeeper at Harbor Lock.
    
    Seeks correct procedure and care of the mechanism.
    - Wick at the lock with key: acknowledge key/procedure
    - Wick at the lock without key: require key
    - Wick absent: inspect mechanism
    
    No ability to open without the key or withhold access with it;
    access remains the physical key rule, not another NPC gate.
    """
    def __init__(self, home_pos: Tuple[int, int] = (6, 14)):
        self.home_pos = home_pos
        self.current_activity: str = "inspecting the mechanism"

    def reset(self) -> None:
        self.current_activity = "inspecting the mechanism"

    def step(self, wick_pos: Tuple[int, int], wick_has_key: bool) -> str:
        wick_here = (wick_pos == self.home_pos)
        if wick_here:
            if wick_has_key:
                self.current_activity = "acknowledging the Lamp-Key and procedure"
            else:
                self.current_activity = "requiring the Lamp-Key"
        else:
            self.current_activity = "inspecting the mechanism"
        return self.current_activity

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": "Dov Harrowgate",
            "home": self.home_pos,
            "goal": "seeks correct procedure and mechanism care",
            "activity": self.current_activity,
            "local_state": {},
        }


class CharacterManager:
    """Coordinates the autonomous characters on the Reach of Vesper."""
    def __init__(self):
        self.gleaner = Gleaner(home_pos=(10, 4))
        self.brannoch = Brannoch(home_pos=(24, 14))
        self.dov = Dov(home_pos=(6, 14))

    def reset(self) -> None:
        self.gleaner.reset()
        self.brannoch.reset()
        self.dov.reset()

    def step(self, wick_pos: Tuple[int, int], wick_has_key: bool) -> None:
        self.gleaner.step(wick_pos)
        self.brannoch.step(wick_pos)
        self.dov.step(wick_pos, wick_has_key)

    def npc_states(self) -> Dict[str, Dict[str, Any]]:
        """Immutable snapshot of all character states for world telemetry."""
        return {
            "Gleaner": self.gleaner.state_snapshot(),
            "Brannoch": self.brannoch.state_snapshot(),
            "Dov": self.dov.state_snapshot(),
            "Ysolde": {
                "name": "Ysolde Marr (Portrait)",
                "home": (20, 0),
                "goal": "static memorial",
                "activity": "watching from the painted portrait",
                "local_state": {"static": True},
            },
        }
