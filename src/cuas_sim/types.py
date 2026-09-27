import dataclasses

@dataclasses.dataclass
class State2D:
    """Represents a 2D kinematic state (position and velocity)."""
    x: float
    y: float
    vx: float
    vy: float
