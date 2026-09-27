import dataclasses
from .types import State2D
from .config import Config

@dataclasses.dataclass
class Agent:
    """Base class for simulation agents.

    Attributes
    ----------
    state: State2D
        Current kinematic state.
    config: Config
        Simulation configuration (shared).
    """
    state: State2D
    config: Config

    def update(self) -> None:
        """Update the agent's state using simple kinematics.

        Position is updated as:
            x += vx * dt
            y += vy * dt
        """
        dt = self.config.time_step
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt

class Target(Agent):
    """Target agent (could be moving obstacle)."""
    pass

class Interceptor(Agent):
    """Interceptor agent (e.g., UAV)."""
    pass
