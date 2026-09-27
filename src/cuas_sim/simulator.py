from typing import Callable, Optional
from .config import Config
from .agents import Target, Interceptor
from .types import State2D


class Simulator:
    """Simple fixed-step 2D kinematic simulator.

    Parameters
    ----------
    config: Config
        Simulation configuration (time step, random seed).
    target: Target
        Target agent instance.
    interceptor: Interceptor
        Interceptor agent instance.
    step_hook: Optional[Callable[[int], None]]
        Optional hook called each simulation step with the current step index.
    """

    def __init__(self, config: Config, target: Target, interceptor: Interceptor, step_hook=None):
        self.config = config
        self.target = target
        self.interceptor = interceptor
        self.step_hook = step_hook
        self.step_count = 0
        self.time = 0.0
        self.history: list[tuple[float, State2D, State2D]] = []

    def step(self) -> None:
        """Advance simulation by one time step.
        Updates both agents and records a snapshot of their states.
        """
        # Call optional hook before updating agents
        if self.step_hook is not None:
            self.step_hook(self.step_count)
        dt = self.config.time_step
        # Update agents' kinematic state
        self.target.update()
        self.interceptor.update()
        self.time += dt
        # Record a deep copy of the states for later inspection
        self.history.append(
            (
                self.time,
                State2D(self.target.state.x, self.target.state.y, self.target.state.vx, self.target.state.vy),
                State2D(self.interceptor.state.x, self.interceptor.state.y, self.interceptor.state.vx, self.interceptor.state.vy),
            )
        )
        # Increment step counter
        self.step_count += 1

    def run(self, steps: int) -> None:
        """Run the simulation for the specified number of steps.
        """
        for _ in range(steps):
            self.step()
