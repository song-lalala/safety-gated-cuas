import dataclasses

@dataclasses.dataclass
class Config:
    """Simulation configuration.

    Attributes
    ----------
    time_step: float
        Fixed time step for the simulation loop (seconds).
    random_seed: int | None
        Seed for the random number generator to ensure reproducibility.
    """
    time_step: float = 0.1
    random_seed: int | None = None
