from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    algorithm: str = "q-learning"
    environment: str = "two-routes"
    seed: int = 0

    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.2
    planning_steps: int = 10
    exploration_bonus: float = 0.001

    change_step: int | None = None
    change_duration: int | None = None
    steps_after_change: int = 1_000
    max_episode_steps: int = 100

    @property
    def total_steps(self) -> int:
        if self.change_step is None:
            return self.steps_after_change

        return self.change_step + self.steps_after_change
