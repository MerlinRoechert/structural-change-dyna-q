import random

from agents.dyna_q_agent import DynaQAgent, ModelTransition
from core.action import Action
from core.grid_world import GridWorld


class StabilityAwareTransition(ModelTransition):

    def __init__(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
        stability: float,
    ):
        super().__init__(next_state, reward, terminated)
        self.stability = stability


class StabilityAwareDynaQAgent(DynaQAgent):
    name = "stability-aware-dyna-q"

    DEFAULT_INITIAL_STABILITY = 0.5
    DEFAULT_STABILITY_INCREASE = 0.1
    DEFAULT_STABILITY_DECREASE = 0.5
    DEFAULT_STABILITY_THRESHOLD = 0.2

    def __init__(
        self,
        world: GridWorld,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        planning_steps: int = DynaQAgent.DEFAULT_PLANNING_STEPS,
        initial_stability: float = DEFAULT_INITIAL_STABILITY,
        stability_increase: float = DEFAULT_STABILITY_INCREASE,
        stability_decrease: float = DEFAULT_STABILITY_DECREASE,
        stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
    ):
        if not 0.0 <= initial_stability <= 1.0:
            raise ValueError("initial_stability must be between 0 and 1")
        if not 0.0 <= stability_threshold <= 1.0:
            raise ValueError("stability_threshold must be between 0 and 1")

        super().__init__(
            world=world,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            planning_steps=planning_steps,
        )
        self.initial_stability = initial_stability
        self.stability_increase = stability_increase
        self.stability_decrease = stability_decrease
        self.stability_threshold = stability_threshold

    def _update_model(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        key = (current_state, action)
        previous = self.model.get(key)

        if previous is None:
            stability = self.initial_stability
        elif self._matches(previous, next_state, reward, terminated):
            stability = min(1.0, previous.stability + self.stability_increase)
        else:
            stability = max(0.0, previous.stability - self.stability_decrease)

        self.model[key] = StabilityAwareTransition(
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            stability=stability,
        )

    @staticmethod
    def _matches(
        previous: StabilityAwareTransition,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> bool:
        return (
            previous.next_state == next_state
            and previous.reward == reward
            and previous.terminated == terminated
        )

    def _run_planning(self) -> None:
        model_entries = tuple(self.model.items())
        if not model_entries:
            return

        weights = [
            max(transition.stability, 1e-6) for _, transition in model_entries
        ]

        for _ in range(self.planning_steps):
            (state, action), transition = random.choices(
                model_entries, weights=weights, k=1
            )[0]

            if transition.stability < self.stability_threshold:
                continue

            self._planning_update(state, action, transition)

    def _planning_update(
        self,
        state: tuple[int, int],
        action: Action,
        transition: StabilityAwareTransition,
    ) -> None:
        self.init_state(state)
        self.init_state(transition.next_state)

        old_q = self.q_table[state][action]
        best_next_q = (
            0.0
            if transition.terminated
            else max(self.q_table[transition.next_state].values())
        )

        effective_lr = self.learning_rate * transition.stability
        self.q_table[state][action] = old_q + effective_lr * (
            self._planning_reward(transition)
            + self.discount_factor * best_next_q
            - old_q
        )