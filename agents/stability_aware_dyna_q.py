import math
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
        self.candidate_transition: ModelTransition | None = None
        self.change_evidence = 0.0


class StabilityAwareDynaQAgent(DynaQAgent):
    name = "stability-aware-dyna-q"

    DEFAULT_INITIAL_STABILITY = 0.5
    DEFAULT_STABILITY_INCREASE = 0.1
    DEFAULT_EVIDENCE_GAIN = 0.01
    DEFAULT_CHANGE_EVIDENCE_DECAY = 0.5

    def __init__(
        self,
        world: GridWorld,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        planning_steps: int = DynaQAgent.DEFAULT_PLANNING_STEPS,
        initial_stability: float = DEFAULT_INITIAL_STABILITY,
        stability_increase: float = DEFAULT_STABILITY_INCREASE,
        evidence_gain: float = DEFAULT_EVIDENCE_GAIN,
        change_evidence_decay: float = DEFAULT_CHANGE_EVIDENCE_DECAY,
        seed: int = 0,
    ):
        if not 0.0 <= initial_stability <= 1.0:
            raise ValueError("initial_stability must be between 0 and 1")
        if evidence_gain <= 0.0:
            raise ValueError("evidence_gain must be greater than 0")

        super().__init__(
            world=world,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            planning_steps=planning_steps,
        )
        self.initial_stability = initial_stability
        self.stability_increase = stability_increase
        self.evidence_gain = evidence_gain
        self.change_evidence_decay = change_evidence_decay
        self.random = random.Random(seed)

        self.stability_score: float | None = None
        self.change_evidence: float | None = None
        self.change_probability = 1.0
        self.change_detected = False

    def learn(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        self._update_model(
            current_state=current_state,
            action=action,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
        )

        if self.change_detected:
            self._repair_q_value(
                current_state=current_state,
                action=action,
                transition=self.model[(current_state, action)],
            )
        else:
            self._update_q_value(
                current_state=current_state,
                action=action,
                next_state=next_state,
                reward=reward,
                terminated=terminated,
                learning_rate=(
                    self.learning_rate * self.change_probability
                ),
            )

        self._run_planning()

    def _update_model(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        self.change_detected = False
        self.change_probability = 1.0
        key = (current_state, action)
        transition = self.model.get(key)

        if transition is None:
            transition = StabilityAwareTransition(
                next_state=next_state,
                reward=reward,
                terminated=terminated,
                stability=self.initial_stability,
            )
            self.model[key] = transition
        elif self._matches(transition, next_state, reward, terminated):
            self._confirm_transition(transition)
        else:
            self._remember_alternative(
                transition=transition,
                next_state=next_state,
                reward=reward,
                terminated=terminated,
            )

            self.change_probability = self._repair_probability(transition)
            if self.random.random() < self.change_probability:
                self.stability_score = transition.stability
                self.change_evidence = transition.change_evidence
                self.model[key] = self._adopt_candidate(transition)
                self.change_detected = True
                return

        self.stability_score = transition.stability
        self.change_evidence = transition.change_evidence

    def _confirm_transition(
        self,
        transition: StabilityAwareTransition,
    ) -> None:
        transition.stability = min(
            1.0,
            transition.stability + self.stability_increase,
        )
        transition.change_evidence = max(
            0.0,
            transition.change_evidence - self.change_evidence_decay,
        )

        if transition.change_evidence == 0.0:
            transition.candidate_transition = None

    def _remember_alternative(
        self,
        transition: StabilityAwareTransition,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        candidate = transition.candidate_transition

        if candidate is not None and self._matches(
            candidate,
            next_state,
            reward,
            terminated,
        ):
            transition.change_evidence += (
                self.evidence_gain * transition.stability
            )
        else:
            transition.candidate_transition = ModelTransition(
                next_state=next_state,
                reward=reward,
                terminated=terminated,
            )
            transition.change_evidence = (
                self.evidence_gain * transition.stability
            )

    @staticmethod
    def _repair_probability(
        transition: StabilityAwareTransition,
    ) -> float:
        return 1.0 - math.exp(-transition.change_evidence)

    def _adopt_candidate(
        self,
        transition: StabilityAwareTransition,
    ) -> StabilityAwareTransition:
        candidate = transition.candidate_transition

        return StabilityAwareTransition(
            next_state=candidate.next_state,
            reward=candidate.reward,
            terminated=candidate.terminated,
            stability=self.initial_stability,
        )

    def _repair_q_value(
        self,
        current_state: tuple[int, int],
        action: Action,
        transition: ModelTransition,
    ) -> None:
        self.init_state(current_state)
        self.init_state(transition.next_state)

        best_next_q = 0.0 if transition.terminated else max(
            self.q_table[transition.next_state].values()
        )
        self.q_table[current_state][action] = (
            transition.reward + self.discount_factor * best_next_q
        )

    @staticmethod
    def _matches(
        transition: ModelTransition,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> bool:
        return (
            transition.next_state == next_state
            and transition.reward == reward
            and transition.terminated == terminated
        )

    def reset_learning(self) -> None:
        super().reset_learning()
        self.stability_score = None
        self.change_evidence = None
        self.change_probability = 1.0
        self.change_detected = False
