from agents.dyna_q_agent import DynaQAgent, ModelTransition
from core.action import Action
from core.grid_world import GridWorld


class LocalChangeModelEntry(ModelTransition):
    def __init__(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
        initial_confidence: float,
    ):
        super().__init__(next_state, reward, terminated)
        self.confidence = initial_confidence
        self.candidate_transition: ModelTransition | None = None
        self.change_evidence = 0.0
        self.model_mismatch = False
        self.repair_triggered = False

    def observe(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
        confidence_rate: float,
        evidence_tolerance: float,
        repair_threshold: float,
    ) -> None:
        self.model_mismatch = not self._matches(
            self,
            next_state,
            reward,
            terminated,
        )
        self.repair_triggered = False

        confidence_before_observation = self.confidence
        surprise = 1.0 if self.model_mismatch else 0.0
        self.confidence += confidence_rate * (
            (1.0 - surprise) - self.confidence
        )

        if not self.model_mismatch:
            self.change_evidence = max(
                0.0,
                self.change_evidence - evidence_tolerance,
            )
            if self.change_evidence == 0.0:
                self.candidate_transition = None
            return

        if not self._candidate_matches(next_state, reward, terminated):
            self.candidate_transition = ModelTransition(
                next_state=next_state,
                reward=reward,
                terminated=terminated,
            )
            self.change_evidence = 0.0

        self.change_evidence = max(
            0.0,
            self.change_evidence
            + confidence_before_observation
            - evidence_tolerance,
        )

        if self.change_evidence > repair_threshold:
            self._repair_model()

    def _candidate_matches(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> bool:
        return (
            self.candidate_transition is not None
            and self._matches(
                self.candidate_transition,
                next_state,
                reward,
                terminated,
            )
        )

    def _repair_model(self) -> None:
        candidate = self.candidate_transition
        self.next_state = candidate.next_state
        self.reward = candidate.reward
        self.terminated = candidate.terminated
        self.candidate_transition = None
        self.change_evidence = 0.0
        self.repair_triggered = True

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


class LocalChangeDynaQAgent(DynaQAgent):
    name = "local-change-dyna-q"

    DEFAULT_CONFIDENCE_RATE = 0.1
    DEFAULT_EVIDENCE_TOLERANCE = 0.25
    DEFAULT_REPAIR_THRESHOLD = 2.0
    DEFAULT_INITIAL_CONFIDENCE = 0.5

    def __init__(
        self,
        world: GridWorld,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        planning_steps: int = DynaQAgent.DEFAULT_PLANNING_STEPS,
        confidence_rate: float = DEFAULT_CONFIDENCE_RATE,
        evidence_tolerance: float = DEFAULT_EVIDENCE_TOLERANCE,
        repair_threshold: float = DEFAULT_REPAIR_THRESHOLD,
        initial_confidence: float = DEFAULT_INITIAL_CONFIDENCE,
    ):
        if not 0.0 < confidence_rate <= 1.0:
            raise ValueError(
                "confidence_rate must be greater than 0 and at most 1"
            )
        if evidence_tolerance < 0.0:
            raise ValueError("evidence_tolerance must not be negative")
        if repair_threshold <= 0.0:
            raise ValueError("repair_threshold must be greater than 0")
        if not 0.0 <= initial_confidence <= 1.0:
            raise ValueError("initial_confidence must be between 0 and 1")

        super().__init__(
            world=world,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            planning_steps=planning_steps,
        )
        self.confidence_rate = confidence_rate
        self.evidence_tolerance = evidence_tolerance
        self.repair_threshold = repair_threshold
        self.initial_confidence = initial_confidence

        self.model_confidence: float | None = None
        self.change_evidence: float | None = None
        self.model_mismatch: bool | None = None
        self.repair_triggered: bool | None = None

    def _update_model(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        key = (current_state, action)
        model_entry = self.model.get(key)

        if model_entry is None:
            model_entry = LocalChangeModelEntry(
                next_state=next_state,
                reward=reward,
                terminated=terminated,
                initial_confidence=self.initial_confidence,
            )
            self.model[key] = model_entry
        else:
            model_entry.observe(
                next_state=next_state,
                reward=reward,
                terminated=terminated,
                confidence_rate=self.confidence_rate,
                evidence_tolerance=self.evidence_tolerance,
                repair_threshold=self.repair_threshold,
            )

            if model_entry.repair_triggered:
                target = self._calculate_q_target(
                    next_state=model_entry.next_state,
                    reward=model_entry.reward,
                    terminated=model_entry.terminated,
                )
                self._update_q_value_towards_target(
                    current_state=current_state,
                    action=action,
                    target=target,
                    learning_rate=1.0,
                )

        self.model_confidence = model_entry.confidence
        self.change_evidence = model_entry.change_evidence
        self.model_mismatch = model_entry.model_mismatch
        self.repair_triggered = model_entry.repair_triggered

    def reset_learning(self) -> None:
        super().reset_learning()
        self.model_confidence = None
        self.change_evidence = None
        self.model_mismatch = None
        self.repair_triggered = None
