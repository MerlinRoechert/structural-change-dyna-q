from agents.dyna_q_agent import DynaQAgent, ModelTransition
from core.action import Action
from core.grid_world import GridWorld


class TransitionCandidate(ModelTransition):

    def __init__(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ):
        super().__init__(next_state, reward, terminated)
        self.evidence = 0.0

    def matches(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> bool:
        return (
            self.next_state == next_state
            and self.reward == reward
            and self.terminated == terminated
        )


class StabilityAwareModelEntry:

    def __init__(self, evidence_decay: float):
        self.evidence_decay = evidence_decay
        self.candidates: list[TransitionCandidate] = []
        self.observed_candidate: TransitionCandidate | None = None
        self.model_mismatch = False

    @property
    def total_evidence(self) -> float:
        return sum(candidate.evidence for candidate in self.candidates)

    @property
    def dominant_candidate(self) -> TransitionCandidate:
        return max(
            self.candidates,
            key=lambda candidate: candidate.evidence,
        )

    @property
    def stability_score(self) -> float:
        return self.stability_of(self.dominant_candidate)

    @property
    def observed_transition_stability(self) -> float:
        return self.stability_of(self.observed_candidate)

    def stability_of(self, candidate: TransitionCandidate) -> float:
        return candidate.evidence / self.total_evidence

    def observe(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        self.model_mismatch = (
            bool(self.candidates)
            and not self.dominant_candidate.matches(
                next_state,
                reward,
                terminated,
            )
        )

        for candidate in self.candidates:
            candidate.evidence *= self.evidence_decay

        observed_candidate = next(
            (
                candidate
                for candidate in self.candidates
                if candidate.matches(next_state, reward, terminated)
            ),
            None,
        )
        if observed_candidate is None:
            observed_candidate = TransitionCandidate(
                next_state=next_state,
                reward=reward,
                terminated=terminated,
            )
            self.candidates.append(observed_candidate)

        observed_candidate.evidence += 1.0
        self.observed_candidate = observed_candidate


class StabilityAwareDynaQAgent(DynaQAgent):
    name = "stability-aware-dyna-q"

    DEFAULT_EVIDENCE_DECAY = 0.9

    def __init__(
        self,
        world: GridWorld,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        planning_steps: int = DynaQAgent.DEFAULT_PLANNING_STEPS,
        evidence_decay: float = DEFAULT_EVIDENCE_DECAY,
    ):
        if not 0.0 < evidence_decay < 1.0:
            raise ValueError("evidence_decay must be between 0 and 1")

        super().__init__(
            world=world,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            planning_steps=planning_steps,
        )
        self.evidence_decay = evidence_decay
        self.model: dict[
            tuple[tuple[int, int], Action],
            StabilityAwareModelEntry,
        ] = {}

        self.stability_score: float | None = None
        self.observed_transition_stability: float | None = None
        self.candidate_count: int | None = None
        self.model_mismatch: bool | None = None
        self._experience_learning_weight = 1.0

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

        self._update_q_value(
            current_state=current_state,
            action=action,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            learning_rate=(
                self.learning_rate * self._experience_learning_weight
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
        self._experience_learning_weight = 1.0
        key = (current_state, action)
        model_entry = self.model.get(key)
        if model_entry is None:
            model_entry = StabilityAwareModelEntry(self.evidence_decay)
            self.model[key] = model_entry

        model_entry.observe(next_state, reward, terminated)

        self.model_mismatch = model_entry.model_mismatch
        self.stability_score = model_entry.stability_score
        self.observed_transition_stability = (
            model_entry.observed_transition_stability
        )
        self.candidate_count = len(model_entry.candidates)
        self._experience_learning_weight = (
            model_entry.observed_transition_stability
        )

    def _planning_update(
        self,
        state: tuple[int, int],
        action: Action,
        model_entry: StabilityAwareModelEntry,
    ) -> None:
        planning_target = sum(
            model_entry.stability_of(candidate)
            * self._calculate_q_target(
                next_state=candidate.next_state,
                reward=candidate.reward,
                terminated=candidate.terminated,
            )
            for candidate in model_entry.candidates
        )

        self._update_q_value_towards_target(
            current_state=state,
            action=action,
            target=planning_target,
            learning_rate=self.learning_rate,
        )

    def reset_learning(self) -> None:
        super().reset_learning()
        self.stability_score = None
        self.observed_transition_stability = None
        self.candidate_count = None
        self.model_mismatch = None
        self._experience_learning_weight = 1.0
