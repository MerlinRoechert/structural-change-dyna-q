import unittest

from agents.local_change_dyna_q_agent import (
    LocalChangeDynaQAgent,
    LocalChangeModelEntry,
)
from agents.stability_aware_dyna_q import (
    StabilityAwareDynaQAgent,
    StabilityAwareModelEntry,
    TransitionCandidate,
)
from core.action import Action
from core.map_factory import MapFactory
from core.world_state_behavior import StaticWorldStateBehavior


def create_world():
    return MapFactory.create(
        world_id="two-routes",
        state_behavior=StaticWorldStateBehavior(),
        seed=0,
    )


class LocalChangeModelEntryTests(unittest.TestCase):
    def create_entry(self, confidence=1.0):
        return LocalChangeModelEntry(
            next_state=(1, 2),
            reward=-1.0,
            terminated=False,
            initial_confidence=confidence,
        )

    def observe(self, entry, next_state):
        entry.observe(
            next_state=next_state,
            reward=-1.0,
            terminated=False,
            confidence_rate=0.1,
            evidence_tolerance=0.25,
            repair_threshold=2.0,
        )

    def test_matching_transition_increases_confidence(self):
        entry = self.create_entry(confidence=0.5)

        self.observe(entry, (1, 2))

        self.assertAlmostEqual(entry.confidence, 0.55)
        self.assertFalse(entry.model_mismatch)

    def test_single_mismatch_does_not_replace_model(self):
        entry = self.create_entry()

        self.observe(entry, (2, 1))

        self.assertEqual(entry.next_state, (1, 2))
        self.assertEqual(entry.candidate_transition.next_state, (2, 1))
        self.assertAlmostEqual(entry.change_evidence, 0.75)
        self.assertFalse(entry.repair_triggered)

    def test_matching_transition_reduces_change_evidence(self):
        entry = self.create_entry()
        self.observe(entry, (2, 1))

        self.observe(entry, (1, 2))

        self.assertAlmostEqual(entry.change_evidence, 0.5)

    def test_different_mismatch_replaces_candidate_and_resets_evidence(self):
        entry = self.create_entry()
        self.observe(entry, (2, 1))

        self.observe(entry, (3, 1))

        self.assertEqual(entry.candidate_transition.next_state, (3, 1))
        self.assertAlmostEqual(entry.change_evidence, 0.65)

    def test_repeated_mismatch_repairs_model(self):
        entry = self.create_entry()

        for _ in range(4):
            self.observe(entry, (2, 1))

        self.assertEqual(entry.next_state, (2, 1))
        self.assertTrue(entry.repair_triggered)
        self.assertIsNone(entry.candidate_transition)
        self.assertEqual(entry.change_evidence, 0.0)

    def test_repair_sets_q_value_directly_to_new_target(self):
        agent = LocalChangeDynaQAgent(
            world=create_world(),
            learning_rate=0.1,
            discount_factor=0.9,
            planning_steps=0,
            initial_confidence=1.0,
        )
        state = (1, 1)
        action = Action.RIGHT
        old_next_state = (1, 2)
        new_next_state = (2, 1)
        agent.learn(state, action, old_next_state, -1.0, False)
        agent.init_state(new_next_state)
        agent.q_table[new_next_state][Action.DOWN] = 5.0

        for _ in range(4):
            agent.learn(state, action, new_next_state, 2.0, False)

        self.assertTrue(agent.repair_triggered)
        self.assertEqual(agent.model[(state, action)].next_state, new_next_state)
        self.assertAlmostEqual(agent.q_table[state][action], 6.5)


class StabilityAwareModelEntryTests(unittest.TestCase):
    def test_evidence_decay_zero_keeps_only_current_observation(self):
        entry = StabilityAwareModelEntry(evidence_decay=0.0)
        entry.observe((1, 2), -1.0, False)
        entry.observe((1, 2), -1.0, False)

        entry.observe((2, 1), -1.0, False)

        first_candidate, second_candidate = entry.candidates
        self.assertEqual(first_candidate.evidence, 0.0)
        self.assertEqual(second_candidate.evidence, 1.0)
        self.assertEqual(entry.observed_transition_stability, 1.0)

    def test_evidence_decay_one_keeps_all_previous_evidence(self):
        entry = StabilityAwareModelEntry(evidence_decay=1.0)
        entry.observe((1, 2), -1.0, False)
        entry.observe((1, 2), -1.0, False)

        entry.observe((2, 1), -1.0, False)

        first_candidate, second_candidate = entry.candidates
        self.assertEqual(first_candidate.evidence, 2.0)
        self.assertEqual(second_candidate.evidence, 1.0)

    def test_observation_decays_old_evidence(self):
        entry = StabilityAwareModelEntry(evidence_decay=0.9)
        entry.observe((1, 2), -1.0, False)
        entry.observe((1, 2), -1.0, False)

        entry.observe((2, 1), -1.0, False)

        first_candidate, second_candidate = entry.candidates
        self.assertAlmostEqual(first_candidate.evidence, 1.71)
        self.assertAlmostEqual(second_candidate.evidence, 1.0)

    def test_candidate_stabilities_sum_to_one(self):
        entry = StabilityAwareModelEntry(evidence_decay=0.9)
        entry.observe((1, 2), -1.0, False)
        entry.observe((2, 1), -1.0, False)

        total_stability = sum(
            entry.stability_of(candidate)
            for candidate in entry.candidates
        )

        self.assertAlmostEqual(total_stability, 1.0)

    def test_single_outlier_remains_weak_after_stable_history(self):
        entry = StabilityAwareModelEntry(evidence_decay=0.9)
        for _ in range(50):
            entry.observe((1, 2), -1.0, False)

        entry.observe((2, 1), -1.0, False)

        self.assertEqual(entry.dominant_candidate.next_state, (1, 2))
        self.assertLess(entry.observed_transition_stability, 0.11)
        self.assertTrue(entry.model_mismatch)

    def test_repeated_new_transition_becomes_dominant(self):
        entry = StabilityAwareModelEntry(evidence_decay=0.9)
        for _ in range(50):
            entry.observe((1, 2), -1.0, False)

        for _ in range(7):
            entry.observe((2, 1), -1.0, False)

        self.assertEqual(entry.dominant_candidate.next_state, (2, 1))
        self.assertGreater(entry.stability_score, 0.5)

    def test_planning_uses_stability_weighted_target(self):
        agent = StabilityAwareDynaQAgent(
            world=create_world(),
            learning_rate=1.0,
            planning_steps=0,
        )
        entry = StabilityAwareModelEntry(evidence_decay=0.9)
        first_candidate = TransitionCandidate((1, 2), 0.0, True)
        first_candidate.evidence = 3.0
        second_candidate = TransitionCandidate((2, 1), 8.0, True)
        second_candidate.evidence = 1.0
        entry.candidates = [first_candidate, second_candidate]

        agent._planning_update((1, 1), Action.RIGHT, entry)

        self.assertAlmostEqual(agent.q_table[(1, 1)][Action.RIGHT], 2.0)


if __name__ == "__main__":
    unittest.main()
