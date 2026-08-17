import math
import unittest

from agents.dyna_q_agent import DynaQAgent, ModelTransition
from agents.dyna_q_plus_agent import DynaQPlusAgent, DynaQPlusTransition
from agents.q_learning_agent import QLearningAgent
from core.action import Action
from core.map_factory import MapFactory
from core.world_state_behavior import StaticWorldStateBehavior


def create_world():
    return MapFactory.create(
        world_id="two-routes",
        state_behavior=StaticWorldStateBehavior(),
        seed=0,
    )


class QLearningAgentTests(unittest.TestCase):
    def test_updates_q_value_towards_bellman_target(self):
        agent = QLearningAgent(
            world=create_world(),
            learning_rate=0.5,
            discount_factor=0.9,
        )
        current_state = (1, 1)
        next_state = (1, 2)
        agent.init_state(next_state)
        agent.q_table[next_state][Action.RIGHT] = 4.0

        agent.learn(
            current_state=current_state,
            action=Action.RIGHT,
            next_state=next_state,
            reward=2.0,
            terminated=False,
        )

        self.assertAlmostEqual(
            agent.q_table[current_state][Action.RIGHT],
            2.8,
        )

    def test_terminal_transition_ignores_future_q_values(self):
        agent = QLearningAgent(
            world=create_world(),
            learning_rate=0.5,
            discount_factor=0.9,
        )
        next_state = (1, 2)
        agent.init_state(next_state)
        agent.q_table[next_state][Action.RIGHT] = 100.0

        agent.learn(
            current_state=(1, 1),
            action=Action.RIGHT,
            next_state=next_state,
            reward=10.0,
            terminated=True,
        )

        self.assertAlmostEqual(
            agent.q_table[(1, 1)][Action.RIGHT],
            5.0,
        )


class DynaQAgentTests(unittest.TestCase):
    def test_real_observation_overwrites_model_transition(self):
        agent = DynaQAgent(world=create_world(), planning_steps=0)
        state = (1, 1)
        action = Action.RIGHT

        agent.learn(state, action, (1, 2), -1.0, False)
        agent.learn(state, action, (2, 1), -10.0, False)

        transition = agent.model[(state, action)]
        self.assertEqual(transition.next_state, (2, 1))
        self.assertEqual(transition.reward, -10.0)

    def test_without_planning_direct_update_matches_q_learning(self):
        q_learning = QLearningAgent(world=create_world(), learning_rate=0.2)
        dyna_q = DynaQAgent(
            world=create_world(),
            learning_rate=0.2,
            planning_steps=0,
        )
        transition = ((1, 1), Action.RIGHT, (1, 2), -1.0, False)

        q_learning.learn(*transition)
        dyna_q.learn(*transition)

        self.assertEqual(dyna_q.q_table, q_learning.q_table)

    def test_planning_update_uses_model_transition(self):
        agent = DynaQAgent(
            world=create_world(),
            learning_rate=1.0,
            discount_factor=0.5,
            planning_steps=0,
        )
        next_state = (1, 2)
        agent.init_state(next_state)
        agent.q_table[next_state][Action.RIGHT] = 4.0

        agent._planning_update(
            state=(1, 1),
            action=Action.RIGHT,
            transition=ModelTransition(next_state, 1.0, False),
        )

        self.assertAlmostEqual(
            agent.q_table[(1, 1)][Action.RIGHT],
            3.0,
        )


class DynaQPlusAgentTests(unittest.TestCase):
    def test_first_state_visit_initializes_all_actions(self):
        agent = DynaQPlusAgent(world=create_world(), planning_steps=0)
        state = (1, 1)

        agent._update_model(
            current_state=state,
            action=Action.RIGHT,
            next_state=(1, 2),
            reward=-1.0,
            terminated=False,
        )

        entries = {
            action: agent.model[(state, action)]
            for action in agent.actions
        }
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[Action.UP].next_state, state)
        self.assertEqual(entries[Action.UP].reward, 0.0)
        self.assertEqual(entries[Action.RIGHT].next_state, (1, 2))

    def test_real_observation_records_current_time_step(self):
        agent = DynaQPlusAgent(world=create_world(), planning_steps=0)
        agent.time_step = 7

        agent._update_model(
            current_state=(1, 1),
            action=Action.RIGHT,
            next_state=(1, 2),
            reward=-1.0,
            terminated=False,
        )

        transition = agent.model[((1, 1), Action.RIGHT)]
        self.assertEqual(transition.last_tried_step, 7)

    def test_planning_reward_contains_time_bonus(self):
        agent = DynaQPlusAgent(
            world=create_world(),
            planning_steps=0,
            exploration_bonus=0.01,
        )
        agent.time_step = 13
        transition = DynaQPlusTransition(
            next_state=(1, 2),
            reward=-1.0,
            terminated=False,
            last_tried_step=4,
        )

        reward = agent._planning_reward(transition)

        self.assertAlmostEqual(reward, -1.0 + 0.01 * math.sqrt(9))

    def test_direct_update_does_not_include_exploration_bonus(self):
        agent = DynaQPlusAgent(
            world=create_world(),
            learning_rate=1.0,
            planning_steps=0,
            exploration_bonus=1.0,
        )

        agent.learn(
            current_state=(1, 1),
            action=Action.RIGHT,
            next_state=(1, 2),
            reward=-1.0,
            terminated=True,
        )

        self.assertEqual(agent.q_table[(1, 1)][Action.RIGHT], -1.0)


if __name__ == "__main__":
    unittest.main()
