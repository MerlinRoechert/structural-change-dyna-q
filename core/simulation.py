from core.agent import Agent
from core.grid_world import GridWorld
from core.world_object import Goal, Slippery, Trap


class SimulationMode:
    TRAIN = "train"
    EVAL = "eval"


class Simulation:
    def __init__(self, world: GridWorld, agent: Agent):
        self.world = world
        self.agent = agent
        self.mode = SimulationMode.TRAIN

        self.episode_summaries: list[dict] = []

        self.reset_episode()

    def reset_episode(self):
        self.state = self.world.reset()
        self.done = False
        self.total_reward = 0
        self.steps = 0

        self.slippery_visits = 0
        self.trap_hits = 0
        self.used_slippery = False
        self.reached_goal = False

    def finish_episode(self):
        self.episode_summaries.append(
            {
                "episode": len(self.episode_summaries) + 1,
                "mode": self.mode,
                "reward": self.total_reward,
                "steps": self.steps,
                "reached_goal": self.reached_goal,
                "trap_hits": self.trap_hits,
                "slippery_visits": self.slippery_visits,
                "used_slippery": self.used_slippery,
            }
        )

    def step(self):
        if self.done:
            return

        if self.mode == SimulationMode.EVAL:
            action = self.agent.choose_action(self.state, epsilon_override=0.0)
        else:
            action = self.agent.choose_action(self.state)

        next_state, reward, done = self.world.step(action)

        if self.mode == SimulationMode.TRAIN:
            self.agent.update_q_value(
                self.state,
                action,
                next_state,
                reward,
                done,
            )

        self.state = next_state
        self.done = done
        self.total_reward += reward
        self.steps += 1

        self.track_current_state()

    def track_current_state(self):
        cell = self.world.get_cell(self.state)

        if isinstance(cell, Slippery):
            self.slippery_visits += 1
            self.used_slippery = True

        if isinstance(cell, Trap):
            self.trap_hits += 1

        if isinstance(cell, Goal):
            self.reached_goal = True

    def clear_history(self):
        self.episode_summaries.clear()

    def get_average_reward(self, last_n: int = 50) -> float:
        if not self.episode_summaries:
            return 0.0

        episodes = self.episode_summaries[-last_n:]
        return sum(episode["reward"] for episode in episodes) / len(episodes)

    def get_average_steps(self, last_n: int = 50) -> float:
        if not self.episode_summaries:
            return 0.0

        episodes = self.episode_summaries[-last_n:]
        return sum(episode["steps"] for episode in episodes) / len(episodes)

    def get_best_reward(self) -> float:
        if not self.episode_summaries:
            return 0.0

        return max(episode["reward"] for episode in self.episode_summaries)

    def get_worst_reward(self) -> float:
        if not self.episode_summaries:
            return 0.0

        return min(episode["reward"] for episode in self.episode_summaries)

    def get_risk_rate(self) -> float:
        if not self.episode_summaries:
            return 0.0

        risky_episodes = sum(
            1 for episode in self.episode_summaries
            if episode["used_slippery"]
        )

        return risky_episodes / len(self.episode_summaries)

    def get_trap_rate(self) -> float:
        if not self.episode_summaries:
            return 0.0

        trap_episodes = sum(
            1 for episode in self.episode_summaries
            if episode["trap_hits"] > 0
        )

        return trap_episodes / len(self.episode_summaries)

    def get_success_rate(self) -> float:
        if not self.episode_summaries:
            return 0.0

        successful_episodes = sum(
            1 for episode in self.episode_summaries
            if episode["reached_goal"]
        )

        return successful_episodes / len(self.episode_summaries)

    def get_average_slippery_visits(self) -> float:
        if not self.episode_summaries:
            return 0.0

        visits = [
            episode["slippery_visits"]
            for episode in self.episode_summaries
        ]

        return sum(visits) / len(visits)
