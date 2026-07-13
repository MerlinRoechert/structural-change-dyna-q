from core.agent import Agent
from core.grid_world import GridWorld
from core.simulation import Simulation, SimulationMode


class SimulationSession:
    def __init__(self):
        self.world = GridWorld(start_position=(0, 0))
        self.agent = Agent()
        self.simulation = Simulation(self.world, self.agent)

        self.run_active = False
        self.run_episode_limit = 0
        self.run_start_episode = 0

    @property
    def total_episodes_done(self) -> int:
        return len(self.simulation.episode_summaries)

    @property
    def current_run_episodes_done(self) -> int:
        return self.total_episodes_done - self.run_start_episode

    def step(self):
        if self.simulation.done:
            return

        self.simulation.step()

        if self.simulation.done:
            self.simulation.finish_episode()

    def start_run(self, episode_limit: int):
        self.run_start_episode = self.total_episodes_done
        self.run_episode_limit = episode_limit
        self.run_active = True
        self.simulation.reset_episode()

    def advance_run(self):
        if not self.run_active:
            return

        if self.simulation.done:
            self.simulation.reset_episode()

        self.step()

        if self.current_run_episodes_done >= self.run_episode_limit:
            self.stop_run()

    def stop_run(self):
        self.run_active = False

    def reset_episode(self):
        self.simulation.reset_episode()

    def reset_all(self):
        self.stop_run()
        self.agent.q_table.clear()
        self.simulation.clear_history()
        self.simulation.mode = SimulationMode.TRAIN
        self.run_start_episode = 0
        self.run_episode_limit = 0
        self.simulation.reset_episode()

    def run_episode(self, max_steps: int = 100):
        self.simulation.reset_episode()

        while not self.simulation.done and self.simulation.steps < max_steps:
            self.simulation.step()

        self.simulation.finish_episode()

    def train(self, episodes: int, max_steps: int = 100):
        for episode in range(episodes):
            self.run_episode(max_steps=max_steps)

            print(
                f"Episode {episode + 1}: "
                f"Reward={self.simulation.total_reward}, "
                f"Steps={self.simulation.steps}"
            )

        self.agent.print_q_table()
