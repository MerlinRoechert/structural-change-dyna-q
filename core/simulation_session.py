from agents import QLearningAgent
from core.grid_world import GridWorld
from core.simulation import EpisodeEndReason, Simulation, SimulationMode


class SimulationSession:
    def __init__(self):
        world = GridWorld(start_position=(0, 0))
        self.agent = QLearningAgent(world)
        self.simulation = Simulation(self.agent)

        self.run_active = False
        self.run_episode_limit = 0
        self.run_start_episode = 0

    @property
    def world(self) -> GridWorld:
        return self.agent.world

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
            self.simulation.finish_episode(
                self.simulation.terminal_end_reason
            )

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

    def set_agent_type(
        self,
        agent_type: type[QLearningAgent],
    ) -> None:
        if type(self.agent) is agent_type:
            return

        world = self.world
        learning_rate = self.agent.learning_rate
        discount_factor = self.agent.discount_factor
        epsilon = self.agent.epsilon
        simulation_mode = self.simulation.mode

        self.stop_run()
        self.agent = agent_type(
            world=world,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
        )
        self.simulation = Simulation(self.agent)
        self.simulation.mode = simulation_mode
        self.run_start_episode = 0
        self.run_episode_limit = 0

    def reset_all(self):
        self.stop_run()
        self.agent.reset_learning()
        self.simulation.clear_history()
        self.simulation.mode = SimulationMode.TRAIN
        self.run_start_episode = 0
        self.run_episode_limit = 0
        self.simulation.reset_episode()

    def run_episode(self, max_steps: int = 100):
        self.simulation.reset_episode()

        while not self.simulation.done and self.simulation.steps < max_steps:
            self.simulation.step()

        if self.simulation.done:
            end_reason = self.simulation.terminal_end_reason
        else:
            end_reason = EpisodeEndReason.MAX_STEPS

        self.simulation.finish_episode(end_reason)

    def train(self, episodes: int, max_steps: int = 100):
        for episode in range(episodes):
            self.run_episode(max_steps=max_steps)

            print(
                f"Episode {episode + 1}: "
                f"Reward={self.simulation.total_reward}, "
                f"Steps={self.simulation.steps}"
            )

        self.agent.print_q_table()
