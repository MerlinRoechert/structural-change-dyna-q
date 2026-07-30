import json
from datetime import datetime

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from agents import DynaQAgent, DynaQPlusAgent
from core.simulation_session import SimulationSession
from ui.control_panel import ControlPanel
from ui.grid_widget import GridWidget
from ui.status_bar import StatusBar


class MainWindow(QMainWindow):
    def __init__(self, session: SimulationSession):
        super().__init__()

        self.session = session
        self.setWindowTitle("RL GridWorld")

        self.control_panel = ControlPanel(self.session)
        self.grid_widget = GridWidget(
            self.session.world,
            self.session.agent,
            cell_size=self.control_panel.get_cell_size(),
        )
        self.grid_widget.map_changed.connect(self.on_map_changed)
        self.status_bar_widget = StatusBar()

        self.grid_scroll_area = QScrollArea()
        self.grid_scroll_area.setWidget(self.grid_widget)
        self.grid_scroll_area.setWidgetResizable(False)

        self.control_scroll_area = QScrollArea()
        self.control_scroll_area.setWidget(self.control_panel)
        self.control_scroll_area.setWidgetResizable(True)
        self.control_scroll_area.setMinimumWidth(320)

        main_content = QWidget()
        main_content_layout = QHBoxLayout()
        main_content.setLayout(main_content_layout)

        main_content_layout.addWidget(self.grid_scroll_area, stretch=1)
        main_content_layout.addWidget(self.control_scroll_area)

        root = QWidget()
        root_layout = QVBoxLayout()
        root.setLayout(root_layout)

        root_layout.addWidget(main_content, stretch=1)
        root_layout.addWidget(self.status_bar_widget)

        self.setCentralWidget(root)

        self.timer = QTimer()
        self.timer.timeout.connect(self.autoplay_step)

        self.connect_signals()
        self.update_ui()

    def connect_signals(self):
        self.control_panel.step_clicked.connect(self.step)
        self.control_panel.autoplay_clicked.connect(self.start_autoplay)
        self.control_panel.stop_clicked.connect(self.stop_autoplay)
        self.control_panel.reset_episode_clicked.connect(self.reset_episode)
        self.control_panel.reset_all_clicked.connect(self.reset_all)
        self.control_panel.export_results_clicked.connect(self.export_results_to_json)

    def step(self):
        self.control_panel.apply_settings()
        self.session.step()
        self.update_ui()

    def start_autoplay(self):
        self.control_panel.apply_settings()
        self.session.start_run(self.control_panel.get_episode_limit())

        self.update_timer_speed()
        self.timer.start()
        self.update_ui()

    def stop_autoplay(self):
        self.session.stop_run()
        self.timer.stop()
        self.update_ui()

    def autoplay_step(self):
        self.control_panel.apply_settings()
        self.session.advance_run()

        if not self.session.run_active:
            self.timer.stop()

        self.update_timer_speed()
        self.update_ui()

    def reset_episode(self):
        self.session.reset_episode()
        self.update_ui()

    def reset_all(self):
        self.timer.stop()
        self.session.reset_all()
        self.update_ui()

    def update_timer_speed(self):
        interval_ms = int(1000 / self.control_panel.get_speed())
        self.timer.setInterval(interval_ms)

    def update_grid_size(self):
        new_cell_size = self.control_panel.get_cell_size()

        if new_cell_size != self.grid_widget.cell_size:
            self.grid_widget.cell_size = new_cell_size
            self.grid_widget.setMinimumSize(
                self.session.world.width * new_cell_size,
                self.session.world.height * new_cell_size,
            )
            self.grid_widget.resize(
                self.session.world.width * new_cell_size,
                self.session.world.height * new_cell_size,
            )

    def update_ui(self):
        self.grid_widget.agent = self.session.agent
        self.update_grid_size()
        self.grid_widget.update()
        self.status_bar_widget.update_status(self.session)

    def on_map_changed(self):
        self.session.reset_episode()
        self.update_ui()

    def export_results_to_json(self):
        self.control_panel.apply_settings()

        data = {
            "config": {
                "learning_rate": self.session.agent.learning_rate,
                "discount_factor": self.session.agent.discount_factor,
                "epsilon": self.session.agent.epsilon,
                "agent_type": self.session.agent.name,
                "simulation_mode": self.session.simulation.mode,
                "autoplay_episode_limit": self.control_panel.get_episode_limit(),
                "autoplay_speed": self.control_panel.get_speed(),
                "cell_size": self.control_panel.get_cell_size(),
                "start_position": self.session.world.start_position,
                "map_height": self.session.world.height,
                "map_width": self.session.world.width,
            },
            "results": {
                "total_episodes_done": self.session.total_episodes_done,
                "current_run_episodes_done": self.session.current_run_episodes_done,
                "current_episode_steps": self.session.simulation.steps,
                "current_episode_reward": self.session.simulation.total_reward,
                "avg_reward_last_50": self.session.simulation.get_average_reward(50),
                "avg_steps_last_50": self.session.simulation.get_average_steps(50),
                "best_reward": self.session.simulation.get_best_reward(),
                "worst_reward": self.session.simulation.get_worst_reward(),
                "risk_rate": self.session.simulation.get_risk_rate(),
                "trap_rate": self.session.simulation.get_trap_rate(),
                "success_rate": self.session.simulation.get_success_rate(),
                "avg_slippery_visits": self.session.simulation.get_average_slippery_visits(),
            },
            "episodes": self.session.simulation.episode_summaries,
            "created_at": datetime.now().isoformat(),
        }

        if isinstance(self.session.agent, DynaQAgent):
            data["config"]["planning_steps"] = (
                self.session.agent.planning_steps
            )

        if isinstance(self.session.agent, DynaQPlusAgent):
            data["config"]["exploration_bonus"] = (
                self.session.agent.exploration_bonus
            )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save results",
            "rl_results.json",
            "JSON Files (*.json)",
        )

        if not file_path:
            return

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_ui()
