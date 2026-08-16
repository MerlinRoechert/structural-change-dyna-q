from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QWidget
from core.simulation_session import SimulationSession
from core.world_object import Candy, Empty, Goal, Hazard, Trap, Wall


class StatusBar(QWidget):
    def __init__(self):
        super().__init__()

        self.setMaximumHeight(80)

        layout = QGridLayout()
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(2)
        self.setLayout(layout)

        title_legend = QLabel("<b>Legende</b>")
        title_status = QLabel("<b>Status</b>")

        layout.addWidget(title_legend, 0, 0)
        layout.addWidget(QLabel(f"Free: white ({Empty.reward})"), 1, 0)
        layout.addWidget(QLabel(f"Wall: gray ({Wall.reward})"), 2, 0)
        layout.addWidget(QLabel(f"Goal: green ({Goal.reward})"), 1, 1)
        layout.addWidget(QLabel(f"Trap: red ({Trap.reward})"), 2, 1)
        layout.addWidget(QLabel(f"Candy: yellow (R={Candy.reward})"), 1, 2)
        layout.addWidget(QLabel(f"Hazard: orange ({Hazard.reward})"), 2, 2)

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator, 0, 3, 3, 1)

        layout.addWidget(title_status, 0, 4)

        self.steps_label = QLabel("0")
        self.reward_label = QLabel("0")
        self.current_run_label = QLabel("0")
        self.total_episodes_label = QLabel("0")
        self.autoplay_label = QLabel("False")
        self.avg_reward_label = QLabel("0")
        self.avg_steps_label = QLabel("0")
        self.best_reward_label = QLabel("0")
        self.worst_reward_label = QLabel("0")

        layout.addWidget(QLabel("Steps:"), 1, 4)
        layout.addWidget(self.steps_label, 1, 5)

        layout.addWidget(QLabel("Reward:"), 2, 4)
        layout.addWidget(self.reward_label, 2, 5)

        layout.addWidget(QLabel("Current Run:"), 1, 6)
        layout.addWidget(self.current_run_label, 1, 7)

        layout.addWidget(QLabel("Total Episodes:"), 2, 6)
        layout.addWidget(self.total_episodes_label, 2, 7)

        layout.addWidget(QLabel("Avg Reward (50):"), 1, 8)
        layout.addWidget(self.avg_reward_label, 1, 9)

        layout.addWidget(QLabel("Avg Steps (50):"), 2, 8)
        layout.addWidget(self.avg_steps_label, 2, 9)

        layout.addWidget(QLabel("Best:"), 1, 10)
        layout.addWidget(self.best_reward_label, 1, 11)

        layout.addWidget(QLabel("Worst:"), 2, 10)
        layout.addWidget(self.worst_reward_label, 2, 11)

        layout.addWidget(QLabel("Autoplay:"), 1, 12)
        layout.addWidget(self.autoplay_label, 1, 13)

        layout.setColumnStretch(14, 1)

    def update_status(self, session: SimulationSession):
        simulation = session.simulation

        self.steps_label.setText(str(simulation.steps))
        self.reward_label.setText(str(simulation.total_reward))
        self.current_run_label.setText(str(session.current_run_episodes_done))
        self.total_episodes_label.setText(str(session.total_episodes_done))
        self.autoplay_label.setText(str(session.run_active))
        self.avg_reward_label.setText(f"{simulation.get_average_reward():.2f}")
        self.avg_steps_label.setText(f"{simulation.get_average_steps():.2f}")
        self.best_reward_label.setText(f"{simulation.get_best_reward():.2f}")
        self.worst_reward_label.setText(f"{simulation.get_worst_reward():.2f}")
