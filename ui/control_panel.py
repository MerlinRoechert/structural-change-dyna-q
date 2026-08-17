from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QLabel,
)

from agents import (
    DynaQAgent,
    DynaQPlusAgent,
    LocalChangeDynaQAgent,
    QLearningAgent,
    StabilityAwareDynaQAgent,
)
from core.simulation import SimulationMode
from core.simulation_session import SimulationSession


class ControlPanel(QWidget):
    step_clicked = Signal()
    autoplay_clicked = Signal()
    stop_clicked = Signal()
    reset_episode_clicked = Signal()
    reset_all_clicked = Signal()
    export_results_clicked = Signal()

    def __init__(self, session: SimulationSession):
        super().__init__()

        self.session = session
        agent = self.session.agent

        layout = QVBoxLayout()
        self.setLayout(layout)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        self.step_button = QPushButton("Step")
        self.autoplay_button = QPushButton("Autoplay")
        self.stop_button = QPushButton("Stop")
        self.reset_episode_button = QPushButton("Reset Episode")
        self.reset_all_button = QPushButton("Reset All")
        self.export_results_button = QPushButton("Result to JSON")

        controls_layout.addWidget(self.step_button)
        controls_layout.addWidget(self.autoplay_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.reset_episode_button)
        controls_layout.addWidget(self.reset_all_button)
        controls_layout.addWidget(self.export_results_button)

        settings_group = QGroupBox("Agent Settings")
        settings_layout = QFormLayout()
        settings_group.setLayout(settings_layout)

        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0, 1.0)
        self.learning_rate_input.setSingleStep(0.01)
        self.learning_rate_input.setDecimals(2)
        self.learning_rate_input.setValue(agent.learning_rate)

        self.discount_input = QDoubleSpinBox()
        self.discount_input.setRange(0.0, 1.0)
        self.discount_input.setSingleStep(0.01)
        self.discount_input.setDecimals(2)
        self.discount_input.setValue(agent.discount_factor)

        self.epsilon_input = QDoubleSpinBox()
        self.epsilon_input.setRange(0.0, 1.0)
        self.epsilon_input.setSingleStep(0.01)
        self.epsilon_input.setDecimals(2)
        self.epsilon_input.setValue(agent.epsilon)

        self.agent_dropdown = QComboBox()
        self.agent_dropdown.addItem("Q-Learning", QLearningAgent)
        self.agent_dropdown.addItem("Dyna-Q", DynaQAgent)
        self.agent_dropdown.addItem("Dyna-Q+", DynaQPlusAgent)
        self.agent_dropdown.addItem(
            "Local-Change Dyna-Q",
            LocalChangeDynaQAgent,
        )
        self.agent_dropdown.addItem(
            "Stability-Aware Dyna-Q",
            StabilityAwareDynaQAgent,
        )

        self.planning_steps_input = QSpinBox()
        self.planning_steps_input.setRange(0, 10_000)
        self.planning_steps_input.setValue(
            DynaQAgent.DEFAULT_PLANNING_STEPS
        )

        self.exploration_bonus_input = QDoubleSpinBox()
        self.exploration_bonus_input.setRange(0.0, 1.0)
        self.exploration_bonus_input.setSingleStep(0.001)
        self.exploration_bonus_input.setDecimals(4)
        self.exploration_bonus_input.setValue(
            DynaQPlusAgent.DEFAULT_EXPLORATION_BONUS
        )

        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItem("Train", SimulationMode.TRAIN)
        self.mode_dropdown.addItem("Eval", SimulationMode.EVAL)

        self.cell_size_input = QSpinBox()
        self.cell_size_input.setRange(40, 200)
        self.cell_size_input.setValue(80)

        settings_layout.addRow("Learning Rate", self.learning_rate_input)
        settings_layout.addRow("Discount Factor", self.discount_input)
        settings_layout.addRow("Epsilon", self.epsilon_input)
        settings_layout.addRow("Agent", self.agent_dropdown)
        settings_layout.addRow("Planning Steps", self.planning_steps_input)
        settings_layout.addRow(
            "Exploration Bonus",
            self.exploration_bonus_input,
        )
        settings_layout.addRow("Simulation Mode", self.mode_dropdown)

        autoplay_group = QGroupBox("Autoplay")
        autoplay_layout = QFormLayout()
        autoplay_group.setLayout(autoplay_layout)

        self.episode_limit_input = QSpinBox()
        self.episode_limit_input.setRange(1, 10000)
        self.episode_limit_input.setValue(10)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 200)
        self.speed_slider.setValue(5)

        self.speed_label = QLabel("5 steps/s")

        autoplay_layout.addRow("Episodes per run", self.episode_limit_input)
        autoplay_layout.addRow("Speed", self.speed_slider)
        autoplay_layout.addRow("", self.speed_label)

        layout.addWidget(controls_group)
        layout.addWidget(settings_group)
        layout.addWidget(autoplay_group)
        layout.addStretch()

        self.step_button.clicked.connect(self.step_clicked.emit)
        self.autoplay_button.clicked.connect(self.autoplay_clicked.emit)
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        self.reset_episode_button.clicked.connect(self.reset_episode_clicked.emit)
        self.reset_all_button.clicked.connect(self.reset_all_clicked.emit)
        self.export_results_button.clicked.connect(self.export_results_clicked.emit)

        self.speed_slider.valueChanged.connect(self.update_speed_label)
        self.agent_dropdown.currentIndexChanged.connect(
            self.update_planning_steps_availability
        )
        self.update_planning_steps_availability()

    def update_speed_label(self):
        self.speed_label.setText(f"{self.speed_slider.value()} steps/s")

    def update_planning_steps_availability(self):
        agent_type = self.agent_dropdown.currentData()
        self.planning_steps_input.setEnabled(
            issubclass(agent_type, DynaQAgent)
        )
        self.exploration_bonus_input.setEnabled(
            agent_type is DynaQPlusAgent
        )

    def apply_settings(self):
        agent_type = self.agent_dropdown.currentData()
        self.session.set_agent_type(agent_type)
        agent = self.session.agent

        agent.learning_rate = self.learning_rate_input.value()
        agent.discount_factor = self.discount_input.value()
        agent.epsilon = self.epsilon_input.value()

        if isinstance(agent, DynaQAgent):
            agent.planning_steps = self.planning_steps_input.value()

        if isinstance(agent, DynaQPlusAgent):
            agent.exploration_bonus = self.exploration_bonus_input.value()

        self.session.simulation.mode = self.mode_dropdown.currentData()

    def get_episode_limit(self) -> int:
        return self.episode_limit_input.value()

    def get_speed(self) -> int:
        return self.speed_slider.value()

    def get_cell_size(self) -> int:
        return self.cell_size_input.value()
