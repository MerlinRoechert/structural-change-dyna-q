import sys

from PySide6.QtWidgets import QApplication

from core.simulation_session import SimulationSession
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    session = SimulationSession()
    window = MainWindow(session)
    window.resize(1050, 700)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
