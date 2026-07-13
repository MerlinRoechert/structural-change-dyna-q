from core.simulation_session import SimulationSession


def main():
    session = SimulationSession()
    session.train(episodes=100)


if __name__ == "__main__":
    main()
