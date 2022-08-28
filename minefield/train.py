from enn_trainer import TrainConfig, State, init_train_state, train
from entity_gym.examples.minefield import Minefield
from minefield import MinefieldEnv
from gym_env_adapter import adapt_gym_env
import hyperstate


@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager[TrainConfig, State]) -> None:
    if state_manager.config.env.id == "entity_gym.Minefield":
        train(state_manager=state_manager, env=Minefield)
    elif state_manager.config.env.id == "openai_gym.Minefield":
        train(state_manager=state_manager, env=adapt_gym_env(MinefieldEnv))
    else:
        raise ValueError(f"Unsupported environment {state_manager.config.env.id}")


if __name__ == "__main__":
    main()
