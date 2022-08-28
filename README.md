# OpenAI Gym - EntityGym Integration

This repository contains a small example of how to use an OpenAI Gym environment with EntityGym/ENN-Trainer.


## Code overview

- [minefield.py](minefield/minefield.py) file contains a OpenAI Gym version of the EntityGym [Minefield example environment](https://github.com/entity-neural-network/entity-gym/blob/cb7e5a7d13edcc9ebbff6601511b21e5d555f73e/entity_gym/examples/minefield.py).
- [gym_env_adapter.py](minefield/gym_env_adapter.py) implements a generic adapter for OpenAI Gym environments to EntityGym environments.
- [train.py](train.py) is a script that uses trains an agent on either the OpenAI Gym or EntityGym version of the Minefield environment with enn-trainer.

## Usage

To install dependencies with poetry:

```bash
poetry install
poetry run pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

Running training with the OpenAI Gym environment implementation:

```bash
poetry run python minefield/train.py env.id=openai_gym.Minefield net.translation='(reference_entity: "Vehicle", position_features: ["_0", "_1"], rotation_angle_feature: "_2")'
```

Running training with the EntityGym implementation:

```bash
poetry run python minefield/train.py env.id=entity_gym.Minefield net.translation='(reference_entity: "Vehicle", position_features: ["x_pos", "y_pos"], rotation_vec_features: ["x_pos", "y_pos"])'
```

Comparison in W&B: [https://wandb.ai/entity-neural-network/enn-ppo/reports/OpenAI-Gym-vs-EntityGym-Minefield-implementation--VmlldzoyNTQwMDI0](https://wandb.ai/entity-neural-network/enn-ppo/reports/OpenAI-Gym-vs-EntityGym-Minefield-implementation--VmlldzoyNTQwMDI0)