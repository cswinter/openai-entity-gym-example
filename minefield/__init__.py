from gym.envs.registration import register

register(
    id="entity_gym_examples/MinefieldEnv-v0",
    entry_point="minefield:MinefieldEnv",
)
