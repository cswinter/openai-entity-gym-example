from typing import Dict, Mapping

import gym.spaces
from entity_gym.env import (
    Action,
    ActionSpace,
    Environment,
    Observation,
    ObsSpace,
    Entity,
    GlobalCategoricalActionSpace,
    GlobalCategoricalActionMask,
)


def adapt_gym_obs_space(gym_space) -> ObsSpace:
    assert isinstance(gym_space, gym.spaces.Dict), "Only gym.spaces.Dict is supported"
    entities = {}
    for key, subspace in gym_space.items():
        assert isinstance(subspace, gym.spaces.Box), "Only gym.spaces.Box is supported"
        assert len(subspace.shape) == 1, "Only 1D spaces are supported"
        entities[key] = Entity(features=[f"_{i}" for i in range(subspace.shape[0])])
    return ObsSpace(entities=entities)


def adapt_action_space(gym_space) -> ActionSpace:
    assert isinstance(
        gym_space, gym.spaces.Discrete
    ), "Only gym.spaces.Discrete is supported"
    return GlobalCategoricalActionSpace([f"_{i}" for i in range(gym_space.n)])


def adapt_gym_env(env):
    class AdaptedGymEnv(Environment):
        def __init__(self, **kwargs) -> None:
            self.env = env(**kwargs)
            self._obs_space = adapt_gym_obs_space(self.env.observation_space)
            self._action_space = {"action": adapt_action_space(self.env.action_space)}
            self._action_mask = {"action": GlobalCategoricalActionMask()}

        def obs_space(self) -> ObsSpace:
            return self._obs_space

        def action_space(self) -> Dict[str, ActionSpace]:
            return self._action_space

        def reset_filter(self, obs_space: ObsSpace) -> Observation:
            return self.reset()

        def reset(self) -> Observation:
            obs, info = self.env.reset()
            return Observation(
                features=obs,
                reward=0,
                done=False,
                metrics=info,
                actions=self._action_mask,
            )

        def act_filter(
            self, action: Mapping[str, Action], obs_filter: ObsSpace
        ) -> Observation:
            return self.act(action)

        def act(self, actions: Mapping[str, Action]) -> Observation:
            obs, rew, done, info = self.env.step(actions["action"].index)
            return Observation(
                features=obs,
                reward=rew,
                done=done,
                metrics=info,
                actions=self._action_mask,
            )

    return AdaptedGymEnv
