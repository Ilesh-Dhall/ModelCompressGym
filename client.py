# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Modelcompressgym Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import ModelcompressgymAction, ModelcompressgymObservation
except ImportError:
    from ModelCompressGym.models import ModelcompressgymAction, ModelcompressgymObservation


class ModelcompressgymEnv(
    EnvClient[ModelcompressgymAction, ModelcompressgymObservation, State]
):
    """Client for the Modelcompressgym Environment."""

    def _step_payload(self, action: ModelcompressgymAction) -> Dict:
        return action.model_dump(exclude_unset=True)

    def _parse_result(self, payload: Dict) -> StepResult[ModelcompressgymObservation]:
        obs_data = payload.get("observation", {})
        observation = ModelcompressgymObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
