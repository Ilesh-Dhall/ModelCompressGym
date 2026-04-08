# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the Modelcompressgym Environment.
"""

from typing import Dict, Any, Optional, Literal
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class ModelcompressgymAction(Action):
    """Action for the Modelcompressgym environment."""
    action_type: Literal["prune", "quantize", "evaluate", "submit"] = Field(
        ..., description="Type of action to perform: prune, quantize, evaluate, submit"
    )
    layer_name: Optional[str] = Field(
        None, description="Name of the target layer (e.g., 'fc1', 'conv1')"
    )
    amount: Optional[float] = Field(
        None, description="Sparsity amount for pruning (0.0 to 1.0)"
    )
    dtype: Optional[Literal["qint8", "float16"]] = Field(
        None, description="Data type for quantization"
    )

class ModelcompressgymObservation(Observation):
    """Observation from the Modelcompressgym environment."""
    total_params: int = Field(default=0, description="Total active number of parameters")
    model_size_mb: float = Field(default=0.0, description="Model physical footprint in MB")
    current_accuracy: float = Field(default=0.0, description="Current evaluated accuracy (0.0 - 1.0)")
    flops: float = Field(default=0.0, description="Total Floating Point Operations (MegaFLOPs)")
    macs: float = Field(default=0.0, description="Total Multiply-Accumulates (MegaMACs)")
    
    layer_status: Dict[str, Any] = Field(default_factory=dict, description="Status mapping of each active layer")
    error_message: Optional[str] = Field(None, description="Error message if the last action failed")
    
    task_difficulty: str = Field(default="easy", description="Current allocated task difficulty (easy | medium | hard)")
    target_accuracy: float = Field(default=0.90, description="Minimum validation accuracy to preserve")
    target_size_mb: float = Field(default=1.0, description="Max acceptable model size in MB")
    target_params: int = Field(default=0, description="Max acceptable active parameters")
    target_flops: float = Field(default=0.0, description="Max acceptable MegaFLOPs")
