# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
from uuid import uuid4
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ModelcompressgymAction, ModelcompressgymObservation
except ImportError:
    from models import ModelcompressgymAction, ModelcompressgymObservation

class CIFAR10_CNN(nn.Module):
    """A standard lightweight CNN architecture for CIFAR-10 (3x32x32 images)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 * 32 * 9 = 864 params + 32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 16x16
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32 * 64 * 9 = 18432 params + 64
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 8x8
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 * 128 * 9 = 73728 params + 128
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2) # 4x4
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512) # 2048 * 512 = 1,048,576
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10) # 512 * 10 = 5120

        # Pseudo-weights initialization for deterministic metric testing
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

class ModelcompressgymEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = -1
        self.model = None
        self.base_accuracy = 0.96 # Assumed Baseline Accuracy
        self.task_difficulty = "easy"
        self.difficulties = ["easy", "medium", "hard"]
        
        # Difficulty boundaries
        self.target_size_mb = 1.0
        self.target_accuracy = 0.90
        self.target_params = 0
        self.target_flops = 0.0
        
        # Stats caching
        self.initial_size = 1.0
        self.initial_params = 1
        self.initial_flops = 1.0
        
        # Mapping bit-widths
        self.quant_map = {
            "conv1": 32, "conv2": 32, "conv3": 32, "fc1": 32, "fc2": 32
        }

    def _calculate_flops_macs(self):
        """Calculates theoretical MACs and FLOPs for exactly 1 inference (1x3x32x32)"""
        flops = 0
        macs = 0
        
        # We explicitly track active parameters (non-zero masked weights)
        activations = {"conv1": (32, 32), "conv2": (16, 16), "conv3": (8, 8)}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                total_w = module.weight.nelement()
                mask_w = total_w
                
                # Check for structural pruning mask
                if hasattr(module, 'weight_mask'):
                    mask_w = int(torch.sum(module.weight_mask != 0).item())
                    
                if isinstance(module, nn.Conv2d):
                    # FLOPs = Active Weights * Output Spatial Dimensions
                    spatial = activations[name][0] * activations[name][1]
                    m = mask_w * spatial
                    macs += m
                    flops += m * 2
                elif isinstance(module, nn.Linear):
                    macs += mask_w
                    flops += mask_w * 2
                    
        return flops / 1e6, macs / 1e6 # Return in MegaFLOPs / MegaMACs

    def _evaluate_model(self):
        """
        Since training a model and running CIFAR-10 is noisy / timeout-risky in Docker containers,
        we use an expert deterministic mathematical proxy representing real-world pruning sensitivities.
        - Convolutional features are highly sensitive.
        - Dense layers (FC) are highly robust due to overparameterization.
        - Quantizing to float16 has minimal loss, qint8 causes strict linear loss.
        """
        drop = 0.0
        
        # Layer mappings and their penalty multipliers (Conv nets drop fast if pruned)
        sensitivities = {
            "conv1": 1.2, "conv2": 1.0, "conv3": 0.8, 
            "fc1": 0.1, "fc2": 0.3
        }
        
        for name, module in self.model.named_modules():
            if name in sensitivities:
                # Calculate sparsity
                sparsity = 0.0
                if hasattr(module, 'weight_mask'):
                    zeros = int(torch.sum(module.weight_mask == 0).item())
                    total = module.weight.nelement()
                    sparsity = zeros / total if total > 0 else 0
                
                # 1. Structural Pruning Penalty 
                # If sparsity > 30% in Conv, it starts dropping.
                # FC layers can tolerate up to 80% without drop.
                tolerance = 0.8 if "fc" in name else 0.3
                if sparsity > tolerance:
                    layer_drop = ((sparsity - tolerance) ** 2) * sensitivities[name]
                    drop += layer_drop
                    
                # 2. Quantization Penalty
                bit_width = self.quant_map[name]
                if bit_width == 16:
                    drop += 0.005 # Minor drop 
                elif bit_width == 8:
                    if "conv" in name: drop += 0.08  # Quants hit Convs hard
                    if "fc" in name: drop += 0.015
                    
        return max(0.10, self.base_accuracy - drop)

    def _get_model_size_mb(self):
        """Calculates physical memory footprint incorporating sparsity and precision types."""
        total_bits = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Check for structural pruning mask
                total_w = module.weight.nelement()
                mask_w = total_w
                if hasattr(module, 'weight_mask'):
                    mask_w = int(torch.sum(module.weight_mask != 0).item())
                
                bit_width = self.quant_map.get(name, 32)
                total_bits += (mask_w * bit_width)
                
                # Add biases (uncompressed, usually 32-bit)
                if module.bias is not None:
                    total_bits += (module.bias.nelement() * 32)
                    
        return total_bits / (8 * 1024 * 1024)
        
    def _get_layer_status(self):
        status = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                sparsity = 0.0
                if hasattr(module, 'weight_mask'):
                    sparsity = float((torch.sum(module.weight_mask == 0) / module.weight.nelement()).item())
                
                status[name] = {
                    "layer_type": module.__class__.__name__, 
                    "sparsity": round(sparsity, 3),
                    "bit_width": self.quant_map.get(name, 32),
                    "active_params": int(module.weight.nelement() * (1 - sparsity))
                }
        return status

    def _get_total_params(self):
        active = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mask_w = module.weight.nelement()
                if hasattr(module, 'weight_mask'):
                    mask_w = int(torch.sum(module.weight_mask != 0).item())
                active += mask_w
                if module.bias is not None:
                    active += module.bias.nelement()
        return active

    def reset(self) -> ModelcompressgymObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        self._reset_count += 1
        task_idx = self._reset_count % 3
        self.task_difficulty = self.difficulties[task_idx]
        
        self.model = CIFAR10_CNN()
        self.model.eval()
        self.quant_map = {name: 32 for name in ["conv1", "conv2", "conv3", "fc1", "fc2"]}
        
        self.initial_size = self._get_model_size_mb()
        self.initial_params = self._get_total_params()
        self.initial_flops, _ = self._calculate_flops_macs()

        # Constraints
        if self.task_difficulty == "easy":
            # Easy: Focus purely on pruning Parameters. 
            self.target_params = int(self.initial_params * 0.70) # 30% param drop
            self.target_size_mb = self.initial_size * 1.0 # Ignore size
            self.target_flops = self.initial_flops * 1.0 # Ignore Flops
            self.target_accuracy = 0.94
        elif self.task_difficulty == "medium":
            # Medium: Focus on Quantization for Size footprint
            self.target_params = self.initial_params # Ignore param count
            self.target_size_mb = self.initial_size * 0.50 # 50% size drop via quant
            self.target_flops = self.initial_flops * 1.0 
            self.target_accuracy = 0.92
        else:
            # Hard: Strict drop in both Memory (Quant) & FLOPs/Params (Prune)
            self.target_params = int(self.initial_params * 0.40) # 60% param drop
            self.target_size_mb = self.initial_size * 0.30 # 70% size drop
            self.target_flops = self.initial_flops * 0.60 # 40% flop drop
            self.target_accuracy = 0.90

        obs = ModelcompressgymObservation(
            total_params=self.initial_params,
            model_size_mb=self.initial_size,
            current_accuracy=self.base_accuracy,
            flops=self.initial_flops,
            macs=self.initial_flops/2.0,
            layer_status=self._get_layer_status(),
            task_difficulty=self.task_difficulty,
            target_accuracy=self.target_accuracy,
            target_size_mb=self.target_size_mb,
            target_params=self.target_params,
            target_flops=self.target_flops
        )
        return obs

    def step(self, action: ModelcompressgymAction) -> ModelcompressgymObservation:
        self._state.step_count += 1
        reward = 0.0
        done = False
        error_msg = None
        
        # Start states for marginal rewards
        start_params = self._get_total_params()
        start_size = self._get_model_size_mb()
        
        try:
            target_module = None
            if action.layer_name:
                for name, module in self.model.named_modules():
                    if name == action.layer_name:
                        target_module = module
                        break
                        
            if action.action_type == "prune":
                if target_module and isinstance(target_module, (nn.Conv2d, nn.Linear)):
                    try:
                        # Global unstructured prune stacks on mask
                        prune.l1_unstructured(target_module, name="weight", amount=action.amount)
                    except Exception as prune_err:
                        error_msg = f"Prune failed: {prune_err}"
                        reward -= 0.5
                else:
                    error_msg = f"Layer '{action.layer_name}' not found or not prunable."
                    reward -= 0.5

            elif action.action_type == "quantize":
                if target_module and action.dtype:
                    # Simulation of quant
                    if action.dtype == "qint8":
                        self.quant_map[action.layer_name] = 8
                    elif action.dtype == "float16":
                        self.quant_map[action.layer_name] = 16
                else:
                    error_msg = "Quantize requires a valid 'layer_name' and 'dtype' ('qint8' or 'float16')."
                    reward -= 0.5

            elif action.action_type == "evaluate":
                # Mild penalty to prevent excessive evaluation spam
                reward -= 0.01

            elif action.action_type == "submit":
                done = True
                
        except Exception as e:
            error_msg = str(e)
            reward -= 0.5
            
        current_acc = self._evaluate_model()
        current_size = self._get_model_size_mb()
        current_params = self._get_total_params()
        current_flops, current_macs = self._calculate_flops_macs()
        
        # Incremental task-specific rewards
        if self.task_difficulty == "easy" and current_params < start_params:
            reward += ((start_params - current_params) / self.initial_params) * 0.5
            
        elif self.task_difficulty == "medium" and current_size < start_size:
            reward += ((start_size - current_size) / self.initial_size) * 0.5
            
        elif self.task_difficulty == "hard":
            if current_params < start_params:
                reward += ((start_params - current_params) / self.initial_params) * 0.25
            if current_size < start_size:
                reward += ((start_size - current_size) / self.initial_size) * 0.25
        
        # Constraints Penalty
        if current_acc < self.target_accuracy:
            reward -= 0.5  # Heavy penalty for busting accuracy boundary!
            
        if done:
            # Grader Evaluator
            win_acc = current_acc >= self.target_accuracy
            win_params = current_params <= self.target_params if self.task_difficulty in ["easy", "hard"] else True
            win_size = current_size <= self.target_size_mb if self.task_difficulty in ["medium", "hard"] else True
            win_flops = current_flops <= self.target_flops if self.task_difficulty == "hard" else True
            
            if win_acc and win_params and win_size and win_flops:
                reward = 1.0  # Absolute flawless score
            else:
                # Partial Grader score based on proximity
                score = 0.0
                if win_acc:
                    score += 0.4
                    if self.task_difficulty == "easy":
                        progress = (self.initial_params - current_params) / (self.initial_params - self.target_params + 1e-6)
                        score += 0.6 * max(0.0, min(1.0, progress))
                    elif self.task_difficulty == "medium":
                        progress = (self.initial_size - current_size) / (self.initial_size - self.target_size_mb + 1e-6)
                        score += 0.6 * max(0.0, min(1.0, progress))
                    else: # Hard
                        prog1 = (self.initial_params - current_params) / (self.initial_params - self.target_params + 1e-6)
                        prog2 = (self.initial_size - current_size) / (self.initial_size - self.target_size_mb + 1e-6)
                        score += 0.6 * max(0.0, min(1.0, (prog1+prog2)/2.0))
                
                # Replace the reward with strict grading logic (0.0 -> 1.0)
                reward = score
            
        return ModelcompressgymObservation(
            total_params=current_params,
            model_size_mb=current_size,
            current_accuracy=current_acc,
            flops=current_flops,
            macs=current_macs,
            layer_status=self._get_layer_status(),
            error_message=error_msg,
            task_difficulty=self.task_difficulty,
            target_accuracy=self.target_accuracy,
            target_size_mb=self.target_size_mb,
            target_params=self.target_params,
            target_flops=self.target_flops,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
