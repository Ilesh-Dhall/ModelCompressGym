---
title: ModelCompressGym Environment Server
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# ModelCompressGym Environment

ModelCompressGym is a real-world Reinforcement Learning environment designed to simulate the everyday task of an ML optimization engineer: significantly shrinking the size of a neural network to fit on edge devices while meticulously preserving model accuracy. 

A pre-trained PyTorch TinyCNN model is instantiated into the environment. The LLM agent receives precise layers and their compression statuses. Its objective is to use unstructured `pruning` actions (and optionally `quantization`) to hit a designated target model size without destroying the inference accuracy.

This requires non-trivial intelligence—specifically: understanding which linear or convolutional layers possess the most parameters, grasping that 90% reduction on an `fc` layer will likely damage accuracy less than a 90% reduction entirely across the `conv` layers, and reasoning out progressive fine-tuning vs. greedy pruning steps.

## Action Space
Defined by `ModelcompressgymAction` taking 3 variables.
1. `action_type (Literal)`: The action to dispatch. (`prune`, `quantize`, `evaluate`, `submit`).
2. `layer_name (Optional[str])`: The precise module reference target (e.g. `fc1` or `conv2`).
3. `amount (Optional[float])`: Compression ratio specifically applied to pruning masks `[0.0 - 1.0]`.

## Observation Space
Defined by `ModelcompressgymObservation` detailing the immediate model state.
1. `target_accuracy / target_size_mb (float)`: The boundaries for the assigned difficulty.
2. `model_size_mb (float)`: The current MB size of the live PyTorch Model.
3. `current_accuracy (float)`: A proxy/evaluated score representing validation accuracy.
4. `total_params (int)`: Live active parameter count.
5.  `layer_status (dict)`: Architectural representation containing parameters and individual sparsity. 
6.  `task_difficulty (str)`: Determines the current difficulty target loop (`easy`, `medium`, `hard`).

## Tasks (Difficulty Mapping)
- **Easy:** Shrink total MB by 20%. Drop in accuracy should not exceed 3% (Requires >= 0.95 Acc).
- **Medium:** Shrink total MB by 40%. Drop in accuracy should not exceed 6% (Requires >= 0.92 Acc).
- **Hard:** Shrink total MB by 60%. Drop in accuracy should not exceed 8% (Requires >= 0.90 Acc).

## Partial Rewards
A `+0.1` x `Sparsity Amount` is immediately given to incentivize size reduction incrementally. The agent is heavily penalized `-0.2` or higher if it prunes indiscriminately and accuracy falls below the `target_accuracy` threshold during the trajectory. Finally an episode score from `[0.0 to 1.0]` is given mathematically resolving both Size Ratio and Accuracy Constraints on `submit`.

## Baseline Inference Script
To use our `inference.py` to baseline models against this Gym using standard OpenEnv configuration protocols:

1. Setup authentication and target (in your shell):
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your-hf-token>"
```

2. Assure your local system has docker installed, built and exposed on port 8000:
```bash
docker build -t modelcompressgym-env:latest -f Dockerfile .
# or natively running: uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```
*(If running Docker, set `export IMAGE_NAME="modelcompressgym-env:latest"`)*

3. Run python agent pipeline:
```bash
python inference.py
```

### Reproducible Scores
- Qwen2.5-72B-Instruct baseline score hits roughly `0.6` across Easy targets recognizing `fc1` sparsity mapping.
