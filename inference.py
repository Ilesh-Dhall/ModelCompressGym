import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import ModelcompressgymEnv
from models import ModelcompressgymAction, ModelcompressgymObservation

IMAGE_NAME = os.getenv("IMAGE_NAME", "modelcompressgym-env:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "model_compression")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "ModelCompressGym")
MAX_STEPS = 15
TEMPERATURE = 0.3
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Machine Learning Engineer.
    Your goal is to optimize a PyTorch CNN running on CIFAR-10 by strategically pruning weights or tracking quantization.
    
    RULES:
    1. Only return your action as a clean JSON object. NO markdown, NO text.
    2. Available action_types: "prune", "quantize", "evaluate", "submit".
    3. If your Task Difficulty requires fewer params, use the "prune" action_type on layers like "fc1", "fc2", "conv1" with a fractional amount. E.g. {"action_type": "prune", "layer_name": "fc1", "amount": 0.5}. Pruning dense layers (fc) cuts size without destroying accuracy instantly.
    4. If your targets are met, emit {"action_type": "submit"}.
    
    JSON EXACT FORMAT:
    {
        "action_type": "prune",
        "layer_name": "fc1",
        "amount": 0.5
    }
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs: ModelcompressgymObservation, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Task Difficulty Context: {obs.task_difficulty}
        Targets -> Active Params: <= {obs.target_params}, Size: <= {obs.target_size_mb:.2f} MB, Acc: >= {obs.target_accuracy:.3f}, Flops: <= {obs.target_flops:.2f}
        Current -> Active Params: {obs.total_params}, Size: {obs.model_size_mb:.2f} MB, Acc: {obs.current_accuracy:.3f}, Flops: {obs.flops:.2f}
        Active Layers and Sparsity -> {json.dumps(obs.layer_status, indent=2)}
        Last reward: {last_reward:.2f}
        Previous traces:
        {history_block}
        
        Send your next JSON action ONLY.
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, obs: ModelcompressgymObservation, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return text.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "submit"}'

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    env = await ModelcompressgymEnv.from_docker_image(IMAGE_NAME)
    
    try:
        for task_iter in range(3):
            task_id = f"task_{task_iter}"
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            
            result = await env.reset()
            obs = result.observation
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_json_str = get_model_message(client, step, obs, last_reward, history)
                
                try:
                    action_data = json.loads(action_json_str)
                    action = ModelcompressgymAction(**action_data)
                except Exception as e:
                    action = ModelcompressgymAction(action_type="submit")
                    action_json_str = '{"action_type": "submit", "error": "parse_failed"}'

                result = await env.step(action)
                obs = result.observation

                reward = result.reward or 0.0
                done = result.done
                action_str_inline = action_json_str.replace('\n', '').replace('\r', '')

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                log_step(step=step, action=action_str_inline, reward=reward, done=done, error=obs.error_message)
                history.append(f"Step {step}: {action_str_inline} -> reward {reward:+.2f}")

                if done:
                    score = reward
                    break

            if not done:
                score = last_reward

            score = min(max(score, 0.01), 0.99)
            
            success = score >= SUCCESS_SCORE_THRESHOLD
            log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)
    finally:
        try:
            await env.close()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
