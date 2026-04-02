from typing import Any, Dict, List, Optional


def log_start(task_id: str, benchmark: str, model_name: str) -> None:
    print(f"[START] task={task_id} env={benchmark} model={model_name}", flush=True)


def log_step(
    step: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]
) -> None:
    action_str = action_to_str(action)
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def action_to_str(action: Dict[str, Any]) -> str:
    parts = [action.get("action_type", "skip")]
    if action.get("email_id"):
        parts.append(action["email_id"])
    if action.get("category"):
        parts.append(action["category"])
    if action.get("urgency"):
        parts.append(action["urgency"])
    return ":".join(parts)
