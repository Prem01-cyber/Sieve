from dotenv import load_dotenv

load_dotenv()

import os
import json
import sys
from typing import Any, Dict, List, Literal, Optional
import httpx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from logger import log_start, log_step, log_end

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
API_KEY: str = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASKS = ["email_classification", "response_drafting", "support_session"]
MAX_STEPS_FALLBACK = 60
BENCHMARK = "Sieve"


class ClassifyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal[
        "billing", "technical", "general", "spam", "account", "feature_request"
    ]
    urgency: Literal["high", "medium", "low"]


class RespondOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_text: str


class SessionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email_id: str
    action_type: Literal["respond", "escalate", "archive"]
    category: Literal[
        "billing", "technical", "general", "spam", "account", "feature_request"
    ]
    urgency: Literal["high", "medium", "low"]
    response_text: Optional[str] = None
    escalation_reason: Optional[str] = None


def strict_schema(output_cls: type[BaseModel]) -> dict:
    schema = output_cls.model_json_schema()
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
        schema["additionalProperties"] = False
    return schema


def structured_call(client: OpenAI, messages: list, output_cls: type[BaseModel]):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": output_cls.__name__,
                "schema": strict_schema(output_cls),
                "strict": True,
            },
        },
        temperature=0,
    )
    return output_cls.model_validate_json(response.choices[0].message.content)


def llm_action(
    observation: Dict[str, Any], task_id: str, client: OpenAI
) -> Dict[str, Any]:
    current = observation.get("current_email") or {}
    queue: List[Dict] = observation.get("email_queue", [])

    try:
        if task_id == "email_classification":
            result = structured_call(
                client,
                messages=[
                    {
                        "role": "system",
                        "content": "Classify customer support emails by category and urgency.",
                    },
                    {
                        "role": "user",
                        "content": f"Subject: {current.get('subject', '')}\nBody: {current.get('body', '')}",
                    },
                ],
                output_cls=ClassifyOutput,
            )
            return {
                "action_type": "classify",
                "category": result.category,
                "urgency": result.urgency,
            }

        if task_id == "response_drafting":
            result = structured_call(
                client,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a professional, empathetic customer support response. "
                            "Address the issue directly, include relevant details (timelines, links, case refs). "
                            "Minimum 80 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Subject: {current.get('subject', '')}\nBody: {current.get('body', '')}",
                    },
                ],
                output_cls=RespondOutput,
            )
            return {"action_type": "respond", "response_text": result.response_text}

        if task_id == "support_session":
            queue_str = "\n".join(
                f"{e['id']} | VIP={e.get('sender_tier') == 'vip'} | {e['subject'][:70]}"
                for e in queue[:15]
            )
            result = structured_call(
                client,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You manage a customer support queue. Pick the single highest-priority email.\n"
                            "Priority: 1) VIP=True first  2) security/breach → escalate  "
                            "3) billing/technical → respond  4) feature requests/spam → archive"
                        ),
                    },
                    {"role": "user", "content": queue_str},
                ],
                output_cls=SessionOutput,
            )
            action: Dict[str, Any] = {
                "action_type": result.action_type,
                "email_id": result.email_id,
                "category": result.category,
                "urgency": result.urgency,
            }
            if result.response_text:
                action["response_text"] = result.response_text
            if result.escalation_reason:
                action["escalation_reason"] = result.escalation_reason
            return action

    except Exception as exc:
        raise RuntimeError(f"LLM call failed for task '{task_id}': {exc}") from exc


def get_task_max_steps(http: httpx.Client, task_id: str) -> int:
    try:
        tasks = http.get("/tasks").json().get("tasks", [])
        for t in tasks:
            if t["id"] == task_id:
                return t["max_steps"]
    except Exception:
        pass
    return MAX_STEPS_FALLBACK


def run_task(task_id: str, client: OpenAI) -> Dict[str, Any]:
    http = httpx.Client(base_url=ENV_BASE_URL, timeout=30.0)
    max_steps = get_task_max_steps(http, task_id)

    log_start(task_id, BENCHMARK, MODEL_NAME)

    resp = http.post("/reset", params={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    done = False
    steps = 0
    rewards: List[float] = []

    try:
        while not done and steps < max_steps:
            action = llm_action(obs, task_id, client)

            resp = http.post("/step", json=action)
            resp.raise_for_status()
            result = resp.json()

            obs = result["observation"]
            reward = result["reward"]["value"]
            done = result["done"]
            error = result.get("info", {}).get("error") or None
            steps += 1
            rewards.append(reward)

            log_step(steps, action, reward, done, error)

        grader = http.get("/grader").json()
        score = grader.get("score", 0.0)
        log_end(success=score > 0.0, steps=steps, score=score, rewards=rewards)

    except Exception as exc:
        print(f"Episode error ({task_id}): {exc}", file=sys.stderr)
        log_end(success=False, steps=steps, score=0.0, rewards=rewards)
        score = 0.0

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps,
        "total_reward": round(sum(rewards), 4),
    }


def main() -> None:
    if not API_KEY:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"Error: could not initialise OpenAI client: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Agent    : {MODEL_NAME}", file=sys.stderr)
    print(f"API base : {API_BASE_URL}", file=sys.stderr)
    print(f"Env URL  : {ENV_BASE_URL}", file=sys.stderr)

    results: List[Dict[str, Any]] = []
    for task_id in TASKS:
        print(f"Running {task_id} ...", file=sys.stderr)
        try:
            result = run_task(task_id, client)
            results.append(result)
            print(
                f"  score={result['score']}  steps={result['steps']}", file=sys.stderr
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            results.append(
                {"task_id": task_id, "score": 0.0, "steps": 0, "error": str(exc)}
            )

    avg = round(sum(r.get("score", 0.0) for r in results) / len(results), 3)
    summary = {"agent": MODEL_NAME, "results": results, "average_score": avg}
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
