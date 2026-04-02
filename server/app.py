from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import Action, Observation, StepResult
from .config import ACTION_SCHEMA
from .data import TASK_CONFIGS
from .environment import EmailSortingEnvironment

app = FastAPI(title="Sieve")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
env = EmailSortingEnvironment()


@app.post("/reset", response_model=Observation)
def reset(task_id: str = "email_classification"):
    try:
        return env.reset(task_id)
    except ValueError as exec:
        raise HTTPException(status_code=400, detail=str(exec))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    if not env.task_id:
        raise HTTPException(
            status_code=400, detail="Not initialized, call /reset first."
        )
    observation, reward, status, info = env.step(action)
    return StepResult(observation=observation, reward=reward, done=status, info=info)


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": task_id,
                "name": config["name"],
                "difficulty": config["difficulty"],
                "description": config["description"],
                "max_steps": config["max_steps"],
                "action_schema": ACTION_SCHEMA,
            }
            for task_id, config in TASK_CONFIGS.items()
        ]
    }


@app.get("/grader")
def grader():
    score = env.last_grader_score
    if score is None and env.episode_actions:
        score = env.compute_final_score()
    return {
        "task_id": env.task_id,
        "score": score,
        "done": env.done,
        "processed_count": len(env.processed_emails),
        "total_emails": len(env.email_queue),
        "episode_actions_summary": [
            {
                "email_id": action["email_id"],
                "action_type": action["action_type"],
                "correct_action": action.get("correct_action"),
            }
            for action in env.episode_actions
        ],
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
