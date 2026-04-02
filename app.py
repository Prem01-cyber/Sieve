from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from models import Action, Observation, StepResult

app = FastAPI(title="Sieve")


@app.get("/")
def root():
    pass


@app.post("/reset", response_model=Observation)
def reset(task_id: str = "email_classificaiton"):
    pass


@app.post("/step", response_model=StepResult)
def step(action: Action):
    pass


@app.get("/state")
def state():
    pass


@app.get("/tasks")
def list_tasks():
    pass


@app.get("/grader")
def grader():
    pass
