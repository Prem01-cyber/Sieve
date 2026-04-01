from typing import Dict, Any, Tuple
from models import Action, Observation, Reward


class EmailSortingEnvironment:
    def __init__(self) -> None:
        pass

    def reset(self, task_id: str = "email_classification") -> Observation:
        pass

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        pass

    def state(self) -> Dict[str, Any]:
        pass
