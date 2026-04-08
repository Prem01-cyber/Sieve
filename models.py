from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionType(str, Enum):
    CLASSIFY = "classify"
    RESPOND = "respond"
    ESCALATE = "escalate"
    ARCHIVE = "archive"
    SKIP = "skip"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    SPAM = "spam"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"


class Urgency(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Action(BaseModel):
    action_type: ActionType
    category: Optional[Category] = None
    urgency: Optional[Urgency] = None
    response_text: Optional[str] = None
    escalation_reason: Optional[str] = None
    email_id: Optional[str] = None


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    sender_tier: str = "standard"
    received_minutes_ago: int = 0


class Observation(BaseModel):
    current_email: Optional[Email] = None
    email_queue: List[Email] = []
    processed_count: int = 0
    step_count: int = 0
    task_id: str = ""
    task_description: str = ""
    available_actions: List[str] = []
    context: Dict[str, Any] = {}


class Reward(BaseModel):
    value: float
    components: Dict[str, float] = {}
    reason: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}
