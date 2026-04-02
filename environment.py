from math import exp
from operator import pos
from os import curdir, wait
from typing import Dict, Any, List, Optional, Tuple

from nltk.pathsec import validate_network_url
from models import Action, ActionType, Email, Observation, Reward
from data import TASK_CONFIGS

import nltk
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
vader = SentimentIntensityAnalyzer()


class EmailSortingEnvironment:
    def __init__(self) -> None:
        self.task_id: str = ""
        self.task_config: Dict = {}
        self.email_queue: List[Dict] = []
        self.current_email_idx: int = 0
        self.step_count: int = 0
        self.processed_emails: List[Dict] = []
        self.episode_actions: List[Dict] = []
        self.done: bool = False
        self.last_grader_score: Optional[float] = None

    def get_observation(self) -> Observation:
        if not self.task_id:
            return Observation()
        task = self.task_config
        available_actions = ["classify", "respond", "escalate", "archive", "skip"]
        if self.task_id == "email_classification":
            available_actions = ["classify"]
        elif self.task_id == "response_drafting":
            available_actions = ["respond"]

        remaining = self.email_queue[self.current_email_idx :]

        def to_email(e: Dict) -> Email:
            return Email(
                id=e["id"],
                subject=e["subject"],
                body=e["body"],
                sender=e["sender"],
                sender_tier=e["sender_tier"],
                received_minutes_ago=e["received_minutes_ago"],
            )

        if self.task_id == "support_session":
            queue = [to_email(e) for e in remaining]
            current = queue[0] if queue else None
        else:
            current = to_email(remaining[0]) if remaining else None
            queue = []

        return Observation(
            current_email=current,
            email_queue=queue,
            processed_count=len(self.processed_emails),
            step_count=self.step_count,
            task_id=self.task_id,
            task_description=task.get("description", ""),
            available_actions=available_actions,
            context={
                "max_steps": task.get("max_steps", 0),
                "remaining_steps": task.get("max_steps", 0) - self.step_count,
                "queue_size": len(remaining),
            },
        )

    def process_email_classification(self, action: Action) -> Tuple[Reward, Dict]:
        if self.current_email_idx >= len(self.email_queue):
            return Reward(value=0.0, reason="Queue is empty"), {"error": "queue_empty"}
        if action.action_type != ActionType.CLASSIFY:
            return (
                Reward(
                    value=-0.05,
                    components={"wrong_action": -0.05},
                    reason="Must use classify action",
                ),
                {},
            )
        email = self.email_queue[self.current_email_idx]
        components: Dict[str, float] = {}
        total = 0.0

        cat_given = action.category_value if action.category else None
        urg_given = action.urgency_value if action.urgency else None

        if cat_given == email.get("correct_category"):
            components["category_correct"] = 0.15
            total += 0.15
        else:
            components["category_wrong"] = -0.05
            total -= 0.05

        if urg_given == email.get("correct_urgency"):
            components["urgency_correct"] = 0.05
            total += 0.05
        else:
            components["urgency_worng"] = -0.02
            total -= 0.02

        self.episode_actions.append(
            {
                "email_id": email["id"],
                "action_type": "classify",
                "category": cat_given,
                "urgency": urg_given,
                "correct_category": email.get("correct_category"),
                "correct_urgency": email.get("correct_urgency"),
            }
        )
        self.processed_emails.append(email)
        self.current_email_idx += 1
        cat_ok = "correct" if components.get("category_correct") else "wrong"
        urg_ok = "correct" if components.get("urgency_correct") else "wrong"
        return Reward(
            value=round(total, 4),
            components=components,
            reason=f"{email['id']}: category={cat_ok}, urgency={urg_ok}",
        ), {"email_id": email["id"]}

    def process_response_drafting(self, action: Action) -> Tuple[Reward, Dict]:
        if self.current_email_idx >= len(self.email_queue):
            return Reward(value=0.0, reason="Queue empty"), {"error": "queue_empty"}
        if action.action_type != ActionType.RESPOND:
            return (
                Reward(
                    value=-0.05,
                    component={"wrong_action": -0.05},
                    reason="Must use respond action",
                ),
                {},
            )
        email = self.email_queue[self.current_email_idx]
        response = (action.response_text or "").strip()
        response_lower = response.lower()
        components: Dict[str, float] = {}
        total = 0.0
        if len(response) < 50:
            components["too_short"] = -0.1
            total -= 0.1
        else:
            components["adequate_length"] = 0.05
            total += 0.05

        required = email.get("required_keywords", [])
        min_matches = email.get("min_keyword_matches", 1)
        response_tokens = word_tokenize(response_lower)
        response_stems = {stemmer.stem(t) for t in response_tokens}
        matched = [kw for kw in required if stemmer.stem(kw.lower()) in response_stems]
        kw_score = round(min(len(matched) / max(min_matches, 1), 1.0) * 0.25, 4)
        total += kw_score

        vader_scores = vader.polarity_scores(response)
        if vader_scores["neg"] > 0.4:
            components["unprofessional"] = -0.1
            total -= 0.1

        self.episode_actions.append(
            {
                "email_id": email["id"],
                "action_type": "respond",
                "response_length": len(response),
                "keywords_matched": matched,
                "keywords_required": required,
            }
        )

        self.processed_emails.append(email)
        self.current_email_idx += 1

        return Reward(
            value=round(total, 4),
            components=components,
            reason=f"{email['id']}: {len(matched)}/{min_matches} keywords matched",
        ), {"email_id": email["id"], "keywords_matched": matched}

    def process_support_session(self, action: Action) -> Tuple[Reward, Dict]:
        remaining = self.email_queue[self.current_email_idx :]
        if not remaining:
            return Reward(value=0.0, reason="Queue empty"), {"error": "queue_empty"}

        target_idx = self.current_email_idx
        if action.email_id:
            for i, e in enumerate(remaining):
                if e["id"] == action.email_id:
                    target_idx = self.current_email_idx + i
                    break

        if target_idx != self.current_email_idx:
            self.email_queue[self.current_email_idx], self.email_queue[target_idx] = (
                self.email_queue[target_idx],
                self.email_queue[self.current_email_idx],
            )

        email = self.email_queue[self.current_email_idx]
        position = len(self.processed_emails)
        vip_check = email.get("sender_tier") == "vip"
        expected_urgency = email.get("correct_urgency", "low")
        components: Dict[str, float] = {}
        total = 0.0

        if vip_check and position < 4:
            components["vip_priority"] = 0.08
            total += 0.08
        elif vip_check and position >= 4:
            components["vip_delayed"] = -0.05
            total -= 0.05
        elif expected_urgency == "high" and position < 6:
            components["high_priority"] = 0.05
            total += 0.05
        elif expected_urgency == "low" and position > 6:
            components["low_priority"] = 0.03
            total += 0.03

        cat_given = action.category.value if action.category else None
        urg_given = action.urgency.value if action.urgency else None

        if cat_given == email.get("correct_urgency"):
            components["category_correct"] = 0.04
            total += 0.04
        if urg_given == email.get("correct_urgency"):
            components["urgency_correct"] = 0.02
            total += 0.02

        action_type = action.action_type.value
        correct_action = email.get("correct_action", "respond")

        if action_type == correct_action or (
            email.get("escalation_required") and action_type == "escalate"
        ):
            components["correct_action"] = 0.06
            total += 0.06
        elif action_type in ("respond", "escalate", "archive"):
            components["wrong_action"] = -0.03
            total -= 0.03

        if (
            action_type == "respond"
            and action.response_text
            and len(action.response_text) > 50
        ):
            components["response_present"] = 0.02
            total += 0.02

    def process_action(self, action: Action) -> Tuple[Reward, Dict]:
        if self.task_id == "email_classification":
            return self.process_email_classification(action)
        if self.task_id == "response_drafting":
            self.process_response_drafting(action)
        if self.task_id == "support_session":
            self.process_support_session(action)
        return Reward(value=0.0, reason="Unknown task"), {}

    def reset(self, task_id: str = "email_classification") -> Observation:
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task: {task_id}. Valid: {list(TASK_CONFIGS.keys())}"
            )
        self.task_id = task_id
        self.task_config = TASK_CONFIGS[task_id]
        self.email_queue = [dict(e) for e in self.task_config["emails"]]
        self.current_email_idx = 0
        self.step_count = 0
        self.processed_emails = []
        self.episode_actions = []
        self.done = False
        self.last_grader_score = None
        return self.get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        pass

    def state(self) -> Dict[str, Any]:
        pass
