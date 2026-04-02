from typing import Dict, Any, List, Optional, Tuple

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

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "processed_count": len(self.processed_emails),
            "queue_size": max(0, len(self.email_queue) - self.current_email_idx),
            "done": self.done,
            "max_steps": (
                self.task_config.get("max_steps", 0) if self.task_config else 0
            ),
            "episode_actions": self.episode_actions,
            "last_grader_score": self.last_grader_score,
        }

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

        cat_given = action.category.value if action.category else None
        urg_given = action.urgency.value if action.urgency else None

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

        if email.get("correct_category") == "spam" and action_type != "archive":
            components["spam_not_archived"] = -0.04
            total -= 0.04

        self.episode_actions.append(
            {
                "email_id": email["id"],
                "action_type": action_type,
                "category": cat_given,
                "urgency": urg_given,
                "correct_category": email.get("correct_category"),
                "correct_urgency": email.get("correct_urgency"),
                "correct_action": correct_action,
                "position": position,
            }
        )

        self.processed_emails.append(email)
        self.current_email_idx += 1

        return Reward(
            value=round(total, 4),
            components=components,
            reason=f"{email['id']} at position {position}: action={action_type}",
        ), {"email_id": email["id"]}

    def process_action(self, action: Action) -> Tuple[Reward, Dict]:
        if self.task_id == "email_classification":
            return self.process_email_classification(action)
        if self.task_id == "response_drafting":
            self.process_response_drafting(action)
        if self.task_id == "support_session":
            self.process_support_session(action)
        return Reward(value=0.0, reason="Unknown task"), {}

    def email_classification_score(self) -> float:
        total = len(TASK_CONFIGS["email_classification"]["emails"])
        cat_correct = sum(
            1
            for a in self.episode_actions
            if a.get("category") == a.get("correct_category")
        )
        urg_correct = sum(
            1
            for a in self.episode_actions
            if a.get("urgency") == a.get("correct_urgency")
        )
        return round(0.7 * cat_correct / total + 0.3 * urg_correct / total, 3)

    def response_drafting_score(self) -> float:
        emails = TASK_CONFIGS["response_drafting"]["emails"]
        total = len(emails)
        email_map = {e["id"]: e for e in emails}
        score = 0.0
        for act in self.episode_actions:
            cfg = email_map.get(act.get("email_id"))
            if not cfg:
                continue
            matched = act.get("keywords_matched", [])
            min_m = cfg["min_keyword_matches"]
            length = act.get("response_length", 0)
            kw = min(len(matched) / max(min_m, 1), 1.0)
            length_bonus = min(length / 200, 0.2) if length > 50 else 0.0
            score += kw * 0.8 + length_bonus
        return round(score / total, 3)

    def support_session_score(self) -> float:
        emails_by_id = {e["id"]: e for e in TASK_CONFIGS["support_session"]["emails"]}
        vip_ids = {
            e["id"]
            for e in TASK_CONFIGS["support_session"]["emails"]
            if e.get("sender_tier") == "vip"
        }
        high_ids = {
            e["id"]
            for e in TASK_CONFIGS["support_session"]["emails"]
            if e.get("correct_urgency") == "high" and e.get("sender_tier") != "vip"
        }
        order = [a["email_id"] for a in self.episode_actions]
        total_emails = len(TASK_CONFIGS["support_session"]["emails"])

        # Prioritization: VIP emails handled early score higher than high-urgency ones
        vip_weight = 0.20 / max(len(vip_ids), 1)
        high_weight = 0.10 / max(len(high_ids), 1)
        priority = 0.0
        for eid in vip_ids:
            if eid in order:
                pos = order.index(eid)
                priority += vip_weight if pos < 4 else vip_weight * 0.4
        for eid in high_ids:
            if eid in order:
                pos = order.index(eid)
                priority += high_weight if pos < 6 else high_weight * 0.4

        n = len(self.episode_actions)
        cat_ok = sum(
            1
            for a in self.episode_actions
            if a.get("category")
            == emails_by_id.get(a["email_id"], {}).get("correct_category")
        )
        urg_ok = sum(
            1
            for a in self.episode_actions
            if a.get("urgency")
            == emails_by_id.get(a["email_id"], {}).get("correct_urgency")
        )
        classification = cat_ok / max(n, 1) * 0.15 + urg_ok / max(n, 1) * 0.15

        act_ok = sum(
            1
            for a in self.episode_actions
            if a.get("action_type")
            == emails_by_id.get(a["email_id"], {}).get("correct_action")
        )
        action_score = act_ok / max(n, 1) * 0.30

        coverage = (n / total_emails) * 0.10

        return round(min(priority + classification + action_score + coverage, 1.0), 3)

    def compute_final_score(self) -> float:
        if not self.episode_actions:
            return 0.0
        if self.task_id == "email_classification":
            return self.email_classification_score()
        if self.task_id == "response_drafting":
            return self.response_drafting_score()
        if self.task_id == "support_session":
            return self.support_session_score()
        return 0.0

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            return (
                self.get_observation(),
                Reward(value=0.0, reason="Episode already done"),
                True,
                {},
            )
        if not self.task_id:
            raise RuntimeError("Environment not initialized, call reset() first.")

        self.step_count += 1
        reward, info = self.process_action(action)
        max_steps = self.task_config["max_steps"]
        processing_status = self.current_email_idx >= len(self.email_queue)

        if processing_status or self.step_count >= max_steps:
            self.done = True
            self.last_grader_score = self.compute_final_score()
            info["final_score"] = self.last_grader_score

        if not self.done and self.step_count > 0:
            reward.value = round(reward.value - 0.005, 4)
            reward.components["step_penality"] = -0.005

        return self.get_observation(), reward, self.done, info

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
