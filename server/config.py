from typing import Dict, Any

ACTION_SCHEMA: Dict[str, Any] = {
    "action_type": {
        "type": "string",
        "enum": ["classify", "respond", "escalate", "archive", "skip"],
        "required": True,
    },
    "category": {
        "type": "string",
        "enum": [
            "billing",
            "technical",
            "general",
            "spam",
            "account",
            "feature_request",
        ],
        "required": False,
    },
    "urgency": {"type": "string", "enum": ["high", "medium", "low"], "required": False},
    "response_text": {
        "type": "string",
        "required": False,
        "description": "Used with action_type=respond",
    },
    "escalation_reason": {
        "type": "string",
        "required": False,
        "description": "Used with action_type=escalate",
    },
    "email_id": {
        "type": "string",
        "required": False,
        "description": "Target email ID; used in support_session to select which email to process",
    },
}
