# Sieve : Customer Support Reinforcement Learning Environment

Primarily there are gonna be three major tasks **Email Classification**, **Response Drafting** and **Support Session**

## Email Classification - Task 1 

The agent here receives one email at a time and must classify into the categories **billing**, **technical**, **general**, **spam**, **account** and **feature_request** and respective urgencies i.e **high**, **medium** and **low**

Rewards shall be assigned for each correct category and urgency classification.

## Response Drafting - Task 2 

The agent reads a customer email and drafts a professional response. Responses are graded on:

- Coverage of required keywords for the specific issue type
- Minimum length (50+ characters)
- Professional tone

Rewards shall be based on the coverage and length of the response drafted.

## Support Session - Task 3 

The agent manages a queue of mixed emails and must perform below actions 

- Identify and prioritize high priority customers first 
- Handle high urgency emails before low urgency 
- Choosing actions i.e **respond**, **escalate** or **archive** accordingly
- Provide correct category and urgency classification

## Observation Space

```json
{
  "current_email": {
    "id": "string",
    "subject": "string",
    "body": "string",
    "sender": "string",
    "sender_tier": "standard | vip",
    "received_minutes_ago": "integer"
  },
  "email_queue": "array of Email (populated in support_session only)",
  "processed_count": "integer",
  "step_count": "integer",
  "task_id": "string",
  "task_description": "string",
  "available_actions": ["classify", "respond", "escalate", "archive", "skip"],
  "context": {
    "max_steps": "integer",
    "remaining_steps": "integer",
    "queue_size": "integer"
  }
}
```

## Action Space

```json
{
  "action_type": "classify | respond | escalate | archive | skip",
  "category": "billing | technical | general | spam | account | feature_request",
  "urgency": "high | medium | low",
  "response_text": "string (for respond action)",
  "escalation_reason": "string (for escalate action)",
  "email_id": "string (for support_session — selects which email to process)"
}
```

# Backend API 

We will be using FastAPI as out backend framework, and we are adding end points addressed as per the mentioned requirements.

## End Points

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset?task_id=<id>` | Reset environment for a task, returns initial Observation |
| `POST` | `/step` | Submit an Action, returns `{observation, reward, done, info}` |
| `GET` | `/state` | Current environment state |
| `GET` | `/tasks` | List all tasks with action schema |
| `GET` | `/grader` | Current grader score (0.0–1.0) |
| `POST` | `/baseline` | Run baseline agent and return scores for all tasks |


