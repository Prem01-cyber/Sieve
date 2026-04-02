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

## Data Models

### Enums

#### ActionType
- `classify` — Classify an email into a category and urgency
- `respond` — Draft a response to an email
- `escalate` — Escalate an email with a reason
- `archive` — Archive an email
- `skip` — Skip the current email

#### Category
- `billing` — Payment, invoices, subscription issues
- `technical` — Bugs, errors, technical failures
- `general` — General inquiries
- `spam` — Unsolicited or irrelevant messages
- `account` — Account access, settings, profile issues
- `feature_request` — Requests for new features

#### Urgency
- `high` — Requires immediate attention
- `medium` — Standard priority
- `low` — Can be handled later

### Models

#### Email
- `id` (`str`) — Unique email identifier
- `subject` (`str`) — Email subject line
- `body` (`str`) — Email body content
- `sender` (`str`) — Sender's email address
- `sender_tier` (`str`, default: `"standard"`) — Customer tier (`standard` or `vip`)
- `received_minutes_ago` (`int`, default: `0`) — How long ago the email was received

#### Action
- `action_type` (`ActionType`) — The action to perform
- `category` (`Category`, optional) — Email category, used with `classify`
- `urgency` (`Urgency`, optional) — Email urgency, used with `classify`
- `response_text` (`str`, optional) — Drafted response, used with `respond`
- `escalation_reason` (`str`, optional) — Reason for escalation, used with `escalate`
- `email_id` (`str`, optional) — Target email ID, used in `support_session` to select which email to process

#### Observation
- `current_email` (`Email`, optional) — The email currently being processed
- `email_queue` (`List[Email]`, default: `[]`) — Queue of pending emails, populated in Task 3 only
- `processed_count` (`int`, default: `0`) — Number of emails processed so far
- `step_count` (`int`, default: `0`) — Current step number
- `task_id` (`str`) — Active task identifier
- `task_description` (`str`) — Human-readable task description
- `available_actions` (`List[str]`) — Actions valid for the current state
- `context` (`Dict`) — Additional context such as `max_steps`, `remaining_steps`, `queue_size`

#### Reward
- `value` (`float`) — Total reward for the step
- `components` (`Dict[str, float]`, default: `{}`) — Breakdown of reward sub-components
- `reason` (`str`, default: `""`) — Human-readable explanation of the reward

#### StepResult
- `observation` (`Observation`) — Next environment observation
- `reward` (`Reward`) — Reward received for the action
- `done` (`bool`) — Whether the episode has ended
- `info` (`Dict`) — Additional diagnostic information

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

# Reward Categorization

The reward is dense — every step produces a signal:

| Component | Value | Trigger |
|-----------|-------|---------|
| `category_correct` | +0.15 | Correct email category |
| `urgency_correct` | +0.05 | Correct urgency level |
| `keyword_score` | 0–0.25 | Response keyword coverage |
| `adequate_length` | +0.05 | Response ≥ 50 characters |
| `vip_priority` | +0.08 | VIP email handled in first 4 steps |
| `high_priority` | +0.05 | High-urgency email handled early |
| `correct_action` | +0.06 | Correct respond/escalate/archive decision |
| `response_present` | +0.02 | Non-empty response for respond action |
| `step_penalty` | −0.005 | Applied every step (encourages efficiency) |
| `wrong_action` | −0.03 to −0.05 | Wrong action type for task |
| `spam_not_archived` | −0.04 | Spam email not archived |
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


