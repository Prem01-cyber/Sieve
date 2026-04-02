---
title: Sieve
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Sieve — Customer Support RL Environment

Sieve is a reinforcement learning environment that simulates a real-world customer support inbox. An AI agent interacts with it through a standard `reset() / step() / state()` HTTP API, receiving emails, taking actions, and earning rewards based on how well it handles each situation.

## How It Works

![How It Works](assets/how_it_works_v2.svg)

The agent calls `/reset` to start an episode, then loops — reading the current email from the `Observation`, posting an `Action` to `/step`, and receiving a `Reward` and next `Observation` — until `done=true`. Each step reward reflects immediate quality. A `-0.005` step penalty discourages unnecessary actions. The final grader score from `/grader` is a holistic metric computed over the full episode.

## Project Structure

```
.
├── models.py          # Shared Pydantic models (Action, Observation, Reward, etc.)
├── inference.py       # Baseline agent script using OpenAI client
├── logger.py          # Structured [START]/[STEP]/[END] stdout logger
├── openenv.yaml       # OpenEnv environment metadata
├── pyproject.toml     # Project config and dependencies
├── Dockerfile         # Container definition
└── server/
    ├── app.py         # FastAPI application and API endpoints
    ├── environment.py # Core environment logic (step, reset, reward, grader)
    ├── data.py        # Email datasets for all three tasks
    └── config.py      # Action schema definition
```

## Tasks

### Task 1 — Email Classification (Easy)

The agent receives one email at a time and must classify it using the `classify` action.

**Available action:** `classify` only

**Step Rewards**
- Correct category: `+0.15`
- Wrong category: `-0.05`
- Correct urgency: `+0.05`
- Wrong urgency: `-0.02`
- Wrong action type: `-0.05`
- Step penalty: `-0.005`

**Final Grader Score**
- Category accuracy: `70%` weight
- Urgency accuracy: `30%` weight

---

### Task 2 — Response Drafting (Medium)

The agent reads a customer email and drafts a professional response using the `respond` action.

**Available action:** `respond` only

**Step Rewards**
- Response >= 50 characters: `+0.05`
- Response < 50 characters: `-0.10`
- Keyword coverage: up to `+0.25` (scaled by `matched / min_required`)
- Negative/unprofessional tone (VADER neg > 0.4): `-0.10`
- Wrong action type: `-0.05`
- Step penalty: `-0.005`

**Final Grader Score**
- Keyword coverage weighted at `0.80`
- Length bonus up to `0.20` (scaled by `length / 200`, requires length > 50)
- Averaged across all emails in the task

---

### Task 3 — Full Support Session (Hard)

The agent manages a queue of 15 mixed emails. It must choose which email to handle, classify it, and take the right action — all in the correct priority order.

**Available actions:** `respond`, `escalate`, `archive`, `skip`

**Priority rules**
- VIP customers (`sender_tier=vip`) must be handled before standard customers
- High urgency emails take precedence over medium and low
- Security breaches and VIP incidents → `escalate`
- Spam and feature requests → `archive`
- Standard billing and technical issues → `respond`
- Use `email_id` in the action to select which email to process

**Step Rewards**
- VIP email handled in first 4 positions: `+0.08`
- VIP email delayed (position >= 4): `-0.05`
- High urgency email in first 6 positions: `+0.05`
- Low urgency email after position 6: `+0.03`
- Correct category: `+0.04`
- Correct urgency: `+0.02`
- Correct action: `+0.06`
- Wrong action: `-0.03`
- Response text provided and > 50 characters: `+0.02`
- Spam not archived: `-0.04`
- Step penalty: `-0.005`

**Final Grader Score**
- VIP prioritization: up to `0.20` (40% credit if handled late)
- High urgency prioritization: up to `0.10` (40% credit if handled late)
- Category accuracy: up to `0.15`
- Urgency accuracy: up to `0.15`
- Action accuracy: up to `0.30`
- Email coverage: up to `0.10`
- Maximum: `1.0`

---

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

## Backend API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset?task_id=<id>` | Reset environment for a task, returns initial Observation |
| `POST` | `/step` | Submit an Action, returns `{observation, reward, done, info}` |
| `GET` | `/state` | Current environment state |
| `GET` | `/tasks` | List all tasks with action schema |
| `GET` | `/grader` | Current grader score (0.0–1.0) |

## Baseline Scores

Baseline agent: `gpt-4o-mini` via OpenAI API

| Task | Score | Steps | Total Reward |
|------|-------|-------|--------------|
| Email Classification | 0.930 | 10 | 1.755 |
| Response Drafting | 0.920 | 6 | 1.650 |
| Support Session | 0.882 | 15 | 1.506 |
