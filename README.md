# 🔐 CVE Triage Environment

An OpenEnv-compliant reinforcement learning environment that simulates the real-world task of a security engineer triaging CVE vulnerability feeds against a company's tech stack.

## 🌍 Motivation

Every week, security teams receive dozens of new CVEs. Most companies use 100+ dependencies. The hard problem isn't finding vulnerabilities — it's **knowing which ones to patch first**. This environment trains and evaluates AI agents on exactly that skill.

## 🎯 Tasks

| Task | Difficulty | Objective | Scoring |
|------|-----------|-----------|---------|
| 1. CVE Matching | Easy | Identify which CVEs affect the company stack | F1 score (precision + recall) |
| 2. Priority Ranking | Medium | Rank affected CVEs by patch priority | Kendall-Tau distance |
| 3. Full Triage | Hard | Filter + rank + suggest fix actions | Weighted combination (40/30/30) |

## 📦 Observation Space

```python
{
  "task_id": int,              # 1, 2, or 3
  "stack_id": str,             # e.g. "stack_001"
  "company_profile": str,      # plain-English company context
  "dependencies": dict,        # {"flask": "2.0.1", ...}
  "cve_feed": list[dict],      # list of CVE objects
  "task_description": str      # instructions for the agent
}
```

## 🎮 Action Space

```python
{
  "task_id": int,
  "affected_cves": list[str],       # Task 1+: CVE IDs that affect the stack
  "priority_ranking": list[str],    # Task 2+: ordered from highest to lowest priority
  "fix_suggestions": dict           # Task 3: {cve_id: "fix description"}
}
```

## 🏆 Reward Function

- **Task 1**: F1 score with -0.1 penalty per missed critical CVE (CVSS ≥ 9.0)
- **Task 2**: Kendall-Tau score + 0.1 bonus for correct #1 pick + 0.05 for correct #2
- **Task 3**: 0.4×(matching) + 0.3×(ranking) + 0.3×(fix quality) — penalties for hallucinated fixes
- **Empty action**: -0.3 penalty

## 🚀 Setup

```bash
# Install dependencies
pip install openenv-core fastapi uvicorn pydantic openai requests

# Install environment
pip install -e .

# Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Validate OpenEnv compliance
openenv validate --verbose
```

## 🧪 Running the Baseline

```bash
set OPENAI_API_KEY=your_key_here
python baseline/run_baseline.py
```

## 📊 Baseline Scores (GPT-4o)

| Task | Score |
|------|-------|
| Task 1 — CVE Matching | ~0.74 |
| Task 2 — Priority Ranking | ~0.61 |
| Task 3 — Full Triage | ~0.48 |
| **Overall** | **~0.61** |

## 🐳 Docker

```bash
docker build -t cve-triage-env -f server/Dockerfile .
docker run -p 8000:8000 cve-triage-env
```

## 📁 Project Structure

```
cve_triage_env/
├── models.py                          # Pydantic Observation + Action models
├── openenv.yaml                       # OpenEnv metadata
├── pyproject.toml                     # Dependencies
├── data/
│   ├── cves.json                      # 12 synthetic CVEs
│   ├── stacks.json                    # 5 company tech stacks
│   └── ground_truth.json             # Correct answers per stack
├── server/
│   ├── app.py                         # FastAPI server
│   ├── cve_triage_env_environment.py  # Core env logic + graders
│   └── Dockerfile
└── baseline/
    └── run_baseline.py                # OpenAI baseline agent
```
