# ETR Agent

Flask application that generates a Technical Requirements Specification (ETR)
from a business need, using Groq as the LLM provider and a harness (CAG -
Context Augmented Generation) with business rules, constraints and an output
format loaded from the `harness/` folder.

## Flow

1. **New Specification** — describe a business need and basic project info.
2. **Clarifying Questions** — the agent asks a few questions to remove ambiguity.
3. **Generation** — the agent produces the specification (runs in the
   background, with a loading screen).
4. **Review** — edit, then approve the specification.
5. **Export** — download the approved specification as PDF.

All specifications are kept in `/history`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Groq API key as an environment variable:

```bash
export GROQ_API_KEY=your_key_here     # Windows: set GROQ_API_KEY=your_key_here
```

## Run

```bash
python app.py
```

The app runs at http://localhost:5000

## Configuration (optional environment variables)

| Variable | Default | Description |
|---|---|---|
| `MAX_OUTPUT_TOKENS` | 4500 | Max tokens for the final specification |
| `QUESTIONS_MAX_TOKENS` | 500 | Max tokens for the clarifying questions |
| `RETRY_DELAY_SECONDS` | 60 | Wait between retries (Groq free tier rate limit) |

## Project structure

```
app.py                      Flask application
harness/                    Context injected into the LLM prompt (CAG)
  rules_arval.json
  constraints_arval.json
  user_guide_agent.md
  doc_technical_agent.md
templates/                  Jinja2 templates
requirements.txt
```
