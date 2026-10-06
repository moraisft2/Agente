"""
ETR Agent - flow P0 (Build: new need from scratch)

Pages / flow:
  /                    Home - fill the business need (Step 1: Fill basic questions)
  /clarify             POST -> AI generates clarifying questions, then:
                          - if it found questions -> redirect to /answer/<id>
                          - if it found none       -> starts generation in the
                                                       background, redirect to /loading/<id>
  /answer/<id>         Shows the clarifying questions, user fills answers
  /generate/<id>       POST -> starts generation in the background (need + Q&A),
                                redirects to /loading/<id>
  /loading/<id>        Shown while the agent is working; polls /status/<id> and
                        redirects to /view/<id> once the document is ready
  /status/<id>         JSON endpoint used by /loading/<id> to poll progress
  /view/<id>           Shows one specification (draft or approved): Edit / Approve / Export PDF
  /edit/<id>           Edit the generated markdown before approval (human review: edit & refine)
  /approve/<id>        POST -> marks the spec as approved
  /export-pdf/<id>     Downloads the approved spec as PDF
  /history             Lists every specification ever generated, clickable

Pipeline (Step 2-4 of the diagram), run in a background thread so the user
gets a loading screen instead of a frozen request:
  prepare_request()       -> build the LLM messages (harness + project data + need + Q&A)
  call_llm()               -> Groq request
  validate_spec()          -> "Spec generation OK?" decision
        No  -> back to call_llm() with feedback (loop, max MAX_ATTEMPTS)
        Yes -> specification is shown to the user for review (edit) and approval

Harness (CAG - Context Augmented Generation), loaded from ./harness:
  - rules_arval.json          (Rules - Arval)
  - constraints_arval.json    (Constraints - Arval)
  - user_guide_agent.md       (User guide - Agent)
  - doc_technical_agent.md    (Doc technical - Agent)
"""

import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import markdown
from flask import Flask, jsonify, make_response, redirect, render_template, request, url_for
from groq import Groq
from xhtml2pdf import pisa

BASE_DIR = Path(__file__).parent
HARNESS_DIR = BASE_DIR / "harness"
DB_NAME = BASE_DIR / "etr_agent.db"

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
MAX_ATTEMPTS = 3  # max LLM requests in the "No" loop of the diagram

# Groq free tier: 8000 tokens/minute, counting prompt + max_tokens of the request.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4500"))
QUESTIONS_MAX_TOKENS = int(os.getenv("QUESTIONS_MAX_TOKENS", "500"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "60"))

HARNESS_FILES = [
    ("RULES ARVAL", "rules_arval.json"),
    ("CONSTRAINTS ARVAL", "constraints_arval.json"),
    ("USER GUIDE (AGENT)", "user_guide_agent.md"),
    ("DOC TECHNICAL (AGENT)", "doc_technical_agent.md"),
]

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("etr_agent")


# ============================================================
# DATABASE
# ============================================================

def db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS etr (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                approved_at TEXT,
                status TEXT DEFAULT 'questions',   -- questions | generating | draft | approved | error
                client TEXT,
                feature TEXT,
                requester TEXT,
                country TEXT,
                actor TEXT,
                need TEXT,
                questions_json TEXT,
                qa_json TEXT,
                content_md TEXT,
                attempts INTEGER,
                issues_json TEXT
            )
        """)


init_db()


# ============================================================
# HARNESS - CAG (Context Augmented Generation)
# ============================================================

def load_harness() -> str:
    blocks = []
    for title, filename in HARNESS_FILES:
        path = HARNESS_DIR / filename
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            content = json.dumps(json.loads(content), separators=(",", ":"), ensure_ascii=False)
        blocks.append(f"--- {title} ---\n{content}")
    return "\n\n".join(blocks)


def project_block(project: dict) -> str:
    return f"""Client: {project['client']}
Feature: {project['feature']}
Requester: {project['requester']}
Date: {project['date']}
Country: {project['country']}
Main actor: {project['actor']}"""


# ============================================================
# STEP 1b - CLARIFYING QUESTIONS
# ============================================================

def prepare_questions_request(project: dict, need: str) -> list:
    context = f"""
HARNESS - INJECTED CONTEXT
{load_harness()}

PROJECT INFORMATION
{project_block(project)}

BUSINESS NEED
{need}

INSTRUCTION
Identify 3 to 5 points that are ambiguous or missing in this business need,
that a requirements engineer would need clarified before writing a specification.
Reply with ONLY a JSON array of short questions (plain strings).
No markdown, no numbering, no extra text, no code fences.
If the need is already fully clear, reply with an empty JSON array: []
"""
    return [
        {"role": "system", "content": "You are a requirements engineering agent for long-term vehicle leasing."},
        {"role": "user", "content": context},
    ]


def parse_questions(raw: str) -> list:
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(q).strip() for q in data if str(q).strip()][:5]
    except (json.JSONDecodeError, TypeError):
        pass
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip() for line in cleaned.splitlines()]
    return [line for line in lines if line][:5]


def generate_clarifying_questions(project: dict, need: str) -> list:
    messages = prepare_questions_request(project, need)
    raw = call_llm(messages, max_tokens=QUESTIONS_MAX_TOKENS)
    return parse_questions(raw)


# ============================================================
# STEP 2 - PREPARE REQUEST (final specification)
# ============================================================

def prepare_request(project: dict, need: str, qa_pairs: list) -> list:
    qa_block = ""
    answered = [(q, a) for q, a in qa_pairs if a and a.strip()]
    if answered:
        qa_lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in answered)
        qa_block = f"\nCLARIFYING QUESTIONS AND ANSWERS\n{qa_lines}\n"

    context = f"""
HARNESS - INJECTED CONTEXT
{load_harness()}

PROJECT INFORMATION
{project_block(project)}

BUSINESS NEED
{need}
{qa_block}
INSTRUCTION
Produce the technical requirements specification matching this need,
following exactly the format defined in DOC TECHNICAL (AGENT) and using
the context provided by the harness.
Use the clarifying answers above to remove ambiguity. Anything still
unclear or left unanswered must be listed in section 4 (OPEN QUESTIONS)
with its own OQ-00X ID instead of being invented. If that same point
resurfaces in sections 5 to 11, write "Pending — see OQ-00X" instead of
a concrete number, provider name or technology.
Never describe the agent itself.
Use exactly the project values above.
"""
    return [
        {"role": "system", "content": "You are a requirements engineering agent for long-term vehicle leasing."},
        {"role": "user", "content": context},
    ]


# ============================================================
# STEP 3 - LLM REQUESTS
# ============================================================

def call_llm(messages: list, max_tokens: int = None) -> str:
    if not API_KEY:
        raise RuntimeError("The GROQ_API_KEY environment variable is not set.")

    response = Groq(api_key=API_KEY).chat.completions.create(
        messages=messages,
        model=MODEL,
        temperature=0.2,
        max_tokens=max_tokens or MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content


# ============================================================
# STEP 4 - DECISION: SPEC GENERATION OK?
# ============================================================

HEADER_FIELDS = ["Client", "Project", "Requester", "Date", "Country", "Main actor"]
GROUP_HEADERS = ["CORE SPECIFICATION", "EXTENDED SPECIFICATION"]
REQUIRED_SECTIONS = {
    1: "INVOLVED ACTORS",
    2: "FUNCTIONAL REQUIREMENTS",
    3: "ACCEPTANCE CRITERIA",
    4: "OPEN QUESTIONS",
    5: "BUSINESS RULES",
    6: "NON-FUNCTIONAL REQUIREMENTS",
    7: "LEGAL REQUIREMENTS",
    8: "SECURITY REQUIREMENTS",
    9: "ARCHITECTURAL REQUIREMENTS",
    10: "USER STORIES",
    11: "TRACEABILITY MATRIX",
}
REQUIRED_ID_PREFIXES = ["FR", "BR", "NFR", "LR", "SR", "AR", "US", "OQ"]
FORBIDDEN_TERMS = r"\b(LLM|RAG|prompt|prompts)\b"


def validate_spec(md: str) -> list:
    """Return the list of problems found. Empty list = 'Yes' branch of the diagram."""
    issues = []

    if not re.search(r"^#\s+TECHNICAL REQUIREMENTS SPECIFICATION", md, re.M | re.I):
        issues.append("Missing the main title '# TECHNICAL REQUIREMENTS SPECIFICATION'.")

    for field in HEADER_FIELDS:
        if not re.search(rf"\*\*{re.escape(field)}:?\*\*", md, re.I):
            issues.append(f"Missing header field '**{field}:**'.")

    for name in GROUP_HEADERS:
        if not re.search(rf"^##\s*{re.escape(name)}", md, re.M | re.I):
            issues.append(f"Missing group heading '## {name}'.")

    for number, name in REQUIRED_SECTIONS.items():
        if not re.search(rf"^###\s*{number}\.\s*{re.escape(name)}", md, re.M | re.I):
            issues.append(f"Missing or misnamed section '### {number}. {name}'.")

    for prefix in REQUIRED_ID_PREFIXES:
        if not re.search(rf"\b{prefix}-\d{{3}}\b", md):
            issues.append(f"No requirement with ID format {prefix}-001.")

    for keyword in ("Given", "When", "Then"):
        if not re.search(rf"\b{keyword}\b", md):
            issues.append(f"BDD criteria must use the keyword '{keyword}'.")

    if re.search(FORBIDDEN_TERMS, md, re.I):
        issues.append("The document must not mention LLM, RAG, prompts or generation.")

    issues += check_open_questions_not_invented(md)

    return issues


def check_open_questions_not_invented(md: str) -> list:
    """Soft check: flag concrete numeric/tech values in sections 5-11 that look like
    they answer a topic word also present in an Open Question, and that were not
    marked as 'Pending'. This is a heuristic, not a full semantic check."""
    issues = []

    oq_match = re.search(r"^###\s*4\.\s*OPEN QUESTIONS(.*?)(?=^##\s)", md, re.M | re.S | re.I)
    ext_match = re.search(r"^##\s*EXTENDED SPECIFICATION(.*)", md, re.M | re.S | re.I)
    if not oq_match or not ext_match:
        return issues

    oq_text = oq_match.group(1)
    ext_text = ext_match.group(1)

    # Keywords that, if they appear in an open question, suggest a concrete decision
    # about them should not be fabricated later in the document.
    watch_terms = {
        "latency": r"\b\d+(\.\d+)?\s*(ms|milliseconds?|seconds?|s)\b",
        "broker": r"\b(kafka|rabbitmq|sqs|azure service bus|pub/sub)\b",
        "language": r"\b(french|english|spanish|portuguese)\b.{0,15}\b(default|primary|optional)\b",
        "retry": r"\bretry (policy|limit|count)\b.{0,20}\d+",
    }
    for keyword, pattern in watch_terms.items():
        if keyword in oq_text.lower() and re.search(pattern, ext_text, re.I):
            issues.append(
                f"'{keyword}' is listed as an open question but a concrete value for it "
                f"appears in the Extended Specification instead of 'Pending — see OQ-00X'."
            )

    return issues


def run_agent(project: dict, need: str, qa_pairs: list):
    """Loop of the diagram: LLM requests -> Spec generation? -> (No) back to LLM requests."""
    base_messages = prepare_request(project, need, qa_pairs)
    messages = base_messages
    document_md, issues = "", []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1 and RETRY_DELAY_SECONDS:
            time.sleep(RETRY_DELAY_SECONDS)  # let the tokens-per-minute window reset

        document_md = call_llm(messages)
        issues = validate_spec(document_md)
        logger.info("LLM attempt %s/%s - %s issue(s)", attempt, MAX_ATTEMPTS, len(issues))

        if not issues:  # Yes
            return document_md, attempt, []

        feedback = (
            "\n\nIMPORTANT - a previous attempt was rejected for these problems. "
            "Avoid them and be concise so the document fits in the output limit:\n- "
            + "\n- ".join(issues)
        )
        messages = [base_messages[0], {"role": "user", "content": base_messages[1]["content"] + feedback}]

    return document_md, MAX_ATTEMPTS, issues


# ============================================================
# BACKGROUND GENERATION (so the user sees a loading screen)
# ============================================================

def row_project(row) -> dict:
    return {
        "client": row["client"],
        "feature": row["feature"],
        "requester": row["requester"],
        "country": row["country"],
        "actor": row["actor"],
        "date": datetime.now().strftime("%d/%m/%Y"),
    }


def run_generation_for(etr_id: int, qa_pairs: list):
    """Runs in a background thread. Writes its own progress/result to the DB."""
    try:
        with db() as conn:
            row = conn.execute("SELECT * FROM etr WHERE id = ?", (etr_id,)).fetchone()
        if row is None:
            return

        document_md, attempts, issues = run_agent(row_project(row), row["need"], qa_pairs)

        with db() as conn:
            conn.execute(
                "UPDATE etr SET status = 'draft', qa_json = ?, content_md = ?, attempts = ?, issues_json = ? WHERE id = ?",
                (json.dumps(qa_pairs), document_md, attempts, json.dumps(issues), etr_id),
            )
        logger.info("ETR draft id=%s", etr_id)

    except Exception as e:
        logger.exception("Background generation error for id=%s: %s", etr_id, e)
        with db() as conn:
            conn.execute(
                "UPDATE etr SET status = 'error', issues_json = ? WHERE id = ?",
                (json.dumps([str(e)]), etr_id),
            )


def start_generation(etr_id: int, qa_pairs: list):
    with db() as conn:
        conn.execute("UPDATE etr SET status = 'generating' WHERE id = ?", (etr_id,))
    threading.Thread(target=run_generation_for, args=(etr_id, qa_pairs), daemon=True).start()


# ============================================================
# ROUTES
# ============================================================

def to_html(md: str) -> str:
    return markdown.markdown(md, extensions=["extra", "tables"])


@app.route("/")
def index():
    return render_template("index.html", active="home")


@app.route("/clarify", methods=["POST"])
def clarify():
    f = request.form
    need = f.get("need", "").strip()
    if not need:
        return redirect(url_for("index"))

    project = {
        "client": f.get("client", "").strip() or "Not provided",
        "feature": f.get("feature", "").strip() or "Not provided",
        "requester": f.get("requester", "").strip() or "Not provided",
        "country": f.get("country", "France"),
        "actor": f.get("actor", "").strip() or "Not provided",
        "date": datetime.now().strftime("%d/%m/%Y"),
    }

    try:
        questions = generate_clarifying_questions(project, need)
    except Exception as e:
        logger.exception("Clarifying questions error: %s", e)
        questions = []  # fall back to generating directly

    with db() as conn:
        cur = conn.execute(
            """INSERT INTO etr (created_at, status, client, feature, requester, country, actor, need, questions_json)
               VALUES (?, 'questions', ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime("%d/%m/%Y %H:%M"),
                project["client"], project["feature"], project["requester"],
                project["country"], project["actor"], need, json.dumps(questions),
            ),
        )
        etr_id = cur.lastrowid

    if not questions:
        start_generation(etr_id, [])
        return redirect(url_for("loading", etr_id=etr_id))

    return redirect(url_for("answer", etr_id=etr_id))


@app.route("/answer/<int:etr_id>")
def answer(etr_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM etr WHERE id = ?", (etr_id,)).fetchone()
    if row is None or row["status"] != "questions":
        return redirect(url_for("index"))

    questions = json.loads(row["questions_json"] or "[]")
    return render_template("answer.html", etr=row, questions=questions, active="home")


@app.route("/generate/<int:etr_id>", methods=["POST"])
def generate(etr_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM etr WHERE id = ?", (etr_id,)).fetchone()
    if row is None:
        return redirect(url_for("index"))

    questions = json.loads(row["questions_json"] or "[]")
    answers = [request.form.get(f"answer_{i}", "").strip() for i in range(len(questions))]
    qa_pairs = list(zip(questions, answers))

    start_generation(etr_id, qa_pairs)
    return redirect(url_for("loading", etr_id=etr_id))


@app.route("/loading/<int:etr_id>")
def loading(etr_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM etr WHERE id = ?", (etr_id,)).fetchone()
    if row is None:
        return redirect(url_for("index"))
    if row["status"] not in ("generating",):
        return redirect(url_for("view", etr_id=etr_id))
    return render_template("loading.html", etr=row, active="home")


@app.route("/status/<int:etr_id>")
def status(etr_id):
    with db() as conn:
        row = conn.execute("SELECT status, attempts FROM etr WHERE id = ?", (etr_id,)).fetchone()
    if row is None:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": row["status"], "attempts": row["attempts"]})


@app.route("/view/<int:etr_id>")
def view(etr_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM etr WHERE id = ?", (etr_id,)).fetchone()
    if row is None:
        return redirect(url_for("index"))
    if row["status"] == "generating":
        return redirect(url_for("loading", etr_id=etr_id))

    issues = json.loads(row["issues_json"] or "[]")
    return render_template(
        "view.html",
        etr=row,
        document_html=to_html(row["content_md"] or ""),
        attempts=row["attempts"],
        issues=issues,
        active="view",
    )


@app.route("/edit/<int:etr_id>", methods=["GET", "POST"])
def edit(etr_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM etr WHERE id = ?", (etr_id,)).fetchone()
    if row is None:
        return redirect(url_for("index"))
    if row["status"] == "approved":
        return redirect(url_for("view", etr_id=etr_id))

    if request.method == "POST":
        new_md = request.form.get("content_md", "")
        with db() as conn:
            conn.execute("UPDATE etr SET content_md = ? WHERE id = ?", (new_md, etr_id))
        logger.info("ETR edited id=%s", etr_id)
        return redirect(url_for("view", etr_id=etr_id))

    return render_template("edit.html", etr=row, active="view")


@app.route("/approve/<int:etr_id>", methods=["POST"])
def approve(etr_id):
    """Final node of the diagram: User Specification (approved)."""
    with db() as conn:
        conn.execute(
            "UPDATE etr SET status = 'approved', approved_at = ? WHERE id = ?",
            (datetime.now().strftime("%d/%m/%Y %H:%M"), etr_id),
        )
    logger.info("ETR approved id=%s", etr_id)
    return redirect(url_for("view", etr_id=etr_id))


@app.route("/history")
def history():
    with db() as conn:
        rows = conn.execute("SELECT * FROM etr ORDER BY id DESC").fetchall()
    return render_template("history.html", rows=rows, active="history")


@app.route("/export-pdf/<int:etr_id>")
def export_pdf(etr_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM etr WHERE id = ?", (etr_id,)).fetchone()

    if row is None or row["status"] != "approved":
        return redirect(url_for("index"))

    pdf_style = """
    <style>
    @page { size: A4; margin: 25px; }
    body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #1a1a1a; }
    h1 { font-size: 16pt; color: #15202b; border-bottom: 1px solid #d0d5dd; padding-bottom: 6px; }
    h2 { font-size: 13pt; color: #15202b; margin-top: 18px; }
    h3 { font-size: 11pt; color: #15202b; margin-top: 14px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border: 1px solid #d0d5dd; padding: 5px; font-size: 9pt; }
    th { background: #f2f4f7; }
    </style>
    """
    html_final = f'<html><head><meta charset="utf-8">{pdf_style}</head><body>{to_html(row["content_md"])}</body></html>'

    buffer = io.BytesIO()
    result = pisa.CreatePDF(io.StringIO(html_final), dest=buffer)
    if result.err:
        return "Error while generating the PDF.", 500

    response = make_response(buffer.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=ETR_{etr_id}.pdf"
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)