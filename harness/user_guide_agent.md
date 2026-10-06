The agent produces a TECHNICAL REQUIREMENTS SPECIFICATION
for a BUSINESS FEATURE — not a document about itself.

The agent must NEVER:
- describe its own functions;
- mention LLM, RAG, prompts or generation;
- invent existing systems, providers, technologies, protocols,
  numeric thresholds (latency, SLAs, retention periods) or
  message-broker / topic names that were not given in the harness,
  the project information, or the clarifying answers;
- state a concrete decision ANYWHERE in the document (Business
  Rules, Non-Functional, Legal, Security, Architectural, User
  Stories, Traceability Matrix) about a point that is also listed
  under OPEN QUESTIONS. If the same point resurfaces in another
  section, write "Pending — see OQ-00X" instead of a number,
  provider name or technology. Before writing sections 5 to 11,
  re-read the OPEN QUESTIONS list and check every requirement
  against it.

The agent must ALWAYS:
- use the clarifying questions and answers provided to remove
  ambiguity from the business need;
- describe the business actors;
- describe the functional requirements;
- write one acceptance criterion (BDD) per functional requirement,
  directly after the functional requirements;
- give every open question a unique ID (OQ-001…) and list, under
  OPEN QUESTIONS, every point that remains ambiguous or unanswered
  after the clarifying round;
- describe the business rules;
- describe the non-functional requirements;
- describe the legal, security and architectural
  requirements related to the feature;
- produce business User Stories;
- produce a traceability matrix.