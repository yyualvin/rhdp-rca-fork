---
name: rca-knowledge-base
description: Query the historical RCA knowledge base to answer questions about past root cause analyses. Use when users ask about patterns in past failures, common root causes for specific services or platforms, whether similar issues have been seen before, or want to search historical RCA data.
allowed-tools:
  - Bash
---

# RCA Knowledge Base

Search and synthesize answers from historical root cause analyses stored in a mem0 memory layer. This skill turns raw similarity search results into clear, evidence-backed answers.

## When to Use

- User asks about patterns in past failures ("What usually causes sandbox failures on AWS?")
- User wants to know if a similar issue has been seen before ("Have we seen DNS resolution issues?")
- User asks about common root causes for a service, platform, or catalog item
- User wants historical context before investigating a new failure

## Prerequisites

This skill requires a mem0-compatible pgvector backend. The following environment variables must be set in `.claude/settings.json` under `env`:

- `PGVECTOR_HOST`
- `PGVECTOR_DB_NAME`
- `PGVECTOR_DB_USER`
- `PGVECTOR_DB_PASSWORD`
- `ANTHROPIC_VERTEX_PROJECT_ID` and `CLOUD_ML_REGION` (for mem0's LLM and embedder via Vertex AI)

RCA analyses must have been previously stored using `cli.py embed` (triggered automatically after each RCA) or via the batch pipeline.

## Instructions

### Step 1: Prepare the environment [Bash]

Ensure the root-cause-analysis virtual environment exists:

```bash
cd ../root-cause-analysis && (test -d .venv || (python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt))
```

If the venv creation or dependency install fails, stop and report the error to the user.

### Step 2: Formulate the search query [Claude]

Rephrase the user's natural language question into failure-domain terms that align with how RCA memories are structured. The memories contain consolidated facts extracted from RCA analyses including:

- Root cause category and summary
- Catalog items, platforms, and cloud providers
- Actions and environment types
- Contributing factors and failed tasks
- Resolution recommendations

**Query formulation guidelines:**

- Extract key technical terms: service names, error messages, infrastructure components, platform names
- Use failure-domain language rather than conversational phrasing
- Example: "Why do sandbox deploys keep failing?" → "sandbox deployment failure"
- Example: "Have we seen issues with AWS credentials?" → "AWS credential authentication failure"

**Optional filters** — if the user's question implies a specific scope, use these CLI flags:

- `--category <cat>` — filter by root cause category (e.g., `configuration`, `infrastructure`, `credential`)
- `--catalog-item <item>` — filter by a specific workload or catalog item name

### Step 3: Run the similarity search [Bash]

```bash
cd ../root-cause-analysis && .venv/bin/python scripts/cli.py similar --text "<formulated_query>" --limit 5
```

Adjust `--limit` up to 10 if the user asks for more results. Add `--category` or `--catalog-item` flags if determined in Step 2.

If the command fails with a configuration error, tell the user to set the required environment variables listed in Prerequisites and stop.

If the command returns an empty JSON array `[]`, tell the user no similar past analyses were found. Suggest they:
- Broaden the search terms
- Check that RCA analyses have been stored (run `cli.py embed` after completing analyses)

### Step 4: Synthesize the results [Claude]

Transform the JSON results into a clear answer. The JSON array contains objects with these fields:

| Field | Description |
|-------|-------------|
| `id` | Unique memory identifier |
| `memory` | The consolidated memory text (may combine facts from multiple RCAs) |
| `root_cause_category` | Category of the root cause |
| `confidence` | Confidence level (high/medium/low) |
| `analyzed_at` | When the analysis was performed |
| `platform` | Affected platform |
| `cloud_provider` | Cloud provider involved |
| `action` | Action that was being performed |
| `score` | Similarity score — higher means more similar |

**Synthesis guidelines:**

1. **Answer the question directly** — lead with a one or two sentence answer
2. **Highlight patterns** — mem0 automatically consolidates similar memories, so each result may represent multiple past incidents
3. **Cite evidence** — reference confidence levels, platforms, and categories
4. **Indicate match quality** — score > 0.7 is a strong match, 0.4–0.7 is moderate, < 0.4 is weak. Flag weak matches explicitly
5. **Note limitations** — if all matches are weak or few results returned, say so

**Output format:**

```
## Answer

<Direct answer to the user's question in 1-2 sentences>

## Matching Patterns

### <root_cause_category>: <memory summary>

<memory text>

- **Platform:** <platform>
- **Cloud provider:** <cloud_provider>
- **Confidence:** <confidence>, **Similarity:** <strong/moderate/weak>

<Repeat for each matching pattern>

## Recommendations

<Actionable suggestions based on the patterns found>
```
