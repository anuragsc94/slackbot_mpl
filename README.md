# SLACKBOT_MPL — RAG-Based Text-to-SQL Analytics Bot

A production-grade Slack bot that converts natural-language analytics questions into **policy-safe, BigQuery-compatible SQL** using a **YAML-driven RAG architecture**.

This system enforces **business semantics, metric correctness, scope safety, and table-level routing** while remaining extensible to multi-table analytics.

---

## 🚀 Key Capabilities

- **Natural Language → BigQuery SQL**
- **Strict policy enforcement** via a binding rulebook (PolicyDoc)
- **Metric-driven SQL generation** (no hallucinated metrics)
- **Multi-table architecture** (gameplay + wallet transactions)
- **Scope-safe filtering** (only allowed scopes applied)
- **Wallet analytics support** (deposits, withdrawals, ratios)
- **Deterministic SQL output** (no explanations, SQL only)

---

## 🧠 Architecture Overview

User Question (Slack)
↓
RAG Retriever (Chroma + Gemini embeddings)
↓
PolicyDoc (MASTER rulebook)
MetricCatalog + TableCards + ColumnCards
ScopeCards + Business Glossary
↓
SQL Generator (Gemini)
↓
BigQuery SQL (validated & safe)



---

## 🧩 YAML-Driven RAG Design

This system is entirely driven by **structured YAML documents**:

### Core Documents
- **PolicyDoc** – Binding SQL and business rules (highest priority)
- **MetricCatalog** – Defines *what can be calculated*
- **TableCard** – Declares *where metrics live*
- **ColumnCard** – Column-level schema grounding
- **ScopeCard** – Allowed and forbidden filters
- **Business Glossary** – Human → schema mapping

> 🔒 If a metric, column, or scope is not defined in YAML, it **cannot** appear in SQL.

---

## 📊 Supported Tables (Current)

| Table | Purpose |
|------|--------|
| `dw_gold.fact_revenue_metrics` | Gameplay-level revenue & margins |
| `dw_silver.transactions_fat`  | Wallet transactions (deposits, withdrawals) |

> Joins are intentionally **not supported yet** to maintain determinism and safety.

---

## 📈 Supported Metric Categories

- Gameplay activity (GPs, Users)
- Revenue (GMV, Winnings, Margin)
- Platform economics (GM, CM1)
- Wallet utilisation (BC, DC, WC)
- Wallet flows (Deposits, Withdrawals)
- Ratios (RTP, Win Rate, W/D)
- Per-user metrics (PU metrics)
- HOF metrics

---

## 🔐 Policy Enforcement

The bot **always**:
- Injects the PolicyDoc MASTER chunk
- Applies only scopes referenced by metrics
- Removes unreferenced scopes
- Prevents `SELECT *`
- Uses `SAFE_DIVIDE` for ratios
- Uses `dt` for date filtering
- Outputs **SQL only** (no markdown, no text)

---

## 🛠️ Setup Instructions

### 1️⃣ Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

---

## 🚀 Deployment on Dataproc VM (Multi Slack Bots → Multi BQ Projects)

You can run **multiple Slack apps (bots)** from the **same VM process**. Each bot is tied to a specific BigQuery project via a config file mapping.

### Multi-bot config

1) Copy `config/bots.example.yaml` to a VM-local path (don’t commit secrets):
- Example: `/etc/slackbot_mpl/bots.yaml`

2) Export Slack tokens as env vars (recommended) and reference them in `bots.yaml`:
- Each bot needs:
  - `xoxb-...` **bot token**
  - `xapp-...` **app-level token** (Socket Mode)

3) Set `bq_project_id` per bot in `bots.yaml`. Your YAML docs and SQL can stay the same (dataset.table).

### Required environment variables

- **LLM / embeddings**
  - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
  - optional: `GEMINI_MODEL`

- **Multi-bot switch**
  - `BOT_CONFIG_PATH=/etc/slackbot_mpl/bots.yaml`

- **GCP auth**
  - Prefer VM-attached service account / workload identity (recommended)
  - Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json`

### Example `systemd` unit (Dataproc VM)

Create `/etc/systemd/system/slackbot-mpl.service`:

```ini
[Unit]
Description=Slack RAG Text-to-SQL Bot (multi-bot)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/slackbot_mpl
Environment=BOT_CONFIG_PATH=/etc/slackbot_mpl/bots.yaml
Environment=LOG_LEVEL=INFO

# Slack tokens referenced by bots.yaml
Environment=SLACK_BOT_TOKEN_MPL_PROD=...
Environment=SLACK_APP_TOKEN_MPL_PROD=...
Environment=SLACK_BOT_TOKEN_MPL_STAGING=...
Environment=SLACK_APP_TOKEN_MPL_STAGING=...

# LLM
Environment=GEMINI_API_KEY=...
Environment=GEMINI_MODEL=gemini-2.5-flash

# If not using VM identity, uncomment:
# Environment=GOOGLE_APPLICATION_CREDENTIALS=/etc/slackbot_mpl/sa.json

ExecStart=/opt/slackbot_mpl/.venv/bin/python /opt/slackbot_mpl/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now slackbot-mpl
sudo systemctl status slackbot-mpl
```

