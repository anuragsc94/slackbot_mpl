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

