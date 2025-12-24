
import re
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import exp


ESSENTIAL_FILTERS = ["country_code", "lobby_type"]


def _expr_to_sql(e: exp.Expression) -> str:
    # pretty print a node back to SQL
    return e.sql(dialect="bigquery")


def _collect_where_expressions(ast: exp.Expression) -> List[exp.Expression]:
    """
    Collect WHERE expressions from:
    - main query
    - nested subqueries
    - CTE bodies
    """
    wheres: List[exp.Expression] = []

    # Every SELECT in the whole tree
    for select in ast.find_all(exp.Select):
        w = select.args.get("where")
        if w and isinstance(w, exp.Where) and w.this is not None:
            wheres.append(w.this)

    return wheres


def _flatten_and_conditions(condition: exp.Expression) -> List[exp.Expression]:
    """
    Turn (A AND (B AND C)) into [A, B, C]
    Keep OR blocks intact (we still show them, but don't split).
    """
    out: List[exp.Expression] = []

    def walk(node: exp.Expression):
        if isinstance(node, exp.And):
            walk(node.left)
            walk(node.right)
        else:
            out.append(node)

    walk(condition)
    return out


def _find_limit(ast: exp.Expression) -> Optional[int]:
    limit = ast.find(exp.Limit)
    if not limit:
        return None
    # BigQuery: LIMIT <number>
    n = limit.args.get("expression")
    if isinstance(n, exp.Literal) and n.is_int:
        return int(n.this)
    return None


def _find_group_by(ast: exp.Expression) -> List[str]:
    gb = ast.find(exp.Group)
    if not gb:
        return []
    exprs = gb.expressions or []
    return [_expr_to_sql(e) for e in exprs]


def _find_select_aliases(ast: exp.Expression) -> List[str]:
    """
    Return aliases from SELECT list: e.g. COUNT(*) AS total_gps -> total_gps
    """
    aliases: List[str] = []
    select = ast.find(exp.Select)
    if not select:
        return []

    for item in select.expressions:
        # SELECT a AS alias  OR  SELECT func(...) alias
        a = None
        if isinstance(item, exp.Alias):
            a = item.alias
        else:
            a = item.alias  # some nodes also carry alias
        if a:
            aliases.append(a)

    # unique preserve order
    seen = set()
    out = []
    for x in aliases:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _collect_columns_in_condition(cond: exp.Expression) -> List[str]:
    cols = []
    for c in cond.find_all(exp.Column):
        # c.name gives column name without table qualifier
        cols.append(c.name)
    return list(dict.fromkeys(cols))


def summarize_sql_for_user_sqlglot(sql: str, user_question: Optional[str] = None) -> Dict[str, Any]:
    try:
        ast = sqlglot.parse_one(sql, read="bigquery")
    except Exception as e:
        # fallback: if parsing fails, you can still show a minimal message
        return {
            "parse_error": str(e),
            "filters": [],
            "group_by": [],
            "metrics": [],
            "limit": None,
            "missing_essential_filters": ESSENTIAL_FILTERS,
        }

    where_exprs = _collect_where_expressions(ast)

    # Merge all WHERE expressions into a single list of readable conditions
    conditions_sql: List[str] = []
    columns_seen: set[str] = set()

    for w in where_exprs:
        for cond in _flatten_and_conditions(w):
            conditions_sql.append(_expr_to_sql(cond))
            for col in _collect_columns_in_condition(cond):
                columns_seen.add(col.lower())

    group_by = _find_group_by(ast)
    limit = _find_limit(ast)
    metrics = _find_select_aliases(ast)

    missing = [c for c in ESSENTIAL_FILTERS if c.lower() not in columns_seen]

    # your special rule: if user asked about GPs/transactions, ensure transaction_type filter exists
    needs_gp_filter = False
    if user_question:
        needs_gp_filter = bool(re.search(r"\b(gp|gameplay|gameplays|transactions)\b", user_question, flags=re.I))
    if needs_gp_filter and ("transaction_type" not in columns_seen):
        missing.append("transaction_type (Gameplay- Battles)")

    return {
        "filters": conditions_sql,
        "group_by": group_by,
        "metrics": metrics,
        "limit": limit,
        "missing_essential_filters": missing,
    }


def format_sql_summary_for_slack(summary: Dict[str, Any]) -> str:
    if summary.get("parse_error"):
        return (
            "*What I ran (plain English)*\n"
            "• I could not reliably parse the SQL to explain it.\n"
            f"• Parser error: `{summary['parse_error']}`\n"
            "\n*Tip*\n"
            "• I can still run the query, but explanations + missing-filter checks may be incomplete."
        )

    lines = ["*What I ran (plain English)*"]

    if summary["filters"]:
        lines.append("• *Filters applied:*")
        for c in summary["filters"]:
            lines.append(f"  - `{c}`")
    else:
        lines.append("• *Filters applied:* (none detected)")

    if summary["group_by"]:
        lines.append(f"• *Grouped by:* {', '.join([f'`{g}`' for g in summary['group_by']])}")

    if summary["metrics"]:
        lines.append(f"• *Outputs/metrics:* {', '.join([f'`{m}`' for m in summary['metrics']])}")

    if summary["limit"] is not None:
        lines.append(f"• *Row limit:* `{summary['limit']}`")

    if summary["missing_essential_filters"]:
        lines.append("\n*Missing filters you may want to add to your prompt*")
        for m in summary["missing_essential_filters"]:
            lines.append(f"• `{m}`")

    return "\n".join(lines)
