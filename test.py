from app.query_orchestrator import handle_query_with_rag

# TEST_CASES = [
#     # -------------------------
#     # CORE METRIC + POLICY
#     # -------------------------
#     "show me CM1 last month",
#     "show me GM last 7 days",
#     "show me GMV last 30 days",

#     # -------------------------
#     # SCOPE ENFORCEMENT
#     # -------------------------
#     "show me GMV for gold lobby last 7 days",
#     "show me GMV for cash lobby last 7 days",
#     "show me GMV for all lobbies last month",

#     # -------------------------
#     # ZSH / GLOSSARY GROUNDING
#     # -------------------------
#     "show me GMV for free lobby last 14 days",
#     "show me number of users in zsh lobby last month",

#     # -------------------------
#     # PER USER METRICS
#     # -------------------------
#     "calculate GMV per user by game last 14 days",
#     "show me CM1 per user last 30 days",

#     # -------------------------
#     # DIMENSION HANDLING
#     # -------------------------
#     "show me GMV by game last 7 days",
#     "show me GMV by app platform last month",

#     # -------------------------
#     # TIME HANDLING
#     # -------------------------
#     "show me GMV yesterday",
#     "show me GMV week over week last 4 weeks",

#     # -------------------------
#     # HALLUCINATION GUARDS
#     # -------------------------
#     "show me event_date wise revenue last 7 days",
#     "show me profit last month",

#     # -------------------------
#     # INVALID / FALLBACK
#     # -------------------------
#     "hi",
#     "show me everything",
#     "give me raw data",
# ]

TEST_CASES = [
    # -------------------------
    # CORE METRIC + POLICY
    "what is the w/d ratio for US in last 2 months, aggregated weekly",   
]


def run_tests():
    for q in TEST_CASES:
        ok, sql, err = handle_query_with_rag(q)
        print("="*100)
        print("QUERY:", q)
        print("-"*100)
        if ok:
            print("✅ SQL GENERATED\n")
            print(sql)
        else:
            print("❌ FALLBACK / ERROR\n")
            print(err)

if __name__ == "__main__":
    run_tests()
