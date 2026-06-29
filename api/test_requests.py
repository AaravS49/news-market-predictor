"""Hit every API endpoint and print results. Run with server live on localhost:8000."""
import json

import requests

BASE = "http://localhost:8000"
DIVIDER = "-" * 60


def show(num: int, desc: str, response: requests.Response, expect: int = 200) -> requests.Response:
    ok = "PASS" if response.status_code == expect else "FAIL"
    print(f"\n[Test {num}] {desc}  →  {ok} (HTTP {response.status_code})")
    print(DIVIDER)
    try:
        body = response.json()
        text = json.dumps(body, indent=2, default=str)
        # Truncate long bodies for readability
        print(text[:800] + (" ..." if len(text) > 800 else ""))
    except Exception:
        print(response.text[:400])
    return response


# ── Test 1: health check ─────────────────────────────────────────────────────
r1 = show(1, "GET /health", requests.get(f"{BASE}/health"))

# ── Test 2: list tickers ─────────────────────────────────────────────────────
r2 = show(2, "GET /tickers", requests.get(f"{BASE}/tickers"))

# ── Test 3: predict on a real AAPL headline ──────────────────────────────────
AAPL_PAYLOAD = {
    "ticker": "AAPL",
    "headline": "Apple reports record quarterly revenue driven by iPhone 15 sales",
    "body": (
        "Apple Inc reported its highest ever quarterly revenue on Thursday, "
        "beating analyst expectations across all product categories."
    ),
    "published_at": "2024-11-01T09:00:00",
    "url": "https://test.example.com/apple-q4-2024-earnings",
}
r3 = show(
    3, "POST /predict  (AAPL, with URL → persisted)",
    requests.post(f"{BASE}/predict", json=AAPL_PAYLOAD),
)

# ── Test 4: invalid ticker ───────────────────────────────────────────────────
r4 = show(
    4,
    "POST /predict  (invalid ticker → expect 400)",
    requests.post(f"{BASE}/predict", json={**AAPL_PAYLOAD, "ticker": "FAKE"}),
    expect=400,
)  # noqa: E501

# ── Test 5: prediction history for AAPL ─────────────────────────────────────
r5 = show(5, "GET /history/AAPL", requests.get(f"{BASE}/history/AAPL"))

# ── Test 6: submit feedback on the prediction from Test 3 ───────────────────
prediction_id = r3.json().get("prediction_id") if r3.status_code == 200 else None
if prediction_id:
    r6 = show(
        6,
        f"POST /feedback/{prediction_id}  (actual_label=1)",
        requests.post(f"{BASE}/feedback/{prediction_id}", json={"actual_label": 1}),
    )
else:
    print(
        "\n[Test 6] SKIPPED — Test 3 did not return a prediction_id "
        "(article may already exist in DB; check /history/AAPL)."
    )

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("All tests complete.")
