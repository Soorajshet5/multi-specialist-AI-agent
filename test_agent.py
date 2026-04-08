"""
test_agent.py
Quick smoke tests — run these while the FastAPI server is up.

Usage:
    python test_agent.py
"""

import requests

BASE = "http://localhost:8000"


def post(task: str, user_id: str) -> dict:
    r = requests.post(f"{BASE}/run-agent", json={"task": task, "user_id": user_id}, timeout=60)
    r.raise_for_status()
    return r.json()


def test_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    print("✓ Health check passed")


def test_general():
    result = post("Explain what a Python decorator is in one sentence", "soo_general")
    assert result["status"] == "success"
    assert result["result"]
    print(f"✓ General task | {result['result'][:80]}...")


def test_stock():
    result = post("Check SBI stock trend", "soo_stock")
    print(f"  Stock result: {result}")
    # May return fallback if Alpha Vantage key is invalid — that's expected
    assert result["status"] in ("success", "fallback")
    print(f"✓ Stock task | status={result['status']}")


def test_compliance():
    result = post("What is the tax treatment of LTCG on equity mutual funds in India?", "soo_tax")
    assert result["status"] in ("success", "fallback")
    print(f"✓ Compliance task | {result['result'][:80] if result['result'] else 'No output'}...")


if __name__ == "__main__":
    test_health()
    test_general()
    test_stock()
    test_compliance()
    print("\nAll smoke tests complete.")