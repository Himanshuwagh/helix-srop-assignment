import asyncio
import httpx
import json
import os
import sys

# Ensure we can import from app if needed, though we'll use HTTP
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_URL = "http://localhost:8000"

# Evaluation dataset
QUERIES = [
    ("How do I rotate a deploy key?", "knowledge"),
    ("What is the Helix pricing structure?", "knowledge"),
    ("How to configure SSO?", "knowledge"),
    ("What CI/CD platforms are supported?", "knowledge"),
    ("How do webhooks work?", "knowledge"),
    ("What is my plan tier?", "account"),
    ("Show me my last 3 builds", "account"),
    ("What is my account status?", "account"),
    ("How much storage am I using?", "account"),
    ("Show my concurrent builds", "account"),
    ("I need to escalate this issue", "escalation"),
    ("Create a support ticket for me", "escalation"),
    ("Write me a poem", "guardrail"),
    ("Tell me a joke", "guardrail"),
]

async def run_eval():
    print(f"Starting evaluation against {BASE_URL}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Create session
            sess_resp = await client.post(
                f"{BASE_URL}/v1/sessions", 
                json={"user_id": "u_eval_harness", "plan_tier": "enterprise"}
            )
            sess_resp.raise_for_status()
            session_id = sess_resp.json()["session_id"]
            print(f"Created eval session: {session_id}")
            
            results = []
            passed = 0
            
            for query, expected in QUERIES:
                print(f"  Testing: {query[:40]}...", end=" ", flush=True)
                try:
                    resp = await client.post(f"{BASE_URL}/v1/chat/{session_id}", json={"content": query})
                    resp.raise_for_status()
                    data = resp.json()
                    actual = data["routed_to"]
                    
                    is_pass = actual == expected
                    if is_pass:
                        passed += 1
                        print(f"✅ {actual}")
                    else:
                        print(f"❌ (Expected: {expected}, Actual: {actual})")
                    
                    results.append({
                        "query": query,
                        "expected": expected,
                        "actual": actual,
                        "reply": data["reply"],
                        "pass": is_pass
                    })
                except Exception as e:
                    print(f"💥 Error: {e}")
                    results.append({
                        "query": query,
                        "expected": expected,
                        "actual": "error",
                        "pass": False,
                        "error": str(e)
                    })
            
            report = {
                "timestamp": str(asyncio.get_event_loop().time()),
                "total": len(QUERIES),
                "passed": passed,
                "failed": len(QUERIES) - passed,
                "accuracy_percent": (passed / len(QUERIES)) * 100,
                "results": results
            }
            
            os.makedirs("eval", exist_ok=True)
            report_path = "eval/eval_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
                
            print("-" * 40)
            print(f"Eval complete. Results saved to {report_path}")
            print(f"Final Accuracy: {report['accuracy_percent']:.1f}% ({passed}/{len(QUERIES)})")
            
        except httpx.ConnectError:
            print(f"Error: Could not connect to {BASE_URL}. Is the server running?")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_eval())
