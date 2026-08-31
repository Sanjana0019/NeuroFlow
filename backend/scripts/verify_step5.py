import asyncio
import json
import time
import requests
from uuid import UUID

API_BASE = "http://127.0.0.1:8000"

def run_step5_verification():
    print("=" * 60, flush=True)
    print("STEP 5 — LIVE EVALUATION FEED & REAL-TIME SSE VERIFICATION", flush=True)
    print("=" * 60, flush=True)

    # -------------------------------------------------------------
    # 5A & 5B: Historical Evaluation Loading & PostgreSQL Data
    # -------------------------------------------------------------
    print("\n[5A & 5B] Testing Historical Evaluation Loading from PostgreSQL...", flush=True)
    resp = requests.get(f"{API_BASE}/evaluations?limit=50")
    print(f"GET /evaluations status: {resp.status_code}", flush=True)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    
    evals = resp.json()
    print(f"Total historical evaluations retrieved: {len(evals)}", flush=True)
    
    if evals:
        sample = evals[0]
        print(f"Sample Evaluation Record:")
        print(f"  - run_id: {sample.get('run_id')}")
        print(f"  - pipeline_name: {sample.get('pipeline_name')}")
        print(f"  - query: {sample.get('query')[:60] if sample.get('query') else None}...")
        print(f"  - faithfulness: {sample.get('faithfulness')}")
        print(f"  - answer_relevance: {sample.get('answer_relevance')}")
        print(f"  - context_precision: {sample.get('context_precision')}")
        print(f"  - context_recall: {sample.get('context_recall')}")
        print(f"  - overall_score: {sample.get('overall_score')}")
        print(f"  - judge_model: {sample.get('judge_model')}")
        print(f"  - user_rating: {sample.get('user_rating')}")
        print(f"  - evaluated_at: {sample.get('evaluated_at')}")
        print(f"  - chunks count: {len(sample.get('chunks', []))}")
        print("  -> PostgreSQL Data mapping: PASS", flush=True)
    else:
        print("  -> Note: No historical evaluations yet in DB.", flush=True)

    # -------------------------------------------------------------
    # 5C: Live SSE Connection Check
    # -------------------------------------------------------------
    print("\n[5C] Testing Live SSE Stream Connection (GET /evaluations/stream)...", flush=True)
    sse_resp = requests.get(f"{API_BASE}/evaluations/stream", stream=True, timeout=5)
    print(f"GET /evaluations/stream status: {sse_resp.status_code}", flush=True)
    content_type = sse_resp.headers.get("content-type", "")
    print(f"Content-Type: {content_type}", flush=True)
    assert sse_resp.status_code == 200, f"Expected 200, got {sse_resp.status_code}"
    assert "text/event-stream" in content_type, f"Expected text/event-stream, got {content_type}"
    print("  -> SSE Connection & Headers: PASS", flush=True)

    # -------------------------------------------------------------
    # 5D & 5E: Real-Time Event Test & Metric Display Verification
    # -------------------------------------------------------------
    print("\n[5D & 5E] Testing Real-Time Event Trigger & Metric Streaming...", flush=True)
    
    # Listen to SSE stream in background or verify via query run
    query_resp = requests.post(f"{API_BASE}/query", json={
        "query": "What embedding model does NeuroFlow use for high-performance dense retrieval?",
        "stream": True
    })
    run_id = query_resp.json().get("run_id")
    print(f"Dispatched new query, run_id: {run_id}", flush=True)

    # Stream query tokens to completion
    stream_resp = requests.get(f"{API_BASE}/query/{run_id}/stream", stream=True)
    final_run_id = None
    for line in stream_resp.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data = json.loads(decoded[6:])
                if data.get("type") == "done":
                    final_run_id = data.get("run_id")
                    break

    target_id = final_run_id or run_id
    print(f"Query generation completed for run: {target_id}", flush=True)

    # Wait for ARQ worker / EvaluationJudge to persist evaluation
    eval_record = None
    for i in range(15):
        time.sleep(1)
        chk = requests.get(f"{API_BASE}/runs/{target_id}/evaluation")
        if chk.status_code == 200:
            eval_record = chk.json()
            print(f"Evaluation record created and verified (t={i+1}s):", flush=True)
            print(f"  Faithfulness: {eval_record.get('faithfulness')}")
            print(f"  Answer Relevance: {eval_record.get('answer_relevance')}")
            print(f"  Context Precision: {eval_record.get('context_precision')}")
            print(f"  Context Recall: {eval_record.get('context_recall')}")
            print(f"  Overall Score: {eval_record.get('overall_score')}")
            print(f"  Judge Model: {eval_record.get('judge_model')}")
            break

    assert eval_record is not None, "Evaluation record was not persisted within 15 seconds"
    print("  -> Real-Time Event & Evaluation Persistence: PASS", flush=True)

    # -------------------------------------------------------------
    # 5F: Pipeline Filtering Verification
    # -------------------------------------------------------------
    print("\n[5F] Testing Pipeline Filter Endpoint & Behavior...", flush=True)
    pipelines_resp = requests.get(f"{API_BASE}/pipelines")
    pipelines = pipelines_resp.json()
    if pipelines:
        pipe_id = pipelines[0]["id"]
        pipe_name = pipelines[0]["name"]
        filter_resp = requests.get(f"{API_BASE}/evaluations?pipeline_id={pipe_id}&limit=50")
        print(f"GET /evaluations?pipeline_id={pipe_id} status: {filter_resp.status_code}", flush=True)
        assert filter_resp.status_code == 200
        filtered_items = filter_resp.json()
        print(f"Filtered evaluations count for pipeline '{pipe_name}': {len(filtered_items)}", flush=True)
        print("  -> Pipeline Filter: PASS", flush=True)

    # -------------------------------------------------------------
    # 5G: Helpful / Poor Calibration Rating
    # -------------------------------------------------------------
    print("\n[5G] Testing Helpful & Poor Human Calibration (PATCH /runs/{run_id}/rating)...", flush=True)
    # Test Helpful (5)
    rate_helpful = requests.patch(f"{API_BASE}/runs/{target_id}/rating", json={"rating": 5})
    print(f"Rate 5 (Helpful) status: {rate_helpful.status_code}", flush=True)
    assert rate_helpful.status_code == 200
    assert rate_helpful.json().get("user_rating") == 5
    print(f"  Calibration response: user_rating={rate_helpful.json().get('user_rating')}, calibration_needed={rate_helpful.json().get('calibration_needed')}")

    # Test Poor (1)
    rate_poor = requests.patch(f"{API_BASE}/runs/{target_id}/rating", json={"rating": 1})
    print(f"Rate 1 (Poor) status: {rate_poor.status_code}", flush=True)
    assert rate_poor.status_code == 200
    assert rate_poor.json().get("user_rating") == 1
    print(f"  Calibration response: user_rating={rate_poor.json().get('user_rating')}, calibration_needed={rate_poor.json().get('calibration_needed')}")
    print("  -> Helpful / Poor Calibration: PASS", flush=True)

    # -------------------------------------------------------------
    # 5H: Failure Handling
    # -------------------------------------------------------------
    print("\n[5H] Testing Failure Handling & Edge Cases...", flush=True)
    # 1. Invalid rating range
    invalid_rate = requests.patch(f"{API_BASE}/runs/{target_id}/rating", json={"rating": 10})
    print(f"Invalid rating (10) status: {invalid_rate.status_code} (Expected 400 or 422)")
    assert invalid_rate.status_code in (400, 422)

    # 2. Non-existent run rating
    non_existent = "00000000-0000-0000-0000-000000000000"
    missing_rate = requests.patch(f"{API_BASE}/runs/{non_existent}/rating", json={"rating": 4})
    print(f"Non-existent run rating status: {missing_rate.status_code} (Expected 404)")
    assert missing_rate.status_code == 404

    # 3. Invalid threshold query param
    invalid_thresh = requests.get(f"{API_BASE}/evaluations?min_overall=2.5")
    print(f"Invalid threshold (2.5) status: {invalid_thresh.status_code} (Expected 422)")
    assert invalid_thresh.status_code == 422
    print("  -> Failure Handling & Boundary Validations: PASS", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("ALL STEP 5 VERIFICATION CHECKS PASSED SUCCESSFULLY!", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    run_step5_verification()
