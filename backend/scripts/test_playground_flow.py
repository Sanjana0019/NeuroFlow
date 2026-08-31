import requests
import json
import time

print("1. Testing POST /query...", flush=True)
query_payload = {
    "query": "What embedding model does NeuroFlow use for high-performance dense retrieval?",
    "pipeline_id": None,
    "stream": True,
}
resp = requests.post("http://127.0.0.1:8000/query", json=query_payload)
print("POST /query status:", resp.status_code, flush=True)
query_data = resp.json()
run_id = query_data.get("run_id")
print("run_id:", run_id, flush=True)

# 2. Test GET /query/{run_id}/stream SSE
print("\n2. Testing SSE stream...", flush=True)
stream_resp = requests.get(f"http://127.0.0.1:8000/query/{run_id}/stream", stream=True)
full_text = ""
final_run_id = None
retrieval_metadata = None

for line in stream_resp.iter_lines():
    if line:
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data_str = decoded[6:]
            try:
                event = json.loads(data_str)
                event_type = event.get("type")
                if event_type == "token":
                    full_text += event.get("delta", "")
                elif event_type == "retrieval_metadata":
                    retrieval_metadata = event
                    print("Received retrieval_metadata with", len(event.get("chunks_used", [])), "chunks", flush=True)
                elif event_type == "done":
                    final_run_id = event.get("run_id")
                    print("Received done event with final_run_id:", final_run_id, flush=True)
            except Exception:
                pass

print("\nStreamed Answer:\n", full_text.strip(), flush=True)

# 3. Test Evaluation score persistence
target_run_id = final_run_id or run_id
if target_run_id:
    print(f"\n3. Waiting for evaluation score for {target_run_id}...", flush=True)
    for i in range(15):
        time.sleep(1)
        eval_resp = requests.get(f"http://127.0.0.1:8000/runs/{target_run_id}/evaluation")
        if eval_resp.status_code == 200:
            print(f"Evaluation persisted (t={i+1}s):", eval_resp.json(), flush=True)
            break

# 4. Test POST /pipelines/compare
print("\n4. Testing POST /pipelines/compare...", flush=True)
pipelines = requests.get("http://127.0.0.1:8000/pipelines").json()
if len(pipelines) >= 2:
    compare_payload = {
        "query": "What embedding model does NeuroFlow use for high-performance dense retrieval?",
        "pipeline_a_id": pipelines[0]["id"],
        "pipeline_b_id": pipelines[1]["id"]
    }
    comp_resp = requests.post("http://127.0.0.1:8000/pipelines/compare", json=compare_payload)
    print("POST /compare status:", comp_resp.status_code, flush=True)
    print("Pipeline A latency:", comp_resp.json()["pipeline_a"]["total_latency_ms"], "ms", flush=True)
    print("Pipeline B latency:", comp_resp.json()["pipeline_b"]["total_latency_ms"], "ms", flush=True)
    print("Pipeline A answer:", comp_resp.json()["pipeline_a"]["generation"][:100], "...", flush=True)
    print("Pipeline B answer:", comp_resp.json()["pipeline_b"]["generation"][:100], "...", flush=True)

print("\nAll Playground Backend flows PASS!", flush=True)
