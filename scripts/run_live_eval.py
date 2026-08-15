"""Run all canonical cases against real HTTP APIs; never substitute mock answers.

Creates isolated fixture stores and launches one uvicorn process per configuration.
Semantic scores require a separately reviewed, response-hash-bound annotation file.
The fixture text, IDs, cases, and scoring rubric are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from ai_equity_research_copilot_backend.embeddings import HashingEmbedder
from ai_equity_research_copilot_backend.schemas import Company, Document, DocumentChunk, ChatResponse, CompareResponse

SOURCES = ["evals/finance_qa_v1.jsonl", "evals/scoring_rubric.json",
           "tests/fixtures/companies.json", "tests/fixtures/documents.json",
           "tests/fixtures/chunks/retrieval_chunks.json"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def http(url, payload=None):
    req = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=240) as response:
        return json.load(response)


def fixture_store(directory):
    fixture = ROOT / "tests/fixtures"
    embedder = HashingEmbedder()
    state = {"companies": [], "documents": [], "chunks": [],
             "conversations": [], "messages": [], "citations": []}
    for name, model in [("companies", Company), ("documents", Document)]:
        state[name] = [model.model_validate(x).model_dump(mode="json")
                       for x in json.loads((fixture / f"{name}.json").read_text())]
    for row in json.loads((fixture / "chunks/retrieval_chunks.json").read_text()):
        row["embedding"] = embedder.embed(row["text"])
        state["chunks"].append(DocumentChunk.model_validate(row).model_dump(mode="json"))
    destination = directory / "storage/state/store.json"
    destination.parent.mkdir(parents=True)
    write(destination, state)
    return {k: len(state[k]) for k in ("companies", "documents", "chunks")}


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(x) for x in (ROOT / SOURCES[0]).read_text().splitlines() if x.strip()]
    assert len(cases) == 35 and len({x["id"] for x in cases}) == 35
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "runner_sha256": sha(__file__), "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
        "python": sys.version, "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "ollama_version": http(args.ollama + "/api/version"),
        "model_name": args.model,
        "model_details": http(args.ollama + "/api/show", {"model": args.model}),
        "model_tags": http(args.ollama + "/api/tags"),
        "corpus": "Unmodified synthetic canonical fixture chunks, not SEC filings; hashing embeddings regenerated.",
        "routing": "comparison -> /research/compare (deterministic in both modes); all other cases -> /research/chat",
        "filters": "None; top_k=8; ordinary application source-period policy applies. No gold documents or answers sent to model.",
        "repetitions": 1, "warmup": "No benchmark warmup; model may already be resident from preflight.",
        "latency_definition": "Sequential end-to-end localhost HTTP wall time, including persistence; excludes server startup.",
    }
    write(out / "metadata.json", metadata)
    for provider in ("ollama", "local"):
        # Stores are kept in the run directory to make the evaluated evidence inspectable.
        directory = out / f"store-{provider}"
        counts = fixture_store(directory)
        env = {**os.environ, "AIERC_DATA_DIR": str(directory), "AIERC_LLM_PROVIDER": provider,
               "AIERC_OLLAMA_MODEL": args.model, "AIERC_OLLAMA_BASE_URL": args.ollama,
               "AIERC_OLLAMA_TIMEOUT_SECONDS": "180", "AIERC_DEMO_MODE": "false"}
        url = f"http://127.0.0.1:{args.port}"
        with (out / f"server-{provider}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn",
                "ai_equity_research_copilot_backend.main:create_app", "--factory",
                "--host", "127.0.0.1", "--port", str(args.port)], cwd=ROOT / "backend", env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError("Eval server exited; inspect its log")
                    try:
                        health = http(url + "/health")
                        assert all(health[k] == v for k, v in counts.items())
                        break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise RuntimeError("Eval server did not become ready")
                rows = []
                for case in cases:
                    endpoint = "/research/compare" if case["category"] == "comparison" else "/research/chat"
                    payload = {"company_ids": case["company_ids"], "question": case["question"], "top_k": 8}
                    started = time.perf_counter()
                    response = http(url + endpoint, payload)
                    elapsed = (time.perf_counter() - started) * 1000
                    schema = CompareResponse if endpoint.endswith("compare") else ChatResponse
                    schema.model_validate(response)
                    limitations = " ".join(response.get("limitations", []))
                    fallback = "deterministic cited synthesis used instead" in limitations
                    unavailable = "configured but unavailable" in response.get("answer", "")
                    accepted = response["usage"]["provider"] == "ollama" and not unavailable and not fallback
                    row = {"id": case["id"], "category": case["category"], "configuration": provider,
                           "endpoint": endpoint, "request": payload, "response": response,
                           "latency_ms": round(elapsed, 3), "format_compliance": 1,
                           "accepted_model_output": accepted, "validation_fallback": fallback,
                           "model_unavailable": unavailable}
                    rows.append(row)
                    write(out / f"{provider}.json", rows)
                    print(f"{provider} {case['id']} {elapsed:.0f}ms accepted={accepted} fallback={fallback}", flush=True)
            finally:
                process.terminate()
                process.wait(timeout=15)
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(out / "metadata.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--port", type=int, default=8011)
    run(parser.parse_args())
