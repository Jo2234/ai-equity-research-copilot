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
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from ai_equity_research_copilot_backend.embeddings import HashingEmbedder
from ai_equity_research_copilot_backend.schemas import (
    ChatResponse,
    Company,
    CompareResponse,
    Document,
    DocumentChunk,
)

SOURCES = ["evals/finance_qa_v1.jsonl", "evals/scoring_rubric.json",
           "tests/fixtures/companies.json", "tests/fixtures/documents.json",
           "tests/fixtures/chunks/retrieval_chunks.json"]
PROVIDER_CHOICES = {"local": ("local",), "ollama": ("ollama",), "both": ("ollama", "local")}


def selected_providers(value):
    try:
        return PROVIDER_CHOICES[value]
    except (KeyError, TypeError):
        raise ValueError("providers must be local, ollama, or both") from None


def _command_text(command):
    value = subprocess.check_output(command, text=True, timeout=5).strip()
    if not value:
        raise ValueError("command returned an empty value")
    return value


def hardware_metadata():
    """Optional platform observations; failures never invalidate a capture."""
    values = {"cpu": None, "memory_bytes": None}
    details = {}
    system = platform.system()
    for field in values:
        source = "unavailable"
        try:
            if system == "Darwin":
                key = "machdep.cpu.brand_string" if field == "cpu" else "hw.memsize"
                source = f"sysctl {key}"
                value = _command_text(["sysctl", "-n", key])
            elif system == "Linux":
                if field == "cpu":
                    # Not platform.processor(): `uname -p` yields "unknown" or a bare ISA on Linux.
                    source = "/proc/cpuinfo"
                    value = None
                    for line in Path("/proc/cpuinfo").read_text().splitlines():
                        name, _, text = line.partition(":")
                        if name.strip() in ("model name", "Hardware", "Processor") and text.strip():
                            value = text.strip()
                            break
                else:
                    source = "os.sysconf physical pages * page size"
                    pages = int(os.sysconf("SC_PHYS_PAGES"))
                    page_size = int(os.sysconf("SC_PAGE_SIZE"))
                    if pages <= 0 or page_size <= 0:
                        raise ValueError("physical pages and page size must be positive")
                    value = pages * page_size
            elif system == "Windows":
                source = "PowerShell Get-CimInstance"
                expression = ("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"
                              if field == "cpu" else "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
                value = _command_text(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", expression])
            else:
                raise ValueError(f"unsupported platform: {system}")
            if field == "memory_bytes":
                value = int(value)
                if value <= 0:
                    raise ValueError("physical memory must be a positive integer")
            elif not value:
                raise ValueError("CPU description unavailable")
            values[field] = value
            details[field] = {"source": source, "status": "collected", "reason": None}
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError) as exc:
            details[field] = {"source": source, "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return {**values, "hardware_metadata": details}


def stop_process(process):
    """Always reap the isolated API child, including after a timeout."""
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


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


def collect_provider(args, out, provider, cases, metadata):
    # Stores and partial rows remain inspectable even when later cases fail.
    directory = out / f"store-{provider}"
    counts = fixture_store(directory)
    rows = []
    write(out / f"{provider}.json", rows)
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
                    # A stale server already on the port must not answer for this configuration.
                    if Path(str(health.get("data_dir"))).resolve() != directory.resolve():
                        raise RuntimeError("Eval server is not the isolated child; is the port already in use?")
                    if not all(health.get(k) == v for k, v in counts.items()):
                        raise RuntimeError("Eval server fixture counts do not match; inspect its log")
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError("Eval server did not become ready")
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
                answer = response.get("answer", "")
                unavailable = ("configured but unavailable" in answer
                               or "timed out before an answer was available" in answer)
                accepted = provider == "ollama" and response["usage"]["provider"] == "ollama" and not unavailable and not fallback
                row = {"id": case["id"], "category": case["category"], "configuration": provider,
                       "endpoint": endpoint, "request": payload, "response": response,
                       "latency_ms": round(elapsed, 3), "format_compliance": 1,
                       "accepted_model_output": accepted, "validation_fallback": fallback,
                       "model_unavailable": unavailable}
                rows.append(row)
                write(out / f"{provider}.json", rows)
                metadata["provider_status"][provider]["completed_cases"] = len(rows)
                metadata["output_sha256"][f"{provider}.json"] = sha(out / f"{provider}.json")
                write(out / "metadata.json", metadata)
                print(f"{provider} {case['id']} {elapsed:.0f}ms accepted={accepted} fallback={fallback}", flush=True)
        finally:
            stop_process(process)


def run(args):
    providers = selected_providers(getattr(args, "providers", "both"))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing", "selected_providers": list(providers),
        "provider_status": {provider: {"status": "pending", "completed_cases": 0} for provider in providers},
        "output_sha256": {}, "runner_sha256": sha(__file__),
        "python": sys.version, "platform": platform.platform(), **hardware_metadata(),
        "ollama_version": None, "model_name": args.model if "ollama" in providers else None,
        "model_details": None, "model_tags": None,
        "ollama_metadata_status": "pending" if "ollama" in providers else "not_requested",
        "corpus": "Unmodified synthetic canonical fixture chunks, not SEC filings; hashing embeddings regenerated.",
        "routing": "comparison -> /research/compare (deterministic in both modes); all other cases -> /research/chat",
        "filters": "None; top_k=8; ordinary application source-period policy applies. No gold documents or answers sent to model.",
        "repetitions": 1, "warmup": "No benchmark warmup; model may already be resident from preflight.",
        "latency_definition": "Sequential end-to-end localhost HTTP wall time, including persistence; excludes server startup.",
    }
    write(out / "metadata.json", metadata)
    active = None
    try:
        cases = [json.loads(x) for x in (ROOT / SOURCES[0]).read_text().splitlines() if x.strip()]
        if len(cases) != 35 or len({x["id"] for x in cases}) != 35:
            raise ValueError("Expected exactly 35 unique canonical cases")
        metadata.update({
            "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
        })
        if "ollama" in providers:
            # Selected Ollama is required: preflight failure is explicit, never a local substitute.
            try:
                metadata["ollama_version"] = http(args.ollama.rstrip("/") + "/api/version")
                metadata["model_details"] = {k: v for k, v in http(args.ollama.rstrip("/") + "/api/show", {"model": args.model}).items()
                                             if k != "modelfile"}
                metadata["model_tags"] = {"models": [m for m in http(args.ollama.rstrip("/") + "/api/tags")["models"] if m["name"] == args.model]}
                metadata["ollama_metadata_status"] = "collected"
            except Exception:
                metadata["ollama_metadata_status"] = "failed"
                raise
        metadata["status"] = "collecting"
        write(out / "metadata.json", metadata)
        for provider in providers:
            active = provider
            metadata["provider_status"][provider]["status"] = "running"
            write(out / "metadata.json", metadata)
            collect_provider(args, out, provider, cases, metadata)
            metadata["provider_status"][provider]["status"] = "completed"
            active = None
        metadata["status"] = "completed"
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as exc:
        if active is not None:
            metadata["provider_status"][active]["status"] = "failed"
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write(out / "metadata.json", metadata)
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--providers", choices=tuple(PROVIDER_CHOICES), default="both",
                        help="local deterministic baseline, Ollama model, or both (default).")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--port", type=int, default=8011)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
