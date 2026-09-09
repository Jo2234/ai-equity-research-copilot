"""Offline orchestration tests; historical responses are fixtures, never new model evidence."""
from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "evals/results/2026-09-29/bounded-structured-output"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_live_eval")
scorer = load_script("score_live_eval")


@pytest.mark.parametrize("error", [FileNotFoundError("sysctl missing"), PermissionError("denied"),
                                    subprocess.CalledProcessError(1, "sysctl"),
                                    subprocess.TimeoutExpired("sysctl", 5)])
def test_darwin_optional_metadata_errors_never_abort(monkeypatch, error):
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runner.subprocess, "check_output", Mock(side_effect=error))
    metadata = runner.hardware_metadata()
    assert metadata["cpu"] is None and metadata["memory_bytes"] is None
    assert all(v["status"] == "unavailable" and v["reason"] for v in metadata["hardware_metadata"].values())


@pytest.mark.parametrize("output", ["", "not a number", "0", "-1"])
def test_darwin_bad_memory_is_null_with_reason(monkeypatch, output):
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runner.subprocess, "check_output", Mock(side_effect=["Test CPU\n", output]))
    metadata = runner.hardware_metadata()
    assert metadata["cpu"] == "Test CPU" and metadata["memory_bytes"] is None
    assert metadata["hardware_metadata"]["memory_bytes"]["reason"]


def test_linux_metadata_uses_no_sysctl(monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    # `uname -p` reports "unknown" or a bare ISA on Linux; it must never be recorded as the CPU.
    monkeypatch.setattr(runner.platform, "processor", Mock(return_value="unknown"))
    monkeypatch.setattr(runner.Path, "read_text", Mock(
        return_value="processor\t: 0\nvendor_id\t: Test\nmodel name\t: Test Linux CPU @ 3.00GHz\n"))
    monkeypatch.setattr(runner.os, "sysconf", lambda key: {"SC_PHYS_PAGES": 1024, "SC_PAGE_SIZE": 4096}[key])
    forbidden = Mock(side_effect=AssertionError("No subprocess metadata on Linux"))
    monkeypatch.setattr(runner.subprocess, "check_output", forbidden)
    metadata = runner.hardware_metadata()
    assert metadata["cpu"] == "Test Linux CPU @ 3.00GHz" and metadata["memory_bytes"] == 4194304
    assert metadata["hardware_metadata"]["cpu"] == {"source": "/proc/cpuinfo", "status": "collected", "reason": None}
    forbidden.assert_not_called()
    runner.platform.processor.assert_not_called()


def test_linux_cpuinfo_without_model_name_is_unavailable(monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner.platform, "processor", lambda: "x86_64")
    monkeypatch.setattr(runner.Path, "read_text", Mock(return_value="processor\t: 0\nflags\t: fpu\n"))
    monkeypatch.setattr(runner.os, "sysconf", lambda key: 4096)
    metadata = runner.hardware_metadata()
    assert metadata["cpu"] is None and metadata["memory_bytes"] == 4096 * 4096
    assert metadata["hardware_metadata"]["cpu"]["status"] == "unavailable"
    assert metadata["hardware_metadata"]["cpu"]["reason"]


def test_linux_unavailable_metadata_and_windows_metadata(monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner.Path, "read_text", Mock(side_effect=PermissionError("proc denied")))
    monkeypatch.setattr(runner.os, "sysconf", Mock(side_effect=ValueError("unsupported sysconf")))
    metadata = runner.hardware_metadata()
    assert metadata["cpu"] is None and metadata["memory_bytes"] is None
    monkeypatch.setattr(runner.platform, "system", lambda: "Windows")
    command = Mock(side_effect=["Windows CPU\n", "8589934592\n"])
    monkeypatch.setattr(runner.subprocess, "check_output", command)
    metadata = runner.hardware_metadata()
    assert metadata["cpu"] == "Windows CPU" and metadata["memory_bytes"] == 8589934592
    assert all(call.args[0][0] == "powershell.exe" and call.kwargs["timeout"] == 5 for call in command.call_args_list)


@pytest.mark.parametrize("value", ["", "local,ollama", "local,local", "unknown"])
def test_provider_selection_rejects_invalid_input(tmp_path, value):
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--output", str(tmp_path / "run"), "--providers", value])
    with pytest.raises(ValueError):
        runner.selected_providers(value)


def test_default_keeps_both_order(tmp_path):
    args = runner.build_parser().parse_args(["--output", str(tmp_path / "run")])
    assert runner.selected_providers(args.providers) == ("ollama", "local")


class FakeProcess:
    def __init__(self, env, exit_code=None):
        self.env = env
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        self.waited = True
        return 0


@pytest.fixture
def fake_capture(monkeypatch):
    processes, urls = [], []
    by_mode = {mode: json.loads((HISTORY / f"{mode}.json").read_text()) for mode in ("local", "ollama")}
    def launch(*args, **kwargs):
        process = FakeProcess(kwargs["env"])
        processes.append(process)
        return process
    def transport(url, payload=None):
        urls.append(url)
        if "/api/" in url:
            if url.endswith("/version"):
                return {"version": "offline fake"}
            if url.endswith("/show"):
                return {"details": {"fixture": True}, "modelfile": "must not persist"}
            if url.endswith("/tags"):
                return {"models": [{"name": "gemma3:4b"}]}
            raise AssertionError(url)
        process = processes[-1]
        store = json.loads((Path(process.env["AIERC_DATA_DIR"]) / "storage/state/store.json").read_text())
        if url.endswith("/health"):
            return {"data_dir": process.env["AIERC_DATA_DIR"],
                    **{key: len(store[key]) for key in ("companies", "documents", "chunks")}}
        mode = process.env["AIERC_LLM_PROVIDER"]
        return next(row["response"] for row in by_mode[mode] if row["request"] == payload)
    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda command, **kwargs: "fixture-commit" if "rev-parse" in command else "")
    monkeypatch.setattr(runner.platform, "platform", lambda: "offline test platform")
    monkeypatch.setattr(runner, "http", transport)
    monkeypatch.setattr(runner, "hardware_metadata", lambda: {"cpu": None, "memory_bytes": None, "hardware_metadata": {}})
    return processes, urls, transport


def args_for(tmp_path, mode="local"):
    return runner.build_parser().parse_args(["--output", str(tmp_path / "run"), "--providers", mode])


@pytest.mark.parametrize("mode", ["local", "ollama", "both"])
def test_capture_only_selected_providers_and_truthful_flags(tmp_path, fake_capture, mode):
    processes, urls, _ = fake_capture
    args = args_for(tmp_path, mode)
    metadata = runner.run(args)
    chosen = runner.selected_providers(mode)
    assert metadata["selected_providers"] == list(chosen) and metadata["status"] == "completed"
    assert [p.env["AIERC_LLM_PROVIDER"] for p in processes] == list(chosen)
    assert all(p.terminated and p.waited for p in processes)
    assert len([url for url in urls if "/api/" in url]) == (0 if mode == "local" else 3)
    assert metadata["ollama_metadata_status"] == ("not_requested" if mode == "local" else "collected")
    if mode == "local":
        assert all(metadata[key] is None for key in ("ollama_version", "model_name", "model_details", "model_tags"))
        assert not (args.output / "ollama.json").exists()
    else:
        assert "modelfile" not in metadata["model_details"]
    for provider in chosen:
        rows = json.loads((args.output / f"{provider}.json").read_text())
        assert len(rows) == 35 and metadata["provider_status"][provider]["completed_cases"] == 35
        assert metadata["output_sha256"][f"{provider}.json"] == runner.sha(args.output / f"{provider}.json")
        if provider == "local":
            assert not any(row["accepted_model_output"] for row in rows)
        else:
            assert any(row["validation_fallback"] or row["model_unavailable"] for row in rows)
            assert all(not row["accepted_model_output"] for row in rows if row["validation_fallback"] or row["model_unavailable"])


def test_ollama_unavailable_is_explicit_failed_preflight(tmp_path, fake_capture, monkeypatch):
    processes, _, _ = fake_capture
    monkeypatch.setattr(runner, "http", Mock(side_effect=OSError("Ollama unavailable")))
    args = args_for(tmp_path, "both")
    with pytest.raises(OSError):
        runner.run(args)
    metadata = json.loads((args.output / "metadata.json").read_text())
    assert metadata["status"] == "failed" and metadata["ollama_metadata_status"] == "failed"
    assert metadata["error"]["type"] == "OSError" and not processes
    assert not (args.output / "local.json").exists()


@pytest.mark.parametrize("failure", [OSError("HTTP timeout"), {}])
def test_partial_http_or_schema_failure_preserves_evidence_and_reaps(tmp_path, fake_capture, monkeypatch, failure):
    processes, _, transport = fake_capture
    calls = 0
    def broken(url, payload=None):
        nonlocal calls
        if not url.endswith("/health"):
            calls += 1
            if calls == 3:
                if isinstance(failure, Exception):
                    raise failure
                return failure
        return transport(url, payload)
    monkeypatch.setattr(runner, "http", broken)
    args = args_for(tmp_path)
    with pytest.raises(OSError if isinstance(failure, Exception) else ValidationError):
        runner.run(args)
    metadata = json.loads((args.output / "metadata.json").read_text())
    rows = json.loads((args.output / "local.json").read_text())
    assert len(rows) == 2 and metadata["provider_status"]["local"] == {"status": "failed", "completed_cases": 2}
    assert metadata["status"] == "failed" and metadata["error"] and processes[0].terminated and processes[0].waited
    assert (args.output / "store-local/storage/state/store.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        scorer.providers_for(args.output)


@pytest.mark.parametrize("failure", ["startup", "health"])
def test_startup_and_health_timeout_cleanup(tmp_path, fake_capture, monkeypatch, failure):
    processes, _, _ = fake_capture
    if failure == "startup":
        def launch(*args, **kwargs):
            process = FakeProcess(kwargs["env"], exit_code=1)
            processes.append(process)
            return process
        monkeypatch.setattr(runner.subprocess, "Popen", launch)
    else:
        monkeypatch.setattr(runner, "http", Mock(side_effect=OSError("not ready")))
        monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError):
        runner.run(args_for(tmp_path))
    assert processes[0].waited
    assert json.loads((tmp_path / "run/metadata.json").read_text())["status"] == "failed"


def test_stale_server_on_port_is_rejected_not_captured(tmp_path, fake_capture, monkeypatch):
    processes, _, transport = fake_capture
    def foreign(url, payload=None):
        health = transport(url, payload)
        if url.endswith("/health"):
            # Same fixture counts, but another process's store (for example an Ollama-configured run).
            health["data_dir"] = str(tmp_path / "other-run/store-ollama")
        return health
    monkeypatch.setattr(runner, "http", foreign)
    args = args_for(tmp_path)
    with pytest.raises(RuntimeError, match="not the isolated child"):
        runner.run(args)
    assert json.loads((args.output / "local.json").read_text()) == []
    assert processes[0].terminated and processes[0].waited
    assert json.loads((args.output / "metadata.json").read_text())["provider_status"]["local"]["status"] == "failed"


def test_model_timeout_is_unavailable_not_accepted(tmp_path, fake_capture, monkeypatch):
    _, _, transport = fake_capture
    accepted = next(row for row in json.loads((HISTORY / "ollama.json").read_text())
                    if row["accepted_model_output"] and row["endpoint"] == "/research/chat")
    def timed_out(url, payload=None):
        response = transport(url, payload)
        if payload == accepted["request"]:
            response = {**response, "answer": "The local model request timed out before an answer was "
                        "available. Try fewer documents or increase the configured timeout.",
                        "key_points": [], "citations": [], "limitations": ["Ollama request timed out"]}
        return response
    monkeypatch.setattr(runner, "http", timed_out)
    args = args_for(tmp_path, "ollama")
    runner.run(args)
    row = next(r for r in json.loads((args.output / "ollama.json").read_text()) if r["id"] == accepted["id"])
    assert row["response"]["usage"]["provider"] == "ollama"
    assert row["model_unavailable"] and not row["accepted_model_output"]


def test_stubborn_child_is_killed_and_reaped():
    process = FakeProcess({})
    process.wait = Mock(side_effect=[subprocess.TimeoutExpired("uvicorn", 15), 0])
    runner.stop_process(process)
    assert process.terminated and process.killed and process.wait.call_count == 2


def test_process_creation_failure_keeps_failed_metadata(tmp_path, fake_capture, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", Mock(side_effect=FileNotFoundError("uvicorn launcher unavailable")))
    args = args_for(tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.run(args)
    metadata = json.loads((args.output / "metadata.json").read_text())
    assert metadata["status"] == "failed" and metadata["provider_status"]["local"]["status"] == "failed"
    assert json.loads((args.output / "local.json").read_text()) == []


def test_required_source_hash_failure_keeps_diagnostics(tmp_path, fake_capture, monkeypatch):
    original = runner.sha
    def missing_source(path):
        if str(path).endswith("scoring_rubric.json"):
            raise FileNotFoundError("missing required scoring provenance")
        return original(path)
    monkeypatch.setattr(runner, "sha", missing_source)
    args = args_for(tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.run(args)
    metadata = json.loads((args.output / "metadata.json").read_text())
    assert metadata["status"] == "failed" and metadata["error"]["type"] == "FileNotFoundError"
    assert not fake_capture[0]


def history_copy(tmp_path, modes=("local",)):
    tmp_path.mkdir(exist_ok=True)
    reviews = json.loads((HISTORY / "review.json").read_text())
    for mode in modes:
        shutil.copyfile(HISTORY / f"{mode}.json", tmp_path / f"{mode}.json")
    (tmp_path / "review.json").write_text(json.dumps({mode: reviews[mode] for mode in modes}))
    return json.loads((HISTORY / "summary.json").read_text())


@pytest.mark.parametrize("modes", [("local",), ("ollama",), ("ollama", "local")])
def test_selected_scoring_preserves_historical_metrics_and_hashes(tmp_path, modes):
    historical = history_copy(tmp_path, modes)
    (tmp_path / "metadata.json").write_text(json.dumps({"selected_providers": list(modes)}))
    summary = scorer.score(tmp_path)
    assert summary == {mode: historical[mode] for mode in modes}
    assert set(summary) == set(modes)
    assert (tmp_path / "scores.csv").exists()


def test_historical_no_selection_defaults_to_both_available_captures(tmp_path):
    historical = history_copy(tmp_path, ("ollama", "local"))
    shutil.copyfile(HISTORY / "metadata.json", tmp_path / "metadata.json")
    assert scorer.score(tmp_path) == historical


def test_explicit_completed_provider_can_be_scored_from_failed_dual_run(tmp_path):
    historical = history_copy(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "status": "failed", "selected_providers": ["ollama", "local"],
        "provider_status": {"local": {"status": "completed"}, "ollama": {"status": "failed"}},
    }))
    assert scorer.score(tmp_path, "local") == {"local": historical["local"]}
    with pytest.raises(ValueError, match="Missing selected provider"):
        scorer.score(tmp_path)


def test_scoring_stale_review_and_selected_missing_capture_fail(tmp_path):
    history_copy(tmp_path)
    review = json.loads((tmp_path / "review.json").read_text())
    review["local"][0]["response_sha256"] = "stale"
    (tmp_path / "review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match="Stale review"):
        scorer.score(tmp_path)
    assert not (tmp_path / "summary.json").exists()
    with pytest.raises(ValueError, match="Missing selected provider"):
        scorer.score(tmp_path, "both")


def test_scoring_hash_mismatch_and_bad_recorded_selection_fail(tmp_path):
    history_copy(tmp_path)
    metadata = {"selected_providers": ["local"], "output_sha256": {"local.json": "wrong"}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash mismatch"):
        scorer.score(tmp_path)
    for invalid in ([], ["local", "local"], ["unknown"], "local"):
        (tmp_path / "metadata.json").write_text(json.dumps({"selected_providers": invalid}))
        with pytest.raises(ValueError):
            scorer.score(tmp_path)


def test_existing_run_directory_is_never_overwritten(tmp_path, fake_capture):
    args = args_for(tmp_path)
    args.output.mkdir()
    marker = args.output / "keep.txt"
    marker.write_text("historical")
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert marker.read_text() == "historical"


def test_all_35_cases_use_real_local_http_with_zero_ollama_requests(tmp_path):
    requests = []

    class ForbiddenOllama(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_error(503, "Ollama forbidden in local-only test")

        do_POST = do_GET

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), ForbiddenOllama) as sentinel:
        thread = threading.Thread(target=sentinel.serve_forever, daemon=True)
        thread.start()
        try:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                api_port = sock.getsockname()[1]
            args = runner.build_parser().parse_args([
                "--output", str(tmp_path / "real-run"), "--providers", "local",
                "--port", str(api_port), "--ollama", f"http://127.0.0.1:{sentinel.server_port}",
            ])
            metadata = runner.run(args)
            rows = json.loads((args.output / "local.json").read_text())
            assert metadata["status"] == "completed" and metadata["selected_providers"] == ["local"]
            assert metadata["ollama_metadata_status"] == "not_requested" and not requests
            assert len(rows) == 35 and len({row["id"] for row in rows}) == 35
            assert all(row["response"]["usage"]["provider"] == "local" for row in rows)
            assert not any(row["accepted_model_output"] or row["model_unavailable"] for row in rows)
            assert not (args.output / "ollama.json").exists()
            with socket.socket() as sock:
                assert sock.connect_ex(("127.0.0.1", api_port)) != 0
        finally:
            sentinel.shutdown()
            thread.join(timeout=2)
