"""Acceptance tests for MinTok production platform: Engine, Codex Console, API Server, and Config."""

import json
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest
from mintok.api.server import MinTokServer
from mintok.cli import main as mintok_main
from mintok.config import (
    clear_credentials,
    get_config,
    load_all_config,
    load_credentials,
    save_credentials,
    set_config,
)
from mintok.engine import MinTokEngine, MockModelProvider, ProviderResponse, calculate_credit_cost
from mintok.interactive import InteractiveConsole, list_sessions, load_session, sanitize_text


@pytest.mark.domain
def test_credit_cost_calculation():
    """Verify itemized credit pricing across tiers."""
    # mintok-max: 0.5 base + 0.8 per 1k
    c_max = calculate_credit_cost("mintok-max", prompt_tokens=2000, completion_tokens=1000)
    assert c_max == round(0.5 + 3.0 * 0.8, 4)

    # mintok-pro: 0.1 base + 0.2 per 1k
    c_pro = calculate_credit_cost("mintok-pro", prompt_tokens=1500, completion_tokens=500)
    assert c_pro == round(0.1 + 2.0 * 0.2, 4)

    # mintok-flash: 0.02 base + 0.05 per 1k
    c_flash = calculate_credit_cost("mintok-flash", prompt_tokens=1000, completion_tokens=0)
    assert c_flash == round(0.02 + 1.0 * 0.05, 4)

    # BYOK is zero credits
    c_byok = calculate_credit_cost("mintok-pro", 5000, 2000, is_byok=True)
    assert c_byok == 0.0


@pytest.mark.domain
def test_secret_scrubbing_defense():
    """Verify hostile/sensitive secrets are redacted from context."""
    raw = "Found API key sk-abc12345678901234567890 and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 in config."
    clean = sanitize_text(raw)
    assert "sk-" not in clean
    assert "Bearer" not in clean
    assert "[REDACTED_SECRET]" in clean


@pytest.mark.domain
def test_engine_execution_and_telemetry():
    """Verify decoupled MinTokEngine runs tasks and yields accurate telemetry."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        (ws / "math_ops.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

        provider = MockModelProvider([
            ProviderResponse(
                content="ACTION: query symbol:math_ops.add",
                action="query",
                target="math_ops.add",
                prompt_tokens=40,
                completion_tokens=10,
            ),
            ProviderResponse(
                content="ACTION: finish done",
                action="finish",
                target="done",
                prompt_tokens=50,
                completion_tokens=10,
            ),
        ])

        engine = MinTokEngine(workspace=ws, provider=provider, model="mintok-pro")
        result = engine.run(task="Inspect add function")

        assert result.success is True
        assert result.status == "completed"
        assert len(result.turns) == 2
        assert result.usage.prompt_tokens == 90
        assert result.usage.completion_tokens == 20
        assert result.usage.credit_cost > 0.0


@pytest.mark.domain
def test_interactive_codex_slash_commands():
    """Verify Codex-style interaction lifecycle and slash commands in console."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        (ws / "service.py").write_text("class Auth:\n    def login(self): pass\n")

        console = InteractiveConsole(workspace=ws)

        # 1. /goal sets durable objective and advances state machine
        assert console.session.state == "GOAL"
        console._handle_slash_command("/goal Migrate auth backend")
        assert console.session.goal == "Migrate auth backend"
        assert console.session.state == "PLAN"

        # 2. /plan generates structured execution steps
        console._handle_slash_command("/plan")
        assert len(console.session.plan) == 4
        assert console.session.state == "ACTION"

        # 3. /side handles side question without disturbing trajectory
        console._handle_slash_command("/side What symbols exist?")
        assert len(console.session.side_questions) == 1

        # 4. /budget updates token budget
        console._handle_slash_command("/budget 30000")
        assert console.session.budget == 30000

        # 5. /fast toggles high-speed mode
        console._handle_slash_command("/fast")
        assert console.session.model == "mintok-flash"

        # 6. /fork branches session
        orig_id = console.session.session_id
        console._handle_slash_command("/fork")
        assert console.session.session_id != orig_id
        assert console.session.parent_session_id == orig_id

        # 7. /compact rebuilds minimal sufficient state from AST IR
        console._handle_slash_command("/compact")

        # 8. /review generates structured code and test review
        console._handle_slash_command("/review")

        # 9. /why provides causal dependency explanation
        console._handle_slash_command("/why service.py")

        # 10. /sessions lists persistent sessions
        sessions = list_sessions()
        assert len(sessions) >= 2


@pytest.mark.domain
def test_config_and_credentials_store():
    """Verify configuration reading, writing, and credentials handling."""
    # Test config
    set_config("test_param", 12345)
    assert get_config("test_param") == 12345

    # Test credentials save/clear
    save_credentials("test_key_xyz", endpoint="https://custom.mintok.ai/v1", email="dev@mintok.ai")
    creds = load_credentials()
    assert creds["api_key"] == "test_key_xyz"
    assert creds["endpoint"] == "https://custom.mintok.ai/v1"
    assert creds["email"] == "dev@mintok.ai"

    assert clear_credentials() is True
    assert load_credentials()["api_key"] == ""


@pytest.mark.integration
def test_api_server_endpoints():
    """Verify standalone MinTok HTTP API server with OpenAI and Agent endpoints."""
    port = 8991
    server = MinTokServer(host="127.0.0.1", port=port, api_key="secret_test_token")
    server.start()
    time.sleep(0.1)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {"Authorization": "Bearer secret_test_token", "Content-Type": "application/json"}

    try:
        # 1. GET /health
        with opener.open(f"http://127.0.0.1:{port}/health") as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "healthy"

        # 2. GET /v1/models
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers=headers)
        with opener.open(req) as resp:
            data = json.loads(resp.read().decode())
            assert len(data["data"]) == 3

        # 3. POST /v1/chat/completions
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": "mintok-pro", "messages": [{"role": "user", "content": "Hello"}]}).encode(),
            headers=headers,
            method="POST",
        )
        with opener.open(req) as resp:
            data = json.loads(resp.read().decode())
            assert "choices" in data
            assert data["mintok_context_metrics"]["information_density_ratio"] > 1.0

        # 4. GET /v1/account
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/account", headers=headers)
        with opener.open(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["credits_balance"] > 0

        # 5. POST /v1/agent/runs
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/agent/runs",
            data=json.dumps({"task": "Test task", "workspace": ".", "sync": True}).encode(),
            headers=headers,
            method="POST",
        )
        with opener.open(req) as resp:
            data = json.loads(resp.read().decode())
            assert "task_id" in data
            assert data["status"] == "completed"

    finally:
        server.stop()


@pytest.mark.integration
def test_cli_subcommands(capsys):
    """Verify CLI subcommands: models, config, usage."""
    code = mintok_main(["models", "--format", "json"])
    assert code == 0
    captured = capsys.readouterr()
    models = json.loads(captured.out)
    assert len(models) >= 3

    code = mintok_main(["config", "list", "--format", "json"])
    assert code == 0

    code = mintok_main(["usage", "--format", "json"])
    assert code == 0
