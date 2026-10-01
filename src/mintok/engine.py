"""MinTok Production Execution Engine.

Environment-agnostic core runtime orchestrating semantic Agent ABI,
output virtualization, causal observation caching, and inference control.
Decoupled from CLI, Web, and Cloud layers: runnable locally, inside CLI,
inside a Celery/Django worker, or inside a RunPod serverless GPU container.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from mintok.abi import AgentABI, ChangeRejected, filtered_tool_surface_json
from mintok.mintok_controller import ControllerDecision, MinTokController
from mintok.predictor import AgentState
from mintok.tokens import estimate_tokens
from mintok.virtualization import ObservationStore, ToolOutputVirtualizer


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call requested by the model or controller."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Standardized response from an LLM provider."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    action: str | None = None
    target: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""


class ModelProvider(ABC):
    """Abstract interface for LLM inference providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> ProviderResponse:
        """Generate model response given prompt, system instructions, and tool schemas."""
        ...


class MockModelProvider(ModelProvider):
    """Deterministic scriptable provider for testing and sandboxed verification."""

    def __init__(
        self,
        canned_responses: list[ProviderResponse] | None = None,
        default_model: str = "mock-model",
    ) -> None:
        self.canned_responses = list(canned_responses or [])
        self.call_count = 0
        self.default_model = default_model
        self.history_records: list[dict[str, Any]] = []

    def add_response(self, response: ProviderResponse) -> None:
        self.canned_responses.append(response)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> ProviderResponse:
        self.history_records.append({
            "prompt": prompt,
            "system": system,
            "tools": tools,
            "history": history,
        })
        if self.call_count < len(self.canned_responses):
            resp = self.canned_responses[self.call_count]
            self.call_count += 1
            return resp

        # Default fallback response
        self.call_count += 1
        prompt_tokens = estimate_tokens(prompt) + (estimate_tokens(system) if system else 0)
        return ProviderResponse(
            content="Task completed successfully.",
            action="finish",
            target="done",
            prompt_tokens=prompt_tokens,
            completion_tokens=20,
            model=self.default_model,
        )


class OpenAICompatibleProvider(ModelProvider):
    """Provider connecting to standard OpenAI-compatible endpoints via stdlib urllib."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> ProviderResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            if "127.0.0.1" in self.base_url or "localhost" in self.base_url:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=self.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            else:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as err:
            raise RuntimeError(f"OpenAI endpoint call failed: {err}") from err

        choice = data.get("choices", [{}])[0].get("message", {})
        content = choice.get("content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", estimate_tokens(prompt))
        completion_tokens = usage.get("completion_tokens", estimate_tokens(content))
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        # Parse tool calls or actions from content
        action = None
        target = None
        tool_calls: list[ToolCall] = []

        if "tool_calls" in choice:
            for tc in choice["tool_calls"]:
                fn = tc.get("function", {})
                tool_calls.append(ToolCall(
                    name=fn.get("name", ""),
                    arguments=json.loads(fn.get("arguments", "{}")),
                ))

        if not action and content:
            action, target = _parse_action_text(content)

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            action=action,
            target=target,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            model=data.get("model", self.model),
        )


class AnthropicProvider(ModelProvider):
    """Provider connecting to Anthropic Claude messages API via stdlib urllib."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        model: str = "claude-3-5-sonnet-20241022",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> ProviderResponse:
        messages: list[dict[str, str]] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        url = f"{self.base_url}/messages"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as err:
            raise RuntimeError(f"Anthropic endpoint call failed: {err}") from err

        content_blocks = data.get("content", [])
        content_text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", estimate_tokens(prompt))
        completion_tokens = usage.get("output_tokens", estimate_tokens(content_text))
        cached_tokens = usage.get("cache_read_input_tokens", 0)

        action, target = _parse_action_text(content_text)

        return ProviderResponse(
            content=content_text,
            action=action,
            target=target,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            model=data.get("model", self.model),
        )



class RunPodProvider(ModelProvider):
    """Provider connecting to RunPod Serverless GPU Inference Workers."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_id: str = "mintok-pro",
        model: str = "mintok-pro",
        base_url: str = "https://api.runpod.ai/v2",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY", "")
        self.endpoint_id = endpoint_id
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> ProviderResponse:
        url = f"{self.base_url}/{self.endpoint_id}/runsync"
        input_data = {
            "prompt": prompt,
            "system": system,
            "tools": list(tools or []),
            "history": list(history or []),
            "model": self.model,
            "temperature": 0.0,
        }
        payload = {"input": input_data}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as err:
            raise RuntimeError(f"RunPod endpoint call failed: {err}") from err

        output = data.get("output", {})
        content = output.get("content", "")
        action = output.get("action")
        target = output.get("target")
        if not action and content:
            action, target = _parse_action_text(content)

        return ProviderResponse(
            content=content,
            action=action,
            target=target,
            prompt_tokens=output.get("prompt_tokens", estimate_tokens(prompt)),
            completion_tokens=output.get("completion_tokens", estimate_tokens(content)),
            cached_tokens=output.get("cached_tokens", 0),
            model=self.model,
        )


def _parse_action_text(text: str) -> tuple[str | None, str | None]:
    """Parse structured action line from model output text if present.

    Format examples:
    - `ACTION: query symbol:MyClass`
    - `ACTION: patch src/app.py`
    - `ACTION: verify pytest`
    - `ACTION: finish done`
    """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ACTION:"):
            parts = line[7:].strip().split(maxsplit=1)
            action = parts[0] if parts else None
            target = parts[1] if len(parts) > 1 else ""
            return action, target
    return None, None


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Itemized token and compute accounting for a complete agent run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    tool_tokens: int = 0
    total_tokens: int = 0
    turns: int = 0
    duration_seconds: float = 0.0
    credit_cost: float = 0.0
    avoided_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """Execution telemetry captured for each single agent turn."""

    turn: int
    phase: str
    action: str
    target: str
    intercepted: bool
    interception_reason: str
    observation_summary: str
    observation_handle: str | None
    raw_observation_tokens: int
    virtualized_tokens: int
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Final result of a MinTok engine execution run."""

    task_id: str
    task: str
    workspace: str
    model: str
    success: bool
    status: str
    patch: str
    turns: list[TurnRecord]
    usage: RunUsage
    verifications_passed: int
    interceptions_count: int
    evidence_sufficiency_achieved: bool
    stop_loss_triggered: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [t.to_dict() if hasattr(t, "to_dict") else t for t in self.turns]
        d["usage"] = self.usage.to_dict() if hasattr(self.usage, "to_dict") else self.usage
        return d


def calculate_credit_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    is_byok: bool = False,
) -> float:
    """Calculate billing credits for a run.

    Hosted models:
    - mintok-max: $0.008 / 1k tokens = 0.8 credits / 1k tokens (1 credit = $0.01) + 0.5 base
    - mintok-pro: $0.002 / 1k tokens = 0.2 credits / 1k tokens + 0.1 base
    - mintok-flash: $0.0005 / 1k tokens = 0.05 credits / 1k tokens + 0.02 base
    BYOK: 0.0 credits.
    """
    if is_byok:
        return 0.0

    m = model.lower()
    total_k = (prompt_tokens + completion_tokens) / 1000.0

    if "max" in m:
        return round(0.5 + total_k * 0.8, 4)
    elif "flash" in m:
        return round(0.02 + total_k * 0.05, 4)
    else:  # default pro
        return round(0.1 + total_k * 0.2, 4)


class MinTokEngine:
    """The MinTok Execution Engine.

    Encapsulates all inference compilation logic into a single callable runtime:
    - ABI environment binding (`AgentABI`)
    - Output virtualization & handle compression (`OutputVirtualizer`, `ObservationStore`)
    - Active policy control (`MinTokController`)
    - Stopping criteria & budget enforcement
    """

    def __init__(
        self,
        workspace: str | Path,
        model: str = "mintok-pro",
        provider: ModelProvider | None = None,
        byok: bool = False,
        api_key: str | None = None,
        api_base: str | None = None,
        max_turns: int = 15,
        token_budget: int = 50000,
        enable_virtualization: bool = True,
        enable_controller: bool = True,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.model = model
        self.byok = byok
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.enable_virtualization = enable_virtualization
        self.enable_controller = enable_controller

        # Select or initialize provider
        if provider is not None:
            self.provider = provider
        elif byok or api_base:
            self.provider = OpenAICompatibleProvider(
                api_key=api_key,
                base_url=api_base or "https://api.openai.com/v1",
                model=model,
            )
        elif os.getenv("RUNPOD_API_KEY"):
            self.provider = RunPodProvider(
                api_key=api_key or os.getenv("RUNPOD_API_KEY"),
                endpoint_id=model,
                model=model,
            )
        else:
            # Fallback to local deterministic mock provider if no keys configured
            self.provider = MockModelProvider(default_model=model)

        # Initialize core components
        self.abi = AgentABI(self.workspace)
        self.obs_store = ObservationStore(threshold_chars=250)
        self.virtualizer = ToolOutputVirtualizer(self.obs_store)
        self.controller = MinTokController()

    def run(self, task: str, task_id: str | None = None) -> RunResult:
        """Execute the agent loop on the workspace to resolve the task."""
        t_start = time.time()
        tid = task_id or hashlib.sha256(f"{task}:{time.time()}".encode("utf-8")).hexdigest()[:12]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_tool_tokens = 0
        total_avoided_tokens = 0
        interceptions_count = 0
        verifications_passed = 0
        turns: list[TurnRecord] = []
        status = "completed"
        error_msg = None
        success = False
        patch = ""

        # Git initial state hash if in git repo
        initial_commit = self._get_git_head()

        for turn_idx in range(1, self.max_turns + 1):
            turn_t0 = time.time()

            # 1. Determine phase
            if self.controller.sufficiency.is_sufficient_to_patch:
                phase = "verify" if self.controller.has_pending_patch else "patch"
            else:
                phase = "localize"

            # 2. Check token budget
            current_tokens = total_prompt_tokens + total_completion_tokens + total_tool_tokens
            if current_tokens >= self.token_budget:
                status = "budget_exceeded"
                break

            # 3. Build turn prompt
            tools_json = filtered_tool_surface_json(phase)
            system_prompt = (
                f"You are MinTok Agent in phase '{phase}'. "
                f"Workspace: {self.workspace.name}. "
                f"Respond with 'ACTION: <action> <target>' followed by details.\n"
                f"Allowed tools: {tools_json}"
            )

            # 4. Generate model response
            try:
                resp = self.provider.generate(
                    prompt=f"Task: {task}\nTurn: {turn_idx}/{self.max_turns}\nCurrent Phase: {phase}",
                    system=system_prompt,
                )
            except Exception as e:
                status = "error"
                error_msg = str(e)
                break

            total_prompt_tokens += resp.prompt_tokens
            total_completion_tokens += resp.completion_tokens
            total_cached_tokens += resp.cached_tokens

            action_type = resp.action or "finish"
            target = resp.target or ""

            # Check for natural finish
            if action_type in ("finish", "done", "stop"):
                success = True
                status = "completed"
                turns.append(TurnRecord(
                    turn=turn_idx,
                    phase=phase,
                    action=action_type,
                    target=target,
                    intercepted=False,
                    interception_reason="Agent finished task.",
                    observation_summary=resp.content,
                    observation_handle=None,
                    raw_observation_tokens=estimate_tokens(resp.content),
                    virtualized_tokens=estimate_tokens(resp.content),
                    duration_ms=(time.time() - turn_t0) * 1000.0,
                ))
                break

            # 5. MinTok Controller Mediation
            intercepted = False
            interception_reason = ""
            allocated_action = action_type

            if self.enable_controller:
                agent_state = AgentState(
                    turn=turn_idx,
                    tokens_spent=current_tokens,
                )
                ctrl_dec = self.controller.process_agent_step(
                    state=agent_state,
                    requested_action=action_type,
                    target=target,
                    raw_tokens=estimate_tokens(resp.content),
                )
                if ctrl_dec.intercepted:
                    intercepted = True
                    interceptions_count += 1
                    interception_reason = ctrl_dec.interception_reason
                    allocated_action = ctrl_dec.allocated_action

            # 6. Execute action on AgentABI
            raw_obs, obs_handle, obs_tokens = self._execute_action(allocated_action, target, resp.content)
            total_tool_tokens += obs_tokens
            if self.enable_virtualization and obs_handle:
                total_avoided_tokens += max(0, estimate_tokens(raw_obs) - obs_tokens)

            if allocated_action == "verify" and "pass" in raw_obs.lower():
                verifications_passed += 1

            turns.append(TurnRecord(
                turn=turn_idx,
                phase=phase,
                action=allocated_action,
                target=target,
                intercepted=intercepted,
                interception_reason=interception_reason,
                observation_summary=raw_obs[:200] if len(raw_obs) > 200 else raw_obs,
                observation_handle=obs_handle,
                raw_observation_tokens=estimate_tokens(raw_obs),
                virtualized_tokens=obs_tokens,
                duration_ms=(time.time() - turn_t0) * 1000.0,
            ))

        # Final patch extraction
        patch = self._compute_patch(initial_commit)
        duration_s = round(time.time() - t_start, 3)
        total_tokens = total_prompt_tokens + total_completion_tokens + total_tool_tokens
        credits = calculate_credit_cost(self.model, total_prompt_tokens, total_completion_tokens, self.byok)

        usage = RunUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=total_cached_tokens,
            tool_tokens=total_tool_tokens,
            total_tokens=total_tokens,
            turns=len(turns),
            duration_seconds=duration_s,
            credit_cost=credits,
            avoided_tokens=total_avoided_tokens,
        )

        return RunResult(
            task_id=tid,
            task=task,
            workspace=str(self.workspace),
            model=self.model,
            success=success,
            status=status,
            patch=patch,
            turns=turns,
            usage=usage,
            verifications_passed=verifications_passed,
            interceptions_count=interceptions_count,
            evidence_sufficiency_achieved=self.controller.sufficiency.is_sufficient_to_patch,
            stop_loss_triggered=self.controller.stop_loss.interventions_count > 0,
            error_message=error_msg,
        )

    def _execute_action(self, action: str, target: str, content: str) -> tuple[str, str | None, int]:
        """Execute action via AgentABI or virtualization layer."""
        act_lower = action.lower()
        obs_text = ""

        try:
            if "query" in act_lower:
                op = "symbol"
                tgt = target
                if ":" in target:
                    op, tgt = target.split(":", 1)
                obs_text = self.abi.query(op, tgt)
            elif "change" in act_lower:
                # Target is symbol id, content contains new source
                res = self.abi.change(target, content)
                obs_text = f"Symbol {res.symbol.id} changed. interface_changed={res.interface_changed}"
            elif "add" in act_lower:
                # Add new symbol
                mod = "src/new_module.py"
                res = self.abi.add(target, mod, content)
                obs_text = f"Symbol {res.symbol.id} added. file_created={res.created_file}"
            elif "verify" in act_lower or "test" in act_lower:
                cmd = target or "pytest"
                cmd_parts = cmd.split()
                # Run safe verification
                sub_res = subprocess.run(
                    cmd_parts,
                    cwd=str(self.workspace),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                obs_text = f"exit_code: {sub_res.returncode}\nstdout:\n{sub_res.stdout}\nstderr:\n{sub_res.stderr}"
            else:
                obs_text = f"Action {action} acknowledged on {target}."
        except Exception as ex:
            obs_text = f"Action error: {ex}"

        # Virtualize if enabled
        if self.enable_virtualization:
            virt_summary, obs = self.virtualizer.virtualize(action, obs_text)
            obs_handle = obs.id if obs else None
            obs_tokens = estimate_tokens(virt_summary)
            return virt_summary, obs_handle, obs_tokens
        else:
            return obs_text, None, estimate_tokens(obs_text)

    def _get_git_head(self) -> str | None:
        try:
            res = subprocess.run(
                ["git", "-C", str(self.workspace), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return None

    def _compute_patch(self, initial_commit: str | None) -> str:
        """Extract git diff of changes made during execution."""
        try:
            cmd = ["git", "-C", str(self.workspace), "diff"]
            if initial_commit:
                cmd.append(initial_commit)
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return res.stdout
        except Exception:
            pass
        return ""
