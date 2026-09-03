from __future__ import annotations

from openfoam_agent import __version__

import argparse
import json
import os
import re
import sysconfig
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence

from openfoam_agent.conversation import ConversationSession, InteractionMode
from openfoam_agent.engineering import EngineeringPolicy
from openfoam_agent.llm import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    CodexLLM,
    LLMConfigurationError,
    OllamaLLM,
    OpenAILLM,
    RuleBasedLLM,
    WorkflowLLMs,
    check_codex_cli,
    check_ollama_health,
    normalize_ollama_base_url,
)
from openfoam_agent.postprocessing import PostProcessingPolicy
from openfoam_agent.progress import CLIProgressReporter, ProgressLevel, ProgressReporter
from openfoam_agent.schemas.intake import CFDIntakeSpec
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.schemas.simulation import RuntimePolicy
from openfoam_agent.workflow.engine import CFDWorkflow
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTALLED_DATA_ROOT = Path(sysconfig.get_path("data")) / "share" / "openfoam-agent"


def _resource_path(source_path: Path, installed_path: Path) -> Path:
    return source_path if source_path.exists() else installed_path


DEFAULT_CAPABILITY_DB = _resource_path(
    PROJECT_ROOT / "config" / "openfoam14_capability_graph.json",
    INSTALLED_DATA_ROOT / "config" / "openfoam14_capability_graph.json",
)
DEFAULT_WORKSPACE = Path(tempfile.gettempdir()) / "openfoam-agent-v2"
SUCCESS_STATES = {
    State.INTAKE_REVIEW_REQUIRED,
    State.CASE_PREVIEW_READY,
    State.MESH_READY,
    State.SOLVE_READY,
    State.RESULT_REVIEW_REQUIRED,
    State.REVISION_READY,
    State.COMPLETE,
    State.DONE,
}
CLARIFICATION_EXIT_CODE = 2
_AUTONOMOUS_BACKENDS = frozenset({"openai", "ollama", "codex"})
_SETTABLE_FACT_PREFIXES = {
    "classification", "objective", "domain", "geometry", "scale", "material",
    "property", "physics", "temporal", "motion", "boundary", "output",
    "fidelity", "assumption",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openfoam-agent",
        description="OpenFOAM Agent v2: autonomous CFD engineering behind deterministic safety gates.",
    )
    parser.add_argument("prompt", nargs="?", help="One-shot CFD prompt.")
    parser.add_argument("--prompt", dest="prompt_option", help="Alternative one-shot prompt.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Conversational mode.")
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in InteractionMode),
        default=InteractionMode.GUIDED.value,
        help="easy authorizes exploratory completion; guided requires user wording; strict forbids it.",
    )
    parser.add_argument(
        "--confirm-intake",
        action="store_true",
        help="Confirm a review-ready intake and authorize bounded case preparation/mesh tools.",
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="One-shot only: after a passing mesh gate, approve bounded foamRun execution.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Allow agent file authoring but do not execute native OpenFOAM tools or solver.",
    )
    parser.add_argument(
        "--backend",
        choices=("rule-based-intake", "openai", "ollama", "codex"),
        default="rule-based-intake",
        help=(
            "rule-based-intake is an offline intake regression baseline only; "
            "autonomous engineering supports --backend openai, --backend ollama, or --backend codex."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Default model for every role. OpenAI otherwise uses OPENAI_MODEL; "
            "Ollama otherwise uses OLLAMA_MODEL or gemma4:31b; Codex otherwise uses "
            "CODEX_MODEL or the Codex CLI default. Role-specific flags override it."
        ),
    )
    parser.add_argument(
        "--intake-model",
        help="Override the model used for intake analysis for the selected backend.",
    )
    parser.add_argument(
        "--engineering-model",
        help=(
            "Override the model used for engineering, mesh/case repair, runtime repair, "
            "and confirmed revisions for the selected backend."
        ),
    )
    parser.add_argument(
        "--postprocess-model",
        help="Override the model used for post-processing for the selected backend.",
    )
    parser.add_argument(
        "--review-model",
        help="Override the model used for human-feedback review for the selected backend.",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=24_000,
        help=(
            "Per-response LLM output-token cap (default: 24000). "
            "This bounds rate-limit reservation and prevents one structured action "
            "from reserving the model's full output window."
        ),
    )
    parser.add_argument(
        "--confirm-api-calls",
        action="store_true",
        help=("Explicitly authorize cloud model calls for --backend openai or --backend codex. "
              "The CFD request and bounded engineering observations/log excerpts are sent to "
              "OpenAI/Codex; local absolute paths are redacted. Codex runs ephemeral/read-only."),
    )
    parser.add_argument(
        "--base-url",
        help=(
            "Ollama OpenAI-compatible base URL. Defaults to OLLAMA_BASE_URL or "
            "http://localhost:11434/v1. Only loopback URLs are accepted so remote Ollama "
            "stays behind SSH local port forwarding."
        ),
    )
    parser.add_argument(
        "--ollama-health-timeout",
        type=float,
        default=3.0,
        help="Startup Ollama /v1/models health-check timeout in seconds (default: 3).",
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--capability-db", type=Path, default=DEFAULT_CAPABILITY_DB)
    parser.add_argument(
        "--progress",
        choices=tuple(level.value for level in ProgressLevel),
        default=ProgressLevel.NORMAL.value,
        help=(
            "Live progress verbosity: quiet disables progress, normal shows major engineering/runtime/postprocess events, "
            "verbose shows every agent action and raw foamRun output."
        ),
    )
    parser.add_argument(
        "--engineering-steps",
        type=int,
        default=12,
        help="Initial autonomous engineering LLM-turn soft budget (default: 12 turns).",
    )
    parser.add_argument(
        "--engineering-hard-cap",
        type=int,
        default=24,
        help="Absolute engineering LLM-turn cap after progress-aware extensions (default: 24).",
    )
    parser.add_argument(
        "--engineering-extension",
        type=int,
        default=6,
        help="Progress-aware LLM-turn extension chunk size (default: 6 turns).",
    )
    parser.add_argument(
        "--finalization-steps",
        type=int,
        default=2,
        help="Plan-finalization-only actions after validated case preparation (default: 2).",
    )
    parser.add_argument(
        "--engineering-tool-budget",
        type=int,
        default=160,
        help=(
            "Maximum deterministic engineering actions executed across single actions and "
            "short sequences/execution plans in one engineering round (default: 160)."
        ),
    )
    parser.add_argument(
        "--native-command-budget",
        type=int,
        default=40,
        help="Maximum executed OpenFOAM validation/mesh commands across engineering (default: 40).",
    )
    parser.add_argument(
        "--mesh-repair-cycles",
        type=int,
        default=6,
        help="Maximum file-repair cycles triggered by failed mesh commands (default: 6).",
    )
    parser.add_argument(
        "--runtime-repair-cycles",
        type=int,
        default=3,
        help="Maximum autonomous foamRun failure-repair-retry cycles (default: 3).",
    )
    parser.add_argument(
        "--runtime-repair-steps",
        type=int,
        default=4,
        help="Maximum LLM turns inside each runtime repair cycle (default: 4).",
    )
    parser.add_argument(
        "--runtime-repair-tool-budget",
        type=int,
        default=48,
        help="Maximum deterministic actions inside each runtime repair cycle (default: 48).",
    )
    parser.add_argument(
        "--postprocess-steps",
        type=int,
        default=4,
        help="Maximum post-processing LLM plans after a successful solve (default: 4).",
    )
    parser.add_argument(
        "--postprocess-native-budget",
        type=int,
        default=8,
        help="Maximum foamPostProcess executions after a successful solve (default: 8).",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Stop at successful foamRun instead of launching the automatic post-processing agent.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str | None:
    prompt_sources = sum(value is not None for value in (args.prompt, args.prompt_option))
    if args.interactive and prompt_sources:
        parser.error("Choose either a one-shot prompt or --interactive, not both.")
    if not args.interactive and prompt_sources == 0:
        parser.error("Provide a prompt or use --interactive.")
    if prompt_sources > 1:
        parser.error("Provide the prompt either positionally or with --prompt, not both.")
    if args.interactive and args.output:
        parser.error("--output is one-shot only; use --json in interactive mode.")
    if args.interactive and args.confirm_intake:
        parser.error("Use /confirm in interactive mode.")
    if args.interactive and args.solve:
        parser.error("Use /solve in interactive mode.")
    if args.force and not args.output:
        parser.error("--force requires --output.")
    if args.solve and not args.confirm_intake:
        parser.error("--solve requires --confirm-intake.")
    if args.solve and args.dry_run:
        parser.error("--solve cannot be combined with --dry-run.")
    if args.backend == "rule-based-intake":
        role_model_flags = {
            "--model": args.model,
            "--intake-model": args.intake_model,
            "--engineering-model": args.engineering_model,
            "--postprocess-model": args.postprocess_model,
            "--review-model": args.review_model,
        }
        invalid_model_flags = [flag for flag, value in role_model_flags.items() if value]
        if invalid_model_flags:
            parser.error(
                f"{', '.join(invalid_model_flags)} require --backend openai, --backend ollama, or --backend codex."
            )
        if args.base_url:
            parser.error("--base-url is only valid with --backend ollama.")
        if args.confirm_api_calls:
            parser.error("--confirm-api-calls is only valid with --backend openai or --backend codex.")
        if args.confirm_intake:
            parser.error(
                "Autonomous engineering has no rule-based template fallback. "
                "Use --backend openai/codex with --confirm-api-calls, or --backend ollama."
            )
    elif args.backend == "openai":
        if args.base_url:
            parser.error("--base-url is only valid with --backend ollama.")
        if not args.confirm_api_calls:
            parser.error("--backend openai requires --confirm-api-calls.")
    elif args.backend == "codex":
        if args.base_url:
            parser.error("--base-url is only valid with --backend ollama.")
        if not args.confirm_api_calls:
            parser.error("--backend codex requires --confirm-api-calls because Codex is a cloud model backend.")
    elif args.backend == "ollama":
        if args.confirm_api_calls:
            parser.error(
                "--confirm-api-calls is for cloud OpenAI/Codex backends and is not used by Ollama."
            )
        try:
            normalize_ollama_base_url(
                args.base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
            )
        except LLMConfigurationError as exc:
            parser.error(str(exc))
    if args.llm_max_output_tokens < 1:
        parser.error("--llm-max-output-tokens must be >= 1.")
    if args.ollama_health_timeout <= 0:
        parser.error("--ollama-health-timeout must be > 0.")
    positive_budget_fields = {
        "--engineering-steps": args.engineering_steps,
        "--engineering-hard-cap": args.engineering_hard_cap,
        "--engineering-extension": args.engineering_extension,
        "--finalization-steps": args.finalization_steps,
        "--engineering-tool-budget": args.engineering_tool_budget,
        "--native-command-budget": args.native_command_budget,
        "--mesh-repair-cycles": args.mesh_repair_cycles,
        "--runtime-repair-steps": args.runtime_repair_steps,
        "--runtime-repair-tool-budget": args.runtime_repair_tool_budget,
        "--postprocess-steps": args.postprocess_steps,
        "--postprocess-native-budget": args.postprocess_native_budget,
    }
    for flag, value in positive_budget_fields.items():
        if value < 1:
            parser.error(f"{flag} must be >= 1.")
    if args.engineering_hard_cap < args.engineering_steps:
        parser.error("--engineering-hard-cap must be >= --engineering-steps.")
    if not 0 <= args.runtime_repair_cycles <= 12:
        parser.error("--runtime-repair-cycles must be between 0 and 12.")
    if not args.capability_db.is_file():
        parser.error(f"Capability database is missing: {args.capability_db}")
    selected = args.prompt_option if args.prompt_option is not None else args.prompt
    if selected is not None and not selected.strip():
        parser.error("Prompt must not be blank.")
    return selected.strip() if selected is not None else None


_ROLE_MODEL_ENV = {
    "openai": {
        "intake": "OPENAI_INTAKE_MODEL",
        "engineering": "OPENAI_ENGINEERING_MODEL",
        "postprocessing": "OPENAI_POSTPROCESS_MODEL",
        "review": "OPENAI_REVIEW_MODEL",
    },
    "ollama": {
        "intake": "OLLAMA_INTAKE_MODEL",
        "engineering": "OLLAMA_ENGINEERING_MODEL",
        "postprocessing": "OLLAMA_POSTPROCESS_MODEL",
        "review": "OLLAMA_REVIEW_MODEL",
    },
    "codex": {
        "intake": "CODEX_INTAKE_MODEL",
        "engineering": "CODEX_ENGINEERING_MODEL",
        "postprocessing": "CODEX_POSTPROCESS_MODEL",
        "review": "CODEX_REVIEW_MODEL",
    },
}


def _cleaned(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_backend_model_names(
    args: argparse.Namespace,
    *,
    backend: str,
    environ: dict[str, str] | None = None,
) -> tuple[str | None, dict[str, str]]:
    env = os.environ if environ is None else environ
    if backend not in {"openai", "ollama", "codex"}:
        raise LLMConfigurationError(f"Unsupported model backend: {backend}")

    default_env = {
        "openai": "OPENAI_MODEL",
        "ollama": "OLLAMA_MODEL",
        "codex": "CODEX_MODEL",
    }[backend]
    built_in_default = {
        "openai": None,
        "ollama": DEFAULT_OLLAMA_MODEL,
        "codex": DEFAULT_CODEX_MODEL,
    }[backend]
    default_model = _cleaned(args.model) or _cleaned(env.get(default_env)) or built_in_default
    cli_overrides = {
        "intake": _cleaned(args.intake_model),
        "engineering": _cleaned(args.engineering_model),
        "postprocessing": _cleaned(args.postprocess_model),
        "review": _cleaned(args.review_model),
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for role, env_name in _ROLE_MODEL_ENV[backend].items():
        model_name = cli_overrides[role] or _cleaned(env.get(env_name)) or default_model
        if model_name is None:
            missing.append(role)
        else:
            resolved[role] = model_name
    if missing:
        roles = ", ".join(missing)
        prefix = {"openai": "OPENAI", "ollama": "OLLAMA", "codex": "CODEX"}[backend]
        raise LLMConfigurationError(
            f"No {backend} model is configured for role(s): {roles}. "
            f"Set --model/{prefix}_MODEL or provide every missing role override."
        )
    return default_model, resolved


def _resolve_openai_model_names(
    args: argparse.Namespace,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str | None, dict[str, str]]:
    return _resolve_backend_model_names(args, backend="openai", environ=environ)


def _resolve_ollama_model_names(
    args: argparse.Namespace,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str | None, dict[str, str]]:
    return _resolve_backend_model_names(args, backend="ollama", environ=environ)


def _resolve_codex_model_names(
    args: argparse.Namespace,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str | None, dict[str, str]]:
    return _resolve_backend_model_names(args, backend="codex", environ=environ)


def _build_llm(args: argparse.Namespace):
    if args.backend == "rule-based-intake":
        return WorkflowLLMs.uniform(RuleBasedLLM()), "rule-based-intake", None

    if args.backend == "openai":
        default_model, names = _resolve_openai_model_names(args)
        clients: dict[str, OpenAILLM] = {}

        def client(model_name: str) -> OpenAILLM:
            if model_name not in clients:
                clients[model_name] = OpenAILLM(
                    model=model_name,
                    max_output_tokens=args.llm_max_output_tokens,
                )
            return clients[model_name]

        llms = WorkflowLLMs(
            intake=client(names["intake"]),
            engineering=client(names["engineering"]),
            postprocessing=client(names["postprocessing"]),
            review=client(names["review"]),
        )
        return llms, "openai", default_model

    if args.backend == "codex":
        default_model, names = _resolve_codex_model_names(args)
        status = check_codex_cli()
        clients: dict[str, CodexLLM] = {}

        def codex_client(model_name: str) -> CodexLLM:
            if model_name not in clients:
                clients[model_name] = CodexLLM(
                    model=None if model_name == DEFAULT_CODEX_MODEL else model_name,
                    status=status,
                )
            return clients[model_name]

        llms = WorkflowLLMs(
            intake=codex_client(names["intake"]),
            engineering=codex_client(names["engineering"]),
            postprocessing=codex_client(names["postprocessing"]),
            review=codex_client(names["review"]),
        )
        return llms, "codex", default_model

    default_model, names = _resolve_ollama_model_names(args)
    base_url = normalize_ollama_base_url(
        args.base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    )
    api_key = os.getenv("OLLAMA_API_KEY", DEFAULT_OLLAMA_API_KEY) or DEFAULT_OLLAMA_API_KEY
    check_ollama_health(
        base_url=base_url,
        models=tuple(names.values()),
        api_key=api_key,
        timeout=args.ollama_health_timeout,
    )
    clients: dict[str, OllamaLLM] = {}

    def ollama_client(model_name: str) -> OllamaLLM:
        if model_name not in clients:
            clients[model_name] = OllamaLLM(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                max_output_tokens=args.llm_max_output_tokens,
            )
        return clients[model_name]

    llms = WorkflowLLMs(
        intake=ollama_client(names["intake"]),
        engineering=ollama_client(names["engineering"]),
        postprocessing=ollama_client(names["postprocessing"]),
        review=ollama_client(names["review"]),
    )
    return llms, "ollama", default_model


def _model_routes(llm: Any) -> dict[str, str | None]:
    return WorkflowLLMs.coerce(llm).model_names()


def _policies_from_args(
    args: argparse.Namespace,
) -> tuple[EngineeringPolicy, RuntimePolicy, PostProcessingPolicy]:
    engineering = EngineeringPolicy(
        max_agent_steps=args.engineering_steps,
        hard_max_agent_steps=args.engineering_hard_cap,
        step_extension=args.engineering_extension,
        max_finalization_steps=args.finalization_steps,
        max_tool_actions=args.engineering_tool_budget,
        max_native_commands=args.native_command_budget,
        max_mesh_repair_cycles=args.mesh_repair_cycles,
        max_runtime_repair_steps=args.runtime_repair_steps,
        max_runtime_repair_tool_actions=args.runtime_repair_tool_budget,
        require_solve_ready_gate=True,
        preload_capabilities=True,
        compact_phase_schemas=True,
        state_delta_context=True,
    )
    runtime = RuntimePolicy(max_attempts=args.runtime_repair_cycles + 1)
    postprocessing = PostProcessingPolicy(
        max_steps=args.postprocess_steps,
        max_native_commands=args.postprocess_native_budget,
        compact_execution_plan=True,
        state_delta_context=True,
    )
    return engineering, runtime, postprocessing


def build_report(
    state: CFDState,
    *,
    backend: str,
    model: str | None,
    request: UserRequest,
    workspace: Path,
    engineering_policy: EngineeringPolicy,
    runtime_policy: RuntimePolicy,
    postprocessing_policy: PostProcessingPolicy,
    model_routes: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    round_events = state.engineering_events[state.engineering_round_start_index:]
    round_llm_turns = len({item.step for item in round_events})
    total_llm_turns = len({item.step for item in state.engineering_events})
    round_sequences = len(
        {item.sequence_id for item in round_events if item.sequence_id is not None}
    )
    tool_actions_per_llm_turn = (
        len(round_events) / round_llm_turns if round_llm_turns else 0.0
    )
    return {
        "architecture": "v2.10.0",
        "run_id": state.run_id,
        "prompt": request.prompt,
        "conversation_turns": list(request.conversation_turns),
        "interaction_mode": request.interaction_mode,
        "exploratory_completion_authorized": request.exploratory_completion_authorized,
        "backend": backend,
        "model": model,
        "model_routes": model_routes or {
            "intake": model,
            "engineering": model,
            "postprocessing": model,
            "review": model,
        },
        "final_state": state.current_state.value,
        "message": state.history[-1]["note"] if state.history else "",
        "workspace": str(workspace),
        "intake": state.intake.model_dump(mode="json") if state.intake else None,
        "intake_confirmed": state.intake_confirmed,
        "intake_sha256": state.intake_digest or (state.intake.digest() if state.intake else None),
        "engineering_plan": (
            state.engineering_plan.model_dump(mode="json") if state.engineering_plan else None
        ),
        "engineering_events": [item.model_dump(mode="json") for item in state.engineering_events],
        "budget": {
            # Backward-compatible aliases keep existing report consumers working.
            "engineering_actions_used": len(round_events),
            "engineering_actions_total": len(state.engineering_events),
            "engineering_llm_turns_used": round_llm_turns,
            "engineering_llm_turns_total": total_llm_turns,
            "engineering_tool_actions_used": len(round_events),
            "engineering_tool_actions_total": len(state.engineering_events),
            "engineering_sequences_used": round_sequences,
            "tool_actions_per_llm_turn": round(tool_actions_per_llm_turn, 3),
            "engineering_soft_limit": engineering_policy.max_agent_steps,
            "engineering_hard_limit": engineering_policy.hard_max_agent_steps,
            "engineering_tool_action_limit": engineering_policy.max_tool_actions,
            "engineering_extensions": [
                item.model_dump(mode="json") for item in state.engineering_budget_extensions
            ],
            "native_commands_executed": sum(
                1
                for item in state.engineering_events[state.engineering_round_start_index:]
                if item.native_command_executed
            ),
            "native_commands_total": sum(
                1 for item in state.engineering_events if item.native_command_executed
            ),
            "native_command_limit": engineering_policy.max_native_commands,
            "mesh_repair_cycle_limit": engineering_policy.max_mesh_repair_cycles,
            "runtime_repair_cycles_limit": runtime_policy.max_repair_cycles,
            "runtime_repair_steps_per_cycle": engineering_policy.max_runtime_repair_steps,
            "runtime_repair_tool_actions_per_cycle": engineering_policy.max_runtime_repair_tool_actions,
            "postprocess_actions_used": len(state.postprocessing_events),
            "postprocess_action_limit": postprocessing_policy.max_steps,
            "postprocess_native_commands_executed": sum(
                1 for item in state.postprocessing_events if item.native_command_executed
            ),
            "postprocess_native_command_limit": postprocessing_policy.max_native_commands,
        },
        "case_dir": state.case_dir,
        "case_seal": state.case_seal.model_dump(mode="json") if state.case_seal else None,
        "mesh_evidence": (
            state.mesh_evidence.model_dump(mode="json") if state.mesh_evidence else None
        ),
        "solve_approved": state.solve_approved,
        "runtime_report": (
            state.runtime_report.model_dump(mode="json") if state.runtime_report else None
        ),
        "postprocessing_report": (
            state.postprocessing_report.model_dump(mode="json")
            if state.postprocessing_report
            else None
        ),
        "postprocessing_events": [
            item.model_dump(mode="json") for item in state.postprocessing_events
        ],
        "human_feedback": [item.model_dump(mode="json") for item in state.human_feedback],
        "revision_proposals": [item.model_dump(mode="json") for item in state.revision_proposals],
        "active_revision_proposal": (
            state.active_revision_proposal.model_dump(mode="json")
            if state.active_revision_proposal
            else None
        ),
        "revision_history": [
            item.model_dump(mode="json") for item in state.revision_history
        ],
        "pending_revision_archive_path": state.pending_revision_archive_path,
        "history": list(state.history),
        "limitations": _limitations(state),
    }


def _limitations(state: CFDState) -> list[str]:
    out: list[str] = []
    if state.current_state == State.CASE_PREVIEW_READY:
        out.append("Native OpenFOAM validation/mesh tools were disabled; this is a file preview only.")
    if state.current_state == State.MESH_READY:
        out.append("checkMesh passed, but pre-solve completeness validation has not yet produced a solve-ready seal.")
    if state.current_state == State.SOLVE_READY:
        out.append("Pre-solve completeness validation passed; foamRun still requires explicit /solve approval.")
    if state.current_state == State.ENGINEERING_BLOCKED:
        out.append("The autonomous engineering/retry budget ended without a safely executable result.")
    if state.current_state == State.RESULT_REVIEW_REQUIRED:
        out.append(
            "Runtime/post-processing evidence is available, but human review is still required; use /accept or /feedback in interactive mode."
        )
    if state.current_state == State.COMPLETE:
        out.append(
            "COMPLETE records explicit human acceptance of the reviewed result; it is not a universal proof of mesh/time-step independence or experimental validation."
        )
    if state.current_state == State.DONE:
        out.append("DONE is retained only for backward compatibility; v2.4 uses RESULT_REVIEW_REQUIRED and COMPLETE.")
    return out


def _write_report(path: Path, report: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; use --force.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _print_human_report(report: dict[str, Any]) -> None:
    print(f"state: {report['final_state']}")
    intake = report["intake"]
    if intake:
        print(f"intake: {intake['title']} [{intake['status']}]")
        review_critical: list[str] = []
        for fact in intake["facts"]:
            if fact["category"] != "context":
                unit = f" {fact['unit']}" if fact["unit"] else ""
                source_label = fact["source"]
                if fact["source"] == "derived" and fact["category"] in {
                    "classification",
                    "temporal",
                    "motion",
                    "boundary",
                }:
                    source_label = "derived/review-critical"
                    reason = (fact.get("reason") or "derived interpretation").strip()
                    review_critical.append(f"{fact['id']}={fact['value']}{unit} — {reason}")
                print(f"- {fact['id']}: {fact['value']}{unit} [{source_label}]")
        if review_critical and report["final_state"] == State.INTAKE_REVIEW_REQUIRED.value:
            print("review-critical derived interpretations (these become immutable on /confirm):")
            for item in review_critical:
                print(f"- {item}")
        if intake["blocking_unknowns"]:
            print("blocking questions:")
            for item in intake["blocking_unknowns"]:
                print(f"- {item['question']}")
    plan = report["engineering_plan"]
    if plan:
        print(f"solver: {plan['solver']}")
        print(f"mesh: {plan['mesh_strategy']}")
        print(
            f"semantics: {plan['temporal_behavior']} / {plan['motion_kind']} / "
            f"{plan['mesh_motion_requirement']}"
        )
        if plan["assumptions"]:
            print("engineering assumptions:")
            for assumption in plan["assumptions"]:
                print(f"- {assumption}")
        if intake and intake.get("semantic_contract_version") == "2":
            bindings = plan.get("confirmed_fact_bindings") or []
            machine_asserted = sum(
                1
                for item in bindings
                if item.get("case_assertions") or item.get("numeric_relation") is not None
            )
            numeric_relations = sum(
                1 for item in bindings if item.get("numeric_relation") is not None
            )
            print(
                "semantic fidelity: contract=v2, "
                f"machineAssertedFacts={machine_asserted}/{len(bindings)}, "
                f"numericRelations={numeric_relations}"
            )
    budget = report.get("budget")
    if budget:
        extensions = budget["engineering_extensions"]
        extension_text = (
            ", ".join(f"{item['previous_limit']}->{item['new_limit']}" for item in extensions)
            if extensions
            else "none"
        )
        print(
            "engineering budget: "
            f"llmTurns={budget.get('engineering_llm_turns_used', budget['engineering_actions_used'])}, "
            f"toolActions={budget.get('engineering_tool_actions_used', budget['engineering_actions_used'])}, "
            f"actionsPerTurn={budget.get('tool_actions_per_llm_turn', 1.0)}, "
            f"native={budget['native_commands_executed']}/{budget['native_command_limit']}, "
            f"extensions={extension_text}"
        )
    if report["case_dir"]:
        print(f"case: {report['case_dir']}")
    mesh = report["mesh_evidence"]
    if mesh:
        print(
            "checkMesh: "
            f"passed={mesh['command_succeeded'] and mesh['mesh_ok']}, "
            f"cells={mesh['cell_count']}, maxNonOrtho={mesh['max_non_orthogonality']}, "
            f"maxSkew={mesh['max_skewness']}"
        )
    runtime = report["runtime_report"]
    if runtime:
        final = runtime["final_result"]
        print(
            f"runtime: success={runtime['success']}, attempts={len(runtime['attempts'])}, "
            f"lastTime={final['last_time']}, maxCo={final['courant_max']}"
        )
    post = report.get("postprocessing_report")
    if post:
        print(
            "postprocess: "
            f"success={post['success']}, actions={post['actions_executed']}, "
            f"native={post['native_commands_executed']}"
        )
        analysis = post.get("force_analysis")
        if analysis:
            print(
                "forces: "
                f"samples={analysis['samples_used']}/{analysis['samples_total']}, "
                f"meanCd={analysis['mean_cd']}, rmsCl={analysis['rms_cl']}, "
                f"f={analysis['shedding_frequency']}, St={analysis['strouhal_number']}"
            )
        if post.get("artifacts"):
            print("result artifacts:")
            for artifact in post["artifacts"]:
                print(f"- {artifact['kind']}: {artifact['path']}")
        print(f"scientific confidence (agent assessment): {post.get('scientific_confidence', 'unknown')}")
        if post.get("review_reasons"):
            print("review reasons:")
            for item in post["review_reasons"]:
                print(f"- {item}")
        if post.get("recommended_human_checks"):
            print("recommended human checks:")
            for item in post["recommended_human_checks"]:
                print(f"- {item}")
        if post.get("limitations"):
            print("postprocess limitations:")
            for item in post["limitations"]:
                print(f"- {item}")
        if report.get("case_dir"):
            print(f"visualize: cd {report['case_dir']} && paraFoam")
    feedback = report.get("human_feedback") or []
    if feedback:
        print("human feedback:")
        for item in feedback:
            print(f"- {item['feedback_id']} [{item['status']}/{item['scope']}]: {item['statement']}")
    proposal = report.get("active_revision_proposal")
    if proposal:
        print(f"revision proposal: {proposal['proposal_id']} (cost={proposal['expected_cost']})")
        print(f"- diagnosis: {proposal['diagnosis_summary']}")
        for change in proposal.get("proposed_changes", []):
            print(f"- change[{change['area']}]: {change['change']}")
        if proposal.get("review_limitations"):
            print("- review limitations:")
            for item in proposal["review_limitations"]:
                print(f"  - {item}")
    revisions = report.get("revision_history") or []
    if revisions:
        latest = revisions[-1]
        print(
            f"revision diff: {latest['revision_id']} proposal={latest['proposal_id']} "
            f"files_changed={len(latest['file_changes'])}"
        )
        for item in latest["file_changes"][:30]:
            print(f"- {item['change']}: {item['path']}")
        if latest.get("archive_path"):
            print(f"revision archive: {latest['archive_path']}")
    if report.get("pending_revision_archive_path"):
        print(f"pending revision archive: {report['pending_revision_archive_path']}")
    if report["limitations"]:
        print("limitations:")
        for item in report["limitations"]:
            print(f"- {item}")
    if report["message"]:
        print(f"message: {report['message']}")
    if report["final_state"] == State.INTAKE_REVIEW_REQUIRED.value:
        print("next: /confirm in interactive mode, or --confirm-intake with --backend openai/ollama/codex")
    if report["final_state"] == State.SOLVE_READY.value:
        print("next: /solve to approve foamRun, or /feedback <observation> to revise the mesh/case")
    elif report["final_state"] == State.MESH_READY.value:
        print("next: pre-solve completeness validation is still required before /solve")
    if report["final_state"] == State.RESULT_REVIEW_REQUIRED.value:
        print("next: /accept to complete, or /feedback <observation> to request a revision")
    if report["final_state"] == State.REVISION_READY.value:
        print("next: /confirm to authorize the proposed revision, or /reject to keep the current sealed case")
    print(f"run_id: {report['run_id']}")


def _exit_code(state: State) -> int:
    if state in SUCCESS_STATES:
        return 0
    if state in {State.NEEDS_CLARIFICATION, State.ENGINEERING_REVIEW_REQUIRED}:
        return CLARIFICATION_EXIT_CODE
    return 1


def run_prompt(
    prompt: str | UserRequest,
    *,
    llm: Any,
    backend: str,
    model: str | None,
    capability_db: Path,
    workspace_root: Path,
    run_id: str | None = None,
    attempt: int | None = None,
    confirmed_intake: CFDIntakeSpec | None = None,
    native_execution: bool = True,
    execute_solver: bool = False,
    stream_solver_output: bool = False,
    engineering_policy: EngineeringPolicy | None = None,
    runtime_policy: RuntimePolicy | None = None,
    postprocessing_policy: PostProcessingPolicy | None = None,
    postprocessing_enabled: bool = True,
    progress: ProgressReporter | None = None,
) -> tuple[CFDState, dict[str, Any]]:
    request = prompt if isinstance(prompt, UserRequest) else UserRequest(prompt=prompt)
    engineering_policy = engineering_policy or EngineeringPolicy()
    runtime_policy = runtime_policy or RuntimePolicy()
    postprocessing_policy = postprocessing_policy or PostProcessingPolicy()
    selected_run_id = run_id or str(uuid.uuid4())
    run_workspace = workspace_root.expanduser().resolve() / selected_run_id
    if attempt is not None:
        run_workspace /= f"attempt-{attempt:03d}"
    state = CFDState(run_id=selected_run_id, user_request=request)
    if confirmed_intake is not None:
        state.intake = confirmed_intake
        state.confirm_intake()
        state.current_state = State.ENGINEERING
        state.history.append(
            {
                "from": State.INTAKE_REVIEW_REQUIRED.value,
                "to": State.ENGINEERING.value,
                "note": f"User confirmed immutable CFD intake {state.intake_digest}.",
            }
        )
    workflow = CFDWorkflow(
        llm=llm,
        capability_db=capability_db,
        workspace=run_workspace,
        native_execution=native_execution,
        stream_solver_output=stream_solver_output,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
        postprocessing_enabled=postprocessing_enabled,
        progress=progress,
    )
    final_state = workflow.run(state)
    if execute_solver and final_state.current_state == State.MESH_READY:
        final_state.approve_solve()
        final_state = workflow.run(final_state)
    report = build_report(
        final_state,
        backend=backend,
        model=model,
        request=request,
        model_routes=_model_routes(llm),
        workspace=run_workspace,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
    )
    return final_state, report


def _emit_report(report: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human_report(report)


def _print_interactive_help() -> None:
    print("commands:")
    print("- /show                 누적 요청과 승인 정책 표시")
    print("- /details              현재 CFDIntakeSpec JSON")
    print("- /confirm              intake 확정 + bounded case/mesh engineering 승인")
    print("- /solve                MESH_READY sealed case의 foamRun 승인")
    print("- /feedback <text>      mesh/result에 대한 human engineering feedback 제출")
    print("- /accept               RESULT_REVIEW_REQUIRED 결과를 최종 수락")
    print("- /reject               REVISION_READY proposal을 거절하고 이전 review 상태로 복귀")
    print("- /set <fact>=<value>   사용자 사실 추가 후 intake 재작성")
    print("- /edit <text>          마지막 사용자 턴 교체")
    print("- /undo                 마지막 사용자 턴 제거")
    print("- /run                  intake 재분석")
    print("- /mode easy|guided|strict")
    print("- /new                  새 세션")
    print("- /exit                  종료")


def _print_session(session: ConversationSession) -> None:
    summary = session.summary()
    print(f"session: {summary['session_id']}")
    print(f"mode: {summary['mode']}")
    print(f"exploratory completion: {summary['exploratory_completion_authorized']}")
    for index, turn in enumerate(summary["turns"], start=1):
        print(f"{index}. {turn}")
    if summary["last_state"]:
        print(f"last state: {summary['last_state']}")


def _progress_from_args(args: argparse.Namespace) -> CLIProgressReporter:
    return CLIProgressReporter(args.progress)


def _run_session(session, args, llm, backend, model):
    attempt = session.next_attempt()
    engineering_policy, runtime_policy, postprocessing_policy = _policies_from_args(args)
    state, report = run_prompt(
        session.to_request(),
        llm=llm,
        backend=backend,
        model=model,
        capability_db=args.capability_db,
        workspace_root=args.workspace,
        run_id=session.session_id,
        attempt=attempt,
        native_execution=not args.dry_run,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
        postprocessing_enabled=not args.skip_postprocess,
        progress=_progress_from_args(args),
    )
    session.last_state = state.current_state.value
    session.set_pending_intake(state.intake)
    report["conversation"] = session.summary()
    _emit_report(report, as_json=args.json)
    if not args.json:
        print()
    return state, report


def _confirm_session(session, args, llm, backend, model) -> None:
    if (
        session.pending_workflow_state is not None
        and session.pending_workflow_state.current_state == State.REVISION_READY
    ):
        _confirm_revision_session(session, args, llm, backend, model)
        return
    intake = session.pending_intake
    if intake is None:
        print("확정할 CFD intake가 없습니다.")
        return
    if intake.status != "ready_for_review":
        print("blocking 질문에 답변한 뒤 확정할 수 있습니다.")
        return
    if backend not in _AUTONOMOUS_BACKENDS:
        print("v2 autonomous engineering requires --backend openai, --backend ollama, or --backend codex.")
        return
    attempt = session.next_attempt()
    engineering_policy, runtime_policy, postprocessing_policy = _policies_from_args(args)
    state, report = run_prompt(
        session.to_request(),
        llm=llm,
        backend=backend,
        model=model,
        capability_db=args.capability_db,
        workspace_root=args.workspace,
        run_id=session.session_id,
        attempt=attempt,
        confirmed_intake=intake,
        native_execution=not args.dry_run,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
        postprocessing_enabled=not args.skip_postprocess,
        progress=_progress_from_args(args),
    )
    session.last_state = state.current_state.value
    session.confirmed_intake_digest = intake.digest()
    session.pending_workflow_state = state
    report["conversation"] = session.summary()
    _emit_report(report, as_json=args.json)
    if not args.json:
        print()


def _feedback_session(session, args, llm, backend, model, feedback_text: str) -> None:
    state = session.pending_workflow_state
    if state is None or state.current_state not in {State.MESH_READY, State.RESULT_REVIEW_REQUIRED}:
        print("/feedback은 MESH_READY 또는 RESULT_REVIEW_REQUIRED에서 사용할 수 있습니다.")
        return
    if backend not in _AUTONOMOUS_BACKENDS:
        print("human-feedback diagnosis requires --backend openai, --backend ollama, or --backend codex.")
        return
    text = feedback_text.strip()
    if not text:
        print("사용법: /feedback <mesh/result에 대한 관찰 또는 우려>")
        return
    if state.case_dir is None:
        print("feedback을 연결할 sealed case directory가 없습니다.")
        return
    run_workspace = Path(state.case_dir).resolve().parent
    engineering_policy, runtime_policy, postprocessing_policy = _policies_from_args(args)
    workflow = CFDWorkflow(
        llm=llm,
        capability_db=args.capability_db,
        workspace=run_workspace,
        native_execution=not args.dry_run,
        stream_solver_output=False,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
        postprocessing_enabled=not args.skip_postprocess,
        progress=_progress_from_args(args),
    )
    try:
        final_state = workflow.review.review(state, text)
    except Exception as exc:
        print(f"feedback review failed: {type(exc).__name__}: {exc}")
        return
    session.last_state = final_state.current_state.value
    session.pending_workflow_state = final_state
    report = build_report(
        final_state,
        backend=backend,
        model=model,
        request=session.to_request(),
        model_routes=_model_routes(llm),
        workspace=run_workspace,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
    )
    report["conversation"] = session.summary()
    _emit_report(report, as_json=args.json)
    if not args.json:
        print()


def _confirm_revision_session(session, args, llm, backend, model) -> None:
    state = session.pending_workflow_state
    if state is None or state.current_state != State.REVISION_READY:
        print("확정할 human-feedback revision proposal이 없습니다.")
        return
    if backend not in _AUTONOMOUS_BACKENDS:
        print("autonomous revision requires --backend openai, --backend ollama, or --backend codex.")
        return
    if state.case_dir is None:
        print("수정할 sealed case directory가 없습니다.")
        return
    run_workspace = Path(state.case_dir).resolve().parent
    engineering_policy, runtime_policy, postprocessing_policy = _policies_from_args(args)
    workflow = CFDWorkflow(
        llm=llm,
        capability_db=args.capability_db,
        workspace=run_workspace,
        native_execution=not args.dry_run,
        stream_solver_output=False,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
        postprocessing_enabled=not args.skip_postprocess,
        progress=_progress_from_args(args),
    )
    try:
        final_state = workflow.engineering.revise_from_feedback(
            state, native_execution=not args.dry_run
        )
    except Exception as exc:
        # A revision may already have archived the previous outputs and begun
        # editing the active case.  Never leave that partially executed state
        # looking like a live ENGINEERING session with no recovery information.
        if state.current_state == State.ENGINEERING:
            archive_note = (
                f" Prior baseline/output archive: {state.pending_revision_archive_path}."
                if state.pending_revision_archive_path
                else ""
            )
            state.transition(
                State.ENGINEERING_BLOCKED,
                f"Human-feedback revision aborted after an unexpected {type(exc).__name__}.{archive_note}",
            )
        session.last_state = state.current_state.value
        session.pending_workflow_state = state
        report = build_report(
            state,
            backend=backend,
            model=model,
            request=session.to_request(),
            model_routes=_model_routes(llm),
            workspace=run_workspace,
            engineering_policy=engineering_policy,
            runtime_policy=runtime_policy,
            postprocessing_policy=postprocessing_policy,
        )
        report["conversation"] = session.summary()
        _emit_report(report, as_json=args.json)
        if not args.json:
            print(f"revision engineering failed: {type(exc).__name__}: {exc}")
            print()
        return
    session.last_state = final_state.current_state.value
    session.pending_workflow_state = final_state
    report = build_report(
        final_state,
        backend=backend,
        model=model,
        request=session.to_request(),
        model_routes=_model_routes(llm),
        workspace=run_workspace,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
    )
    report["conversation"] = session.summary()
    _emit_report(report, as_json=args.json)
    if not args.json:
        print()


def _reject_revision_session(session, args, llm, backend, model) -> None:
    state = session.pending_workflow_state
    if state is None or state.current_state != State.REVISION_READY:
        print("/reject는 REVISION_READY에서만 사용할 수 있습니다.")
        return
    state.reject_revision()
    session.last_state = state.current_state.value
    run_workspace = Path(state.case_dir).resolve().parent if state.case_dir else args.workspace
    engineering_policy, runtime_policy, postprocessing_policy = _policies_from_args(args)
    report = build_report(
        state,
        backend=backend,
        model=model,
        request=session.to_request(),
        model_routes=_model_routes(llm),
        workspace=run_workspace,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
    )
    report["conversation"] = session.summary()
    _emit_report(report, as_json=args.json)
    if not args.json:
        print()


def _accept_session(session, args, llm, backend, model) -> None:
    state = session.pending_workflow_state
    if state is None or state.current_state != State.RESULT_REVIEW_REQUIRED:
        print("/accept는 RESULT_REVIEW_REQUIRED에서만 사용할 수 있습니다.")
        return
    state.accept_result()
    session.last_state = state.current_state.value
    run_workspace = Path(state.case_dir).resolve().parent if state.case_dir else args.workspace
    engineering_policy, runtime_policy, postprocessing_policy = _policies_from_args(args)
    report = build_report(
        state,
        backend=backend,
        model=model,
        request=session.to_request(),
        model_routes=_model_routes(llm),
        workspace=run_workspace,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
    )
    report["conversation"] = session.summary()
    _emit_report(report, as_json=args.json)
    if not args.json:
        print()


def _solve_session(session, args, llm, backend, model) -> None:
    if args.dry_run:
        print("--dry-run 세션에서는 /solve를 사용할 수 없습니다.")
        return
    state = session.pending_workflow_state
    if state is None or state.current_state != State.SOLVE_READY:
        print("/solve를 실행하려면 먼저 /confirm으로 SOLVE_READY에 도달해야 합니다.")
        return
    if state.case_dir is None:
        print("실행할 sealed case directory가 없습니다.")
        return
    state.approve_solve()
    run_workspace = Path(state.case_dir).resolve().parent
    engineering_policy, runtime_policy, postprocessing_policy = _policies_from_args(args)
    workflow = CFDWorkflow(
        llm=llm,
        capability_db=args.capability_db,
        workspace=run_workspace,
        native_execution=True,
        stream_solver_output=args.progress == ProgressLevel.VERBOSE.value,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
        postprocessing_enabled=not args.skip_postprocess,
        progress=_progress_from_args(args),
    )
    final_state = workflow.run(state)
    session.last_state = final_state.current_state.value
    session.pending_workflow_state = final_state
    report = build_report(
        final_state,
        backend=backend,
        model=model,
        request=session.to_request(),
        model_routes=_model_routes(llm),
        workspace=run_workspace,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
    )
    report["conversation"] = session.summary()
    _emit_report(report, as_json=args.json)
    if not args.json:
        print()


def _handle_command(command, session, args, llm, backend, model) -> bool:
    name, _, value = command.partition(" ")
    name = name.casefold()
    value = value.strip()
    if name in {"/exit", "/quit"}:
        return False
    if name == "/help":
        _print_interactive_help()
    elif name == "/show":
        _print_session(session)
    elif name == "/details":
        if session.pending_intake is None:
            print("표시할 CFD intake가 없습니다.")
        else:
            print(session.pending_intake.model_dump_json(indent=2))
    elif name == "/confirm":
        _confirm_session(session, args, llm, backend, model)
    elif name == "/solve":
        _solve_session(session, args, llm, backend, model)
    elif name == "/feedback":
        _feedback_session(session, args, llm, backend, model, value)
    elif name == "/accept":
        _accept_session(session, args, llm, backend, model)
    elif name == "/reject":
        _reject_revision_session(session, args, llm, backend, model)
    elif name == "/set":
        field, separator, field_value = value.partition("=")
        if not separator or not field.strip() or not field_value.strip():
            print("사용법: /set <fact>=<value>")
        elif not re.fullmatch(r"[a-z][a-z0-9_.-]*", field.strip()) or (
            field.strip().split(".", maxsplit=1)[0] not in _SETTABLE_FACT_PREFIXES
        ):
            print("지원되지 않는 fact ID입니다.")
        else:
            session.add_turn(f"Set {field.strip()} to {field_value.strip()} as a user-provided value.")
            _run_session(session, args, llm, backend, model)
    elif name == "/mode":
        if not value:
            print(f"current mode: {session.mode.value}")
        else:
            try:
                session.mode = InteractionMode(value.casefold())
            except ValueError:
                print("mode는 easy, guided, strict 중 하나여야 합니다.")
            else:
                print(f"mode: {session.mode.value}")
    elif name == "/undo":
        try:
            print(f"제거됨: {session.undo_last()}")
        except ValueError as exc:
            print(exc)
    elif name == "/edit":
        try:
            session.edit_last(value)
        except ValueError as exc:
            print(exc)
        else:
            _run_session(session, args, llm, backend, model)
    elif name == "/run":
        if not session.turns:
            print("먼저 CFD 프롬프트를 입력하세요.")
        else:
            _run_session(session, args, llm, backend, model)
    elif name == "/new":
        session.reset()
        print(f"새 세션: {session.session_id}")
    else:
        print("알 수 없는 명령입니다. /help를 사용하세요.")
    return True


def _interactive(args, llm, backend, model) -> int:
    session = ConversationSession(mode=InteractionMode(args.mode))
    print(f"OpenFOAM Agent v{__version__} (mode={session.mode.value}; progress={args.progress}; /help for commands)")
    if backend == "openai":
        routes = _model_routes(llm)
        print(
            "OpenAI model routing: "
            f"intake={routes['intake']}, engineering={routes['engineering']}, "
            f"postprocess={routes['postprocessing']}, review={routes['review']}; "
            "runtime-repair/revision use the engineering model. "
            "Cloud agent API calls authorized (task data/tool observations are transmitted; "
            "local paths are redacted)."
        )
    elif backend == "codex":
        routes = _model_routes(llm)
        status = getattr(WorkflowLLMs.coerce(llm).intake, "status", None)
        version = getattr(status, "version", "codex")
        print(
            "Codex CLI model routing: "
            f"intake={routes['intake']}, engineering={routes['engineering']}, "
            f"postprocess={routes['postprocessing']}, review={routes['review']}; "
            f"cli={version}; runtime-repair/revision use the engineering model. "
            "ChatGPT/Codex login is used; API-key routing variables are ignored. "
            "Each model call is ephemeral in an isolated read-only working directory."
        )
    elif backend == "ollama":
        routes = _model_routes(llm)
        base_url = getattr(WorkflowLLMs.coerce(llm).intake, "base_url", DEFAULT_OLLAMA_BASE_URL)
        print(
            "Ollama local model routing: "
            f"intake={routes['intake']}, engineering={routes['engineering']}, "
            f"postprocess={routes['postprocessing']}, review={routes['review']}; "
            f"base_url={base_url}; runtime-repair/revision use the engineering model. "
            "No OpenAI fallback is enabled; the endpoint must be reachable through the SSH tunnel."
        )
    while True:
        try:
            prompt = input("OpenFOAM Agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.startswith("/"):
            if not _handle_command(prompt, session, args, llm, backend, model):
                return 0
            continue
        pending = session.pending_workflow_state
        if pending is not None and pending.current_state in {
            State.MESH_READY,
            State.SOLVE_READY,
            State.RESULT_REVIEW_REQUIRED,
            State.REVISION_READY,
        }:
            if pending.current_state == State.RESULT_REVIEW_REQUIRED:
                print("현재 result review 상태입니다. /feedback <내용> 또는 /accept를 사용하세요. 새 문제는 /new 후 입력하세요.")
            elif pending.current_state == State.SOLVE_READY:
                print("현재 solve-ready review 상태입니다. /feedback <내용> 또는 /solve를 사용하세요. 새 문제는 /new 후 입력하세요.")
            elif pending.current_state == State.MESH_READY:
                print("현재 mesh-ready 상태지만 pre-solve validation이 남아 있습니다. /feedback <내용>을 사용하거나 /confirm 흐름을 완료하세요.")
            else:
                print("revision proposal이 승인 대기 중입니다. /confirm으로 승인하거나 /reject로 거절하세요.")
            continue
        session.add_turn(prompt)
        _run_session(session, args, llm, backend, model)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt = _validate_args(args, parser)
    try:
        llm, backend, model = _build_llm(args)
    except LLMConfigurationError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(f"Could not configure backend: {exc}")
    if args.interactive:
        return _interactive(args, llm, backend, model)

    engineering_policy, runtime_policy, postprocessing_policy = _policies_from_args(args)
    assert prompt is not None
    session = ConversationSession(mode=InteractionMode(args.mode))
    session.add_turn(prompt)
    request = session.to_request()
    state, report = run_prompt(
        request,
        llm=llm,
        backend=backend,
        model=model,
        capability_db=args.capability_db,
        workspace_root=args.workspace,
        native_execution=not args.dry_run,
        engineering_policy=engineering_policy,
        runtime_policy=runtime_policy,
        postprocessing_policy=postprocessing_policy,
        postprocessing_enabled=not args.skip_postprocess,
        progress=_progress_from_args(args),
    )
    if args.confirm_intake and state.current_state == State.INTAKE_REVIEW_REQUIRED:
        assert state.intake is not None
        state, report = run_prompt(
            request,
            llm=llm,
            backend=backend,
            model=model,
            capability_db=args.capability_db,
            workspace_root=args.workspace,
            run_id=state.run_id,
            confirmed_intake=state.intake,
            native_execution=not args.dry_run,
            execute_solver=args.solve,
            stream_solver_output=args.solve and args.progress == ProgressLevel.VERBOSE.value,
            engineering_policy=engineering_policy,
            runtime_policy=runtime_policy,
            postprocessing_policy=postprocessing_policy,
            postprocessing_enabled=not args.skip_postprocess,
            progress=_progress_from_args(args),
        )
    if args.output:
        try:
            _write_report(args.output, report, force=args.force)
        except FileExistsError as exc:
            parser.error(str(exc))
    _emit_report(report, as_json=args.json)
    return _exit_code(state.current_state)


if __name__ == "__main__":
    raise SystemExit(main())
