import argparse
from copy import deepcopy
import json
import operator
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from threading import Thread
from typing import Annotated, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


_WARMED_UP_IPEX_RUNTIMES: set[tuple[str, bool, bool, bool]] = set()
_FIRST_GENERATE_IPEX_RUNTIMES: set[tuple[str, bool, bool, bool]] = set()
_IPEX_UR_ERROR_MARKER = "IPEX XPU generation failed with 'UR error'"
_IPEX_FALLBACK_ACTIVE = False
_IPEX_FALLBACK_REASON: Optional[str] = None
_IPEX_FALLBACK_ANNOUNCED = False


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    ollama_model: str
    hf_model_id: str
    load_in_4bit: bool
    cpu_embedding: bool
    optimize_model: bool
    max_new_tokens: int


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_config() -> RuntimeConfig:
    backend = os.getenv("MAS_LLM_BACKEND", "ollama").strip().lower()
    if backend not in {"ollama", "ipex"}:
        raise ValueError(
            "MAS_LLM_BACKEND must be either 'ollama' or 'ipex'."
        )

    return RuntimeConfig(
        backend=backend,
        ollama_model=os.getenv("MAS_OLLAMA_MODEL", "llama3.1:8b"),
        hf_model_id=os.getenv("MAS_HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
        load_in_4bit=_env_flag("MAS_IPEX_LOAD_IN_4BIT", True),
        cpu_embedding=_env_flag("MAS_IPEX_CPU_EMBEDDING", True),
        optimize_model=_env_flag("MAS_IPEX_OPTIMIZE_MODEL", False),
        max_new_tokens=int(os.getenv("MAS_MAX_NEW_TOKENS", "768")),
    )


def apply_runtime_overrides(
    backend: Optional[str] = None,
    ollama_model: Optional[str] = None,
    hf_model_id: Optional[str] = None,
    load_in_4bit: Optional[bool] = None,
    cpu_embedding: Optional[bool] = None,
    optimize_model: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
) -> None:
    if backend is not None:
        os.environ["MAS_LLM_BACKEND"] = backend
    if ollama_model is not None:
        os.environ["MAS_OLLAMA_MODEL"] = ollama_model
    if hf_model_id is not None:
        os.environ["MAS_HF_MODEL_ID"] = hf_model_id
    if load_in_4bit is not None:
        os.environ["MAS_IPEX_LOAD_IN_4BIT"] = str(load_in_4bit).lower()
    if cpu_embedding is not None:
        os.environ["MAS_IPEX_CPU_EMBEDDING"] = str(cpu_embedding).lower()
    if optimize_model is not None:
        os.environ["MAS_IPEX_OPTIMIZE_MODEL"] = str(optimize_model).lower()
    if max_new_tokens is not None:
        os.environ["MAS_MAX_NEW_TOKENS"] = str(max_new_tokens)


def _normalize_prompt(messages: List[Any]) -> List[dict[str, str]]:
    normalized: List[dict[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "user"

        normalized.append({"role": role, "content": str(message.content)})
    return normalized


def _select_safe_pad_token_id(tokenizer: Any) -> int:
    for token_id in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        0,
    ):
        if token_id is not None:
            return int(token_id)
    return 0


def _print_ipex_progress(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] IPEX progress: {message}")


def _print_runtime_notice(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Runtime: {message}")


def _export_mermaid_graph(app: Any) -> str:
    mermaid_graph = app.get_graph().draw_mermaid()
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "MAS_orch_graph.mmd",
    )
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(mermaid_graph)
        handle.write("\n")
    _print_runtime_notice(f"Saved Mermaid graph to {output_path}")
    return mermaid_graph


def _ipex_runtime_key(config: RuntimeConfig) -> tuple[str, bool, bool, bool]:
    return (
        config.hf_model_id,
        config.load_in_4bit,
        config.cpu_embedding,
        config.optimize_model,
    )


@lru_cache(maxsize=1)
def _load_ipex_runtime(
    model_id: str,
    load_in_4bit: bool,
    cpu_embedding: bool,
    optimize_model: bool,
):
    os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")

    try:
        import torch
        from ipex_llm.transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "IPEX backend requested, but required packages are missing. "
            "Install with: pip install --pre --upgrade \"ipex-llm[xpu_2.6]\" "
            "--extra-index-url https://download.pytorch.org/whl/xpu"
        ) from exc

    _print_ipex_progress(f"loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token_id = _select_safe_pad_token_id(tokenizer)
    _print_ipex_progress(f"tokenizer ready; pad_token_id={tokenizer.pad_token_id}")

    _print_ipex_progress(
        f"loading model weights and running low-bit conversion for {model_id}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=load_in_4bit,
        cpu_embedding=cpu_embedding,
        optimize_model=optimize_model,
        trust_remote_code=True,
        use_cache=True,
        attn_implementation="eager",
    ).to("xpu")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    _print_ipex_progress("model load and XPU transfer completed")

    return torch, tokenizer, model


def _build_ipex_generation_kwargs(
    model: Any,
    tokenizer: Any,
    *,
    do_sample: bool,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    generation_config = deepcopy(model.generation_config)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.do_sample = do_sample
    generation_config.max_new_tokens = max_new_tokens

    if do_sample:
        if temperature is not None:
            generation_config.temperature = temperature
    else:
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50

    kwargs: dict[str, Any] = {"generation_config": generation_config}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def _prepare_ipex_inputs(tokenizer: Any, prompt: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to("xpu") for key, value in encoded.items()}


def _wrap_ipex_runtime_error(exc: BaseException, config: RuntimeConfig) -> RuntimeError:
    message = str(exc).strip() or exc.__class__.__name__
    if "UR error" in message:
        return RuntimeError(
            f"{_IPEX_UR_ERROR_MARKER} after successful model load. "
            "On this Windows Lenovo T16 with Iris Xe, the direct Python IPEX path is currently unstable. "
            "Use the Ollama backend for normal runs, ideally with the ollama-ipex-llm build if you want Intel GPU acceleration in the server. "
            f"Current local model: {config.hf_model_id}."
        )
    return RuntimeError(message)


def _build_ollama_client(config: RuntimeConfig, temperature: float, json_mode: bool = False):
    options = {"model": config.ollama_model, "temperature": temperature}
    if json_mode:
        options["format"] = "json"
    return ChatOllama(**options)


def _is_ipex_ur_error(exc: BaseException) -> bool:
    return _IPEX_UR_ERROR_MARKER in (str(exc).strip() or exc.__class__.__name__)


def _activate_ipex_fallback(config: RuntimeConfig, reason: str) -> None:
    global _IPEX_FALLBACK_ACTIVE, _IPEX_FALLBACK_REASON, _IPEX_FALLBACK_ANNOUNCED

    _IPEX_FALLBACK_ACTIVE = True
    _IPEX_FALLBACK_REASON = reason
    if not _IPEX_FALLBACK_ANNOUNCED:
        _print_runtime_notice(
            "Switching from local IPEX XPU inference to Ollama after the known Iris Xe 'UR error'. "
            f"Ollama model: {config.ollama_model}."
        )
        _IPEX_FALLBACK_ANNOUNCED = True


class IpexLLMChatAdapter:
    def __init__(self, config: RuntimeConfig, temperature: float):
        self.config = config
        self.temperature = temperature
        self._warmed_up = False

    def invoke(self, messages: List[Any]) -> AIMessage:
        runtime_key = _ipex_runtime_key(self.config)
        torch, tokenizer, model = _load_ipex_runtime(
            self.config.hf_model_id,
            self.config.load_in_4bit,
            self.config.cpu_embedding,
            self.config.optimize_model,
        )

        chat_messages = _normalize_prompt(messages)
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"{message['role']}: {message['content']}" for message in chat_messages
            )

        with torch.inference_mode():
            model_inputs = _prepare_ipex_inputs(tokenizer, prompt)
            input_ids = model_inputs["input_ids"]
            generation_kwargs = _build_ipex_generation_kwargs(
                model,
                tokenizer,
                do_sample=self.temperature > 0,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
            )
            generation_kwargs.update(model_inputs)

            if runtime_key not in _WARMED_UP_IPEX_RUNTIMES:
                _print_ipex_progress("starting warm-up generate on XPU")
                warmup_kwargs = _build_ipex_generation_kwargs(
                    model,
                    tokenizer,
                    do_sample=False,
                    max_new_tokens=1,
                )
                warmup_kwargs.update(model_inputs)
                try:
                    model.generate(**warmup_kwargs)
                except Exception as exc:
                    raise _wrap_ipex_runtime_error(exc, self.config) from exc
                _WARMED_UP_IPEX_RUNTIMES.add(runtime_key)
                _print_ipex_progress("warm-up generate completed")

            if runtime_key not in _FIRST_GENERATE_IPEX_RUNTIMES:
                _print_ipex_progress("starting first full generate")

            try:
                output = model.generate(**generation_kwargs)
            except Exception as exc:
                raise _wrap_ipex_runtime_error(exc, self.config) from exc
            generated_tokens = output[:, input_ids.shape[1]:].cpu()

            if runtime_key not in _FIRST_GENERATE_IPEX_RUNTIMES:
                _FIRST_GENERATE_IPEX_RUNTIMES.add(runtime_key)
                _print_ipex_progress("first full generate completed")

        content = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        return AIMessage(content=content)


class AutoFallbackLLMClient:
    def __init__(self, config: RuntimeConfig, temperature: float, json_mode: bool = False):
        self.config = config
        self.temperature = temperature
        self.json_mode = json_mode
        self._ipex_client = IpexLLMChatAdapter(config=config, temperature=temperature)
        self._ollama_client = _build_ollama_client(
            config,
            temperature=temperature,
            json_mode=json_mode,
        )

    def invoke(self, messages: List[Any]) -> AIMessage:
        if _IPEX_FALLBACK_ACTIVE:
            return self._ollama_client.invoke(messages)

        try:
            return self._ipex_client.invoke(messages)
        except RuntimeError as exc:
            if not _is_ipex_ur_error(exc):
                raise
            _activate_ipex_fallback(self.config, str(exc))
            return self._ollama_client.invoke(messages)


def build_llm_client(temperature: float, json_mode: bool = False):
    config = load_runtime_config()
    if config.backend == "ipex":
        return AutoFallbackLLMClient(
            config=config,
            temperature=temperature,
            json_mode=json_mode,
        )

    return _build_ollama_client(config, temperature=temperature, json_mode=json_mode)


def get_xpu_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {
            "available": False,
            "reason": f"torch import failed: {exc}",
        }

    if not hasattr(torch, "xpu"):
        return {
            "available": False,
            "reason": "This torch build does not expose torch.xpu.",
        }

    available = torch.xpu.is_available()
    status: dict[str, Any] = {
        "available": available,
        "device_count": torch.xpu.device_count() if available else 0,
    }
    if available:
        status["device_name"] = torch.xpu.get_device_name(0)
    return status


def benchmark_ollama_first_token_latency(config: RuntimeConfig) -> float:
    llm = ChatOllama(model=config.ollama_model, temperature=0)
    start = time.perf_counter()
    for chunk in llm.stream([HumanMessage(content="Antworte nur mit OK.")]):
        if str(chunk.content).strip():
            return time.perf_counter() - start

    raise RuntimeError("Ollama returned no streamed token content.")


def benchmark_ollama_full_generation_latency(config: RuntimeConfig) -> float:
    llm = ChatOllama(model=config.ollama_model, temperature=0)
    start = time.perf_counter()
    response = llm.invoke([HumanMessage(content="Antworte nur mit OK.")])
    latency = time.perf_counter() - start
    if not str(response.content).strip():
        raise RuntimeError("Ollama returned empty content.")
    return latency


def benchmark_ipex_first_token_latency(config: RuntimeConfig) -> float:
    torch, tokenizer, model = _load_ipex_runtime(
        config.hf_model_id,
        config.load_in_4bit,
        config.cpu_embedding,
        config.optimize_model,
    )

    try:
        from transformers import TextIteratorStreamer
    except ImportError as exc:
        raise RuntimeError("transformers TextIteratorStreamer is unavailable.") from exc

    chat_messages = [{"role": "user", "content": "Antworte nur mit OK."}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "user: Antworte nur mit OK."

    model_inputs = _prepare_ipex_inputs(tokenizer, prompt)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_kwargs = _build_ipex_generation_kwargs(
        model,
        tokenizer,
        do_sample=False,
        max_new_tokens=16,
        extra_kwargs={"streamer": streamer},
    )
    generation_kwargs.update(model_inputs)

    generation_error: list[BaseException] = []

    def _generate() -> None:
        try:
            model.generate(**generation_kwargs)
        except BaseException as exc:
            generation_error.append(_wrap_ipex_runtime_error(exc, config))

    thread = Thread(
        target=_generate,
        daemon=True,
    )
    start = time.perf_counter()
    thread.start()

    first_chunk = None
    for chunk in streamer:
        if str(chunk).strip():
            first_chunk = chunk
            break

    thread.join()

    if generation_error:
        raise RuntimeError(f"IPEX generate failed: {generation_error[0]}") from generation_error[0]

    if first_chunk is None:
        raise RuntimeError("IPEX generation returned no streamed token content.")

    return time.perf_counter() - start


def benchmark_ipex_full_generation_latency(config: RuntimeConfig) -> float:
    torch, tokenizer, model = _load_ipex_runtime(
        config.hf_model_id,
        config.load_in_4bit,
        config.cpu_embedding,
        config.optimize_model,
    )

    chat_messages = [{"role": "user", "content": "Antworte nur mit OK."}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "user: Antworte nur mit OK."

    model_inputs = _prepare_ipex_inputs(tokenizer, prompt)
    input_ids = model_inputs["input_ids"]
    generation_kwargs = _build_ipex_generation_kwargs(
        model,
        tokenizer,
        do_sample=False,
        max_new_tokens=16,
    )
    generation_kwargs.update(model_inputs)

    with torch.inference_mode():
        start = time.perf_counter()
        try:
            output = model.generate(**generation_kwargs)
        except Exception as exc:
            raise _wrap_ipex_runtime_error(exc, config) from exc
        latency = time.perf_counter() - start
        generated_tokens = output[:, input_ids.shape[1]:].cpu()

    content = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
    if not content:
        raise RuntimeError("IPEX full generation returned empty content.")
    return latency


def _selected_model_name(config: RuntimeConfig) -> str:
    return config.hf_model_id if config.backend == "ipex" else config.ollama_model


def format_self_test_summary(result: dict[str, Any]) -> str:
    status = "PASS" if result["success"] else "FAIL"
    parts = [
        status,
        f"backend={result['backend']}",
        f"model={result['model']}",
    ]
    if result.get("metric"):
        parts.append(f"metric={result['metric']}")
    if result.get("latency_seconds") is not None:
        parts.append(f"latency={result['latency_seconds']:.2f}s")
    if result.get("error"):
        parts.append(f"error={result['error']}")
    return " ".join(parts)


def run_ipex_smoke_test(config: RuntimeConfig) -> int:
    print("\n" + "=" * 60)
    print("IPEX SMOKE TEST")
    print("=" * 60)

    if config.backend != "ipex":
        print("IPEX smoke test requires --backend ipex.")
        return 2

    prompt = "Antworte in genau einem kurzen Satz: GPU smoke test erfolgreich."
    client = IpexLLMChatAdapter(config=config, temperature=0.0)
    start = time.perf_counter()

    try:
        response = client.invoke([HumanMessage(content=prompt)])
    except Exception as exc:
        print(f"IPEX smoke test failed: {exc}")
        print(f"FAIL backend=ipex model={config.hf_model_id} error={exc}")
        return 1

    latency = time.perf_counter() - start
    text = str(response.content).strip()
    print("Smoke test response:")
    print(text if text else "<empty>")
    print(f"PASS backend=ipex model={config.hf_model_id} latency={latency:.2f}s")
    return 0


def run_startup_self_test(config: RuntimeConfig) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("SELF-TEST")
    print("=" * 60)

    result: dict[str, Any] = {
        "success": False,
        "backend": config.backend,
        "model": _selected_model_name(config),
        "metric": None,
        "latency_seconds": None,
        "error": None,
    }

    xpu_status = get_xpu_status()
    print(f"XPU available: {xpu_status['available']}")
    if xpu_status.get("device_count"):
        print(f"XPU devices: {xpu_status['device_count']}")
    if xpu_status.get("device_name"):
        print(f"XPU device 0: {xpu_status['device_name']}")
    if xpu_status.get("reason"):
        print(f"XPU status detail: {xpu_status['reason']}")

    try:
        if config.backend == "ipex":
            latency = benchmark_ipex_first_token_latency(config)
            result["success"] = True
            result["metric"] = "first-token"
            result["latency_seconds"] = latency
            print(
                f"First-token latency for {config.hf_model_id}: {latency:.2f}s"
            )
        else:
            latency = benchmark_ollama_first_token_latency(config)
            result["success"] = True
            result["metric"] = "first-token"
            result["latency_seconds"] = latency
            print(
                f"First-token latency for {config.ollama_model}: {latency:.2f}s"
            )
    except Exception as exc:
        print(f"Streaming benchmark unavailable: {exc}")
        try:
            if config.backend == "ipex":
                latency = benchmark_ipex_full_generation_latency(config)
                result["success"] = True
                result["metric"] = "full-generation"
                result["latency_seconds"] = latency
                print(
                    f"Fallback full-generation latency for {config.hf_model_id}: {latency:.2f}s"
                )
            else:
                latency = benchmark_ollama_full_generation_latency(config)
                result["success"] = True
                result["metric"] = "full-generation"
                result["latency_seconds"] = latency
                print(
                    f"Fallback full-generation latency for {config.ollama_model}: {latency:.2f}s"
                )
        except Exception as fallback_exc:
            result["error"] = str(fallback_exc)
            print(f"Self-test failed: {fallback_exc}")

    return result


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MAS_orch multi-agent research app.",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "ipex"],
        help="Choose the model backend without setting environment variables.",
    )
    parser.add_argument(
        "--ollama-model",
        help="Ollama model name, for example qwen2.5:3b.",
    )
    parser.add_argument(
        "--hf-model-id",
        help="Hugging Face model ID for the IPEX backend.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--question",
        default="Welche Auswirkungen hat Quantum Computing auf aktuelle Kryptographie-Standards?",
        help="Research question for the orchestrator.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a startup self-test that checks XPU visibility and first-token latency.",
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Run the startup self-test and exit without starting the research workflow.",
    )
    parser.add_argument(
        "--ipex-smoke-test",
        action="store_true",
        help="Load the local IPEX model and run one fixed prompt without starting LangGraph.",
    )
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Enable 4-bit loading for the IPEX backend.",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading for the IPEX backend.",
    )
    parser.add_argument(
        "--cpu-embedding",
        dest="cpu_embedding",
        action="store_true",
        help="Keep embeddings on CPU for the IPEX backend.",
    )
    parser.add_argument(
        "--no-cpu-embedding",
        dest="cpu_embedding",
        action="store_false",
        help="Place embeddings on XPU for the IPEX backend.",
    )
    parser.add_argument(
        "--ipex-optimize-model",
        dest="optimize_model",
        action="store_true",
        help="Enable IPEX optimize_model during low-bit conversion.",
    )
    parser.add_argument(
        "--no-ipex-optimize-model",
        dest="optimize_model",
        action="store_false",
        help="Disable IPEX optimize_model for a more conservative XPU path.",
    )
    parser.set_defaults(load_in_4bit=None, cpu_embedding=None, optimize_model=None)
    return parser.parse_args(argv)


# State Definition - zentraler Speicherort für alle Agenten
class ResearchState(TypedDict):
    """
    Zentraler State - ALLE Agenten lesen und schreiben hierhin.
    
    Wichtig: Annotated[List, operator.add] bedeutet:
    - Jeder Agent APPENDED zum State
    - Kein Agent ÜBERSCHREIBT anderen Agent's Output
    - Deterministische State-History
    """
    # Input
    research_question: str
    
    # Kommunikation zwischen Agenten
    messages: Annotated[List[dict], operator.add]
    
    # Task-Tracking
    subtasks: List[str]           # Orchestrator definiert diese
    completed_subtasks: Annotated[List[str], operator.add]
    
    # Ergebnisse (append-only, jeder Agent fügt hinzu)
    research_results: Annotated[List[dict], operator.add]
    
    # Routing-Information
    next_agent: str               # Orchestrator bestimmt, wer als nächstes ran kommt
    
    # Quality Control
    quality_score: Optional[float]
    needs_revision: bool
    
    # Final Output
    final_report: str


# Der Orchestrator Agent steuert den gesamten Research-Prozess.
class OrchestratorAgent:
    """
    Der Orchestrator kennt das Big Picture.
    Er PLANT, DELEGIERT und EVALUIERT - führt selbst keine Research durch.
    
    Kernverantwortlichkeiten:
    1. Task Decomposition (große Aufgabe → kleine Subtasks)
    2. Routing (welcher spezialisierte Agent übernimmt was)
    3. Synthesis (Ergebnisse zusammenführen)
    """
    
    def __init__(self):
        self.llm = build_llm_client(temperature=0, json_mode=True)
        
    def plan_and_route(self, state: ResearchState) -> ResearchState:
        """
        Phase 1: Task Decomposition
        Der Orchestrator zerlegt die Research-Question in Subtasks.
        """
        
        # Wenn noch keine Subtasks existieren → Planning Phase
        if not state.get("subtasks"):
            planning_prompt = f"""
            Du bist ein Research Orchestrator. Deine Aufgabe: Zerlege die folgende 
            Research-Frage in 3-4 konkrete Subtasks, die von spezialisierten Agenten 
            bearbeitet werden können.
            
            Research-Frage: {state['research_question']}
            
            Verfügbare spezialisierte Agenten:
            - "web_researcher": Recherchiert aktuelle Informationen und Fakten
            - "data_analyzer": Analysiert und interpretiert Daten/Statistiken  
            - "synthesizer": Fasst alle Ergebnisse zu einem kohärenten Report zusammen
            
            Antworte NUR mit validem JSON:
            {{
                "subtasks": ["task1", "task2", "task3"],
                "routing_plan": [
                    {{"task": "task1", "agent": "web_researcher"}},
                    {{"task": "task2", "agent": "data_analyzer"}},
                    {{"task": "task3", "agent": "synthesizer"}}
                ],
                "next_agent": "web_researcher"
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=planning_prompt)])
            raw = response.content.strip()
            # Strip markdown code fences if the LLM wrapped the JSON
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0].strip()
            if not raw:
                raise ValueError(
                    "Orchestrator LLM returned an empty response. "
                    "Check that the Ollama model is running and supports JSON mode."
                )
            plan = json.loads(raw)
            
            # State Update mit Plan
            first_task = next(
                (entry["task"] for entry in plan["routing_plan"] if entry["agent"] == plan["next_agent"]),
                plan["subtasks"][0]
            )
            return {
                "subtasks": plan["subtasks"],
                "next_agent": plan["next_agent"],
                "messages": [{
                    "from": "orchestrator",
                    "to": plan["next_agent"],
                    "content": f"Bitte bearbeite: {first_task}",
                    "context": plan["routing_plan"]
                }]
            }
        
        # Phase 2: Routing nach Ergebnissen der Agenten
        else:
            completed = state.get("completed_subtasks", [])
            all_subtasks = state["subtasks"]
            
            # Alle Tasks erledigt? → Zum Synthesizer
            if len(completed) >= len(all_subtasks):
                return {
                    "next_agent": "synthesizer",
                    "messages": [{
                        "from": "orchestrator",
                        "to": "synthesizer",
                        "content": "Alle Research-Ergebnisse sind bereit. Bitte synthesize.",
                        "results_count": len(state.get("research_results", []))
                    }]
                }
            
            # Nächste ausstehende Task bestimmen
            pending = [t for t in all_subtasks if t not in completed]
            next_task = pending[0]
            
            # Agent für nächste Task bestimmen (simple Heuristik)
            next_agent = "data_analyzer" if "Daten" in next_task or "Statistik" in next_task \
                        else "web_researcher"
            
            return {
                "next_agent": next_agent,
                "messages": [{
                    "from": "orchestrator",
                    "to": next_agent,
                    "content": f"Nächste Aufgabe: {next_task}"
                }]
            }
        

# Spezialisierte Worker-Agenten - führen die eigentliche Research-Arbeit durch
class WebResearcherAgent:
    """
    Spezialisiert auf: Information Retrieval & Faktenrecherche
    
    In Production würde hier ein echter Tool-Call stattfinden:
    - Tavily Search API
    - SerpAPI  
    - Browser-Use
    
    Für das Beispiel: Simulated research via LLM
    """
    
    def __init__(self):
        self.llm = build_llm_client(temperature=0.1, json_mode=True)
    
    def research(self, state: ResearchState) -> ResearchState:
        """
        Empfängt Task vom Orchestrator über State.
        Führt Research durch.
        Schreibt Ergebnisse zurück in State.
        """
        
        # Aktuellen Task aus Messages extrahieren
        # (In echtem System: strukturiertes Message-Parsing)
        latest_message = state["messages"][-1] if state["messages"] else {}
        current_task = latest_message.get("content", state["research_question"])
        # Strip routing prefixes so the task matches the bare subtask string
        for _prefix in ("Bitte bearbeite: ", "Nächste Aufgabe: "):
            if current_task.startswith(_prefix):
                current_task = current_task[len(_prefix):]
                break
        
        research_prompt = f"""
        Du bist ein spezialisierter Web-Research-Agent.
        
        Deine aktuelle Aufgabe: {current_task}
        Übergeordnete Research-Frage: {state['research_question']}
        
        Führe eine detaillierte Recherche durch und liefere:
        1. Konkrete Fakten und Informationen
        2. Quellenangaben (auch wenn simuliert)
        3. Relevanz für die Hauptfrage
        
        Antworte als JSON:
        {{
            "findings": ["finding1", "finding2", "finding3"],
            "key_facts": {{"fact1": "value1", "fact2": "value2"}},
            "sources": ["source1", "source2"],
            "task_completed": "{current_task}",
            "confidence": 0.85
        }}
        """
        
        response = self.llm.invoke([
            SystemMessage(content="Du bist ein präziser Research-Agent. Antworte nur mit validem JSON."),
            HumanMessage(content=research_prompt)
        ])
        
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        
        # Zurück an Orchestrator signalisieren
        return {
            "research_results": [{
                "agent": "web_researcher",
                "task": current_task,
                **result
            }],
            "completed_subtasks": [result["task_completed"]],
            "messages": [{
                "from": "web_researcher",
                "to": "orchestrator",
                "content": f"Research abgeschlossen. {len(result['findings'])} Findings gefunden.",
                "status": "completed"
            }],
            "next_agent": "orchestrator"  # Kontrolle zurück an Orchestrator!
        }


class DataAnalyzerAgent:
    """
    Spezialisiert auf: Quantitative Analyse, Pattern Recognition, Statistiken
    """
    
    def __init__(self):
        self.llm = build_llm_client(temperature=0, json_mode=True)
    
    def analyze(self, state: ResearchState) -> ResearchState:
        # Alle bisherigen Research-Ergebnisse als Kontext nutzen
        previous_results = state.get("research_results", [])
        context = json.dumps(previous_results, ensure_ascii=False, indent=2)
        
        latest_message = state["messages"][-1] if state["messages"] else {}
        current_task = latest_message.get("content", "Analysiere die vorhandenen Daten")
        # Strip routing prefixes so the task matches the bare subtask string
        for _prefix in ("Bitte bearbeite: ", "Nächste Aufgabe: "):
            if current_task.startswith(_prefix):
                current_task = current_task[len(_prefix):]
                break
        
        analysis_prompt = f"""
        Du bist ein Data-Analysis-Agent.
        
        Aufgabe: {current_task}
        
        Bisherige Research-Ergebnisse anderer Agenten:
        {context}
        
        Analysiere die Daten und identifiziere:
        1. Signifikante Patterns und Trends
        2. Quantitative Zusammenhänge
        3. Widersprüche oder Datenlücken
        4. Statistische Relevanz der Findings
        
        JSON-Response:
        {{
            "patterns": ["pattern1", "pattern2"],
            "quantitative_insights": {{"metric1": "value1"}},
            "data_quality": "high|medium|low",
            "gaps_identified": ["gap1"],
            "task_completed": "{current_task}",
            "confidence": 0.9
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        
        return {
            "research_results": [{
                "agent": "data_analyzer",
                "task": current_task,
                **result
            }],
            "completed_subtasks": [result["task_completed"]],
            "messages": [{
                "from": "data_analyzer",
                "to": "orchestrator",
                "content": f"Analyse abgeschlossen. Datenqualität: {result['data_quality']}",
                "status": "completed"
            }],
            "next_agent": "orchestrator"
        }


class SynthesizerAgent:
    """
    Spezialisiert auf: Integration aller Ergebnisse → kohärenter Final Report
    Wird IMMER als letzter Agent ausgeführt.
    """
    
    def __init__(self):
        self.llm = build_llm_client(temperature=0.3)
    
    def synthesize(self, state: ResearchState) -> ResearchState:
        all_results = state.get("research_results", [])
        
        synthesis_prompt = f"""
        Du bist ein Synthesis-Agent. Deine Aufgabe: Erstelle einen professionellen,
        kohärenten Research-Report.
        
        Ursprüngliche Research-Frage: {state['research_question']}
        
        Alle gesammelten Ergebnisse:
        {json.dumps(all_results, ensure_ascii=False, indent=2)}
        
        Erstelle einen strukturierten Report mit:
        1. Executive Summary (3-5 Sätze)
        2. Haupterkenntnisse (priorisiert)
        3. Datenbasierte Schlussfolgerungen
        4. Offene Fragen / Empfehlungen für weitere Recherche
        5. Confidence-Score des Gesamtergebnisses
        
        Schreibe professionell, präzise und evidenzbasiert.
        """
        
        response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        
        return {
            "final_report": response.content,
            "messages": [{
                "from": "synthesizer",
                "to": "orchestrator",
                "content": "Final Report erstellt.",
                "status": "final"
            }],
            "next_agent": "quality_checker"
        }


class QualityCheckerAgent:
    """
    Optional: Validierungsschicht
    Überprüft ob der Report die Research-Frage tatsächlich beantwortet.
    """
    
    def __init__(self):
        self.llm = build_llm_client(temperature=0, json_mode=True)
    
    def check(self, state: ResearchState) -> ResearchState:
        check_prompt = f"""
        Bewerte den folgenden Research-Report:
        
        Ursprüngliche Frage: {state['research_question']}
        
        Report:
        {state.get('final_report', '')}
        
        Bewertungskriterien:
        - Beantwortet der Report die Frage? (0-1)
        - Ist er evidenzbasiert? (0-1)
        - Gibt es kritische Lücken? (true/false)
        
        JSON:
        {{
            "answers_question": 0.9,
            "is_evidence_based": 0.85,
            "has_critical_gaps": false,
            "overall_score": 0.87,
            "needs_revision": false,
            "revision_reason": ""
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=check_prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        
        return {
            "quality_score": result["overall_score"],
            "needs_revision": result["needs_revision"],
            "messages": [{
                "from": "quality_checker",
                "content": f"Quality Score: {result['overall_score']}",
                "needs_revision": result["needs_revision"]
            }],
            "next_agent": "orchestrator" if result["needs_revision"] else "done"
        }
    

#Graph Construction: Hier definieren wir, wie die Agenten miteinander verbunden sind und wie der Flow durch das System läuft.
def create_research_graph() -> StateGraph:
    """
    Baut den LangGraph-Graphen.
    
    Kritisches Konzept: 
    - Nodes = Agenten (was wird ausgeführt)
    - Edges = Routing (wer kommt als nächstes)
    - Conditional Edges = Dynamisches Routing basierend auf State
    """
    
    # Agent-Instanzen
    orchestrator = OrchestratorAgent()
    researcher = WebResearcherAgent()
    analyzer = DataAnalyzerAgent()
    synthesizer = SynthesizerAgent()
    quality_checker = QualityCheckerAgent()
    
    # Graph initialisieren
    graph = StateGraph(ResearchState)
    
    # ── NODES HINZUFÜGEN ────────────────────────────────────────────
    graph.add_node("orchestrator", orchestrator.plan_and_route)
    graph.add_node("web_researcher", researcher.research)
    graph.add_node("data_analyzer", analyzer.analyze)
    graph.add_node("synthesizer", synthesizer.synthesize)
    graph.add_node("quality_checker", quality_checker.check)
    
    # ── ENTRY POINT ─────────────────────────────────────────────────
    graph.set_entry_point("orchestrator")
    
    # ── CONDITIONAL EDGES (Das Herzstück des Routings) ───────────────
    def route_from_orchestrator(state: ResearchState) -> str:
        """
        Diese Funktion bestimmt nach jedem Orchestrator-Call:
        Wohin geht der Flow als nächstes?
        
        Basiert NUR auf State - kein direkter Agent-zu-Agent-Call!
        """
        next_agent = state.get("next_agent", "web_researcher")
        
        # Quality-Check: Revision nötig?
        if state.get("needs_revision"):
            return "web_researcher"  # Re-research triggern
        
        valid_routes = {
            "web_researcher": "web_researcher",
            "data_analyzer": "data_analyzer", 
            "synthesizer": "synthesizer",
            "quality_checker": "quality_checker",
        }
        
        return valid_routes.get(next_agent, "synthesizer")
    
    def route_after_quality_check(state: ResearchState) -> str:
        """Terminierung oder Revision?"""
        if state.get("needs_revision", False):
            return "orchestrator"  # Zurück zum Start → Revision Loop
        return END
    
    # Orchestrator kann zu ALLEN Agenten routen
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "web_researcher": "web_researcher",
            "data_analyzer": "data_analyzer",
            "synthesizer": "synthesizer",
            "quality_checker": "quality_checker",
        }
    )
    
    # Worker-Agenten geben Kontrolle IMMER zurück an Orchestrator
    graph.add_edge("web_researcher", "orchestrator")
    graph.add_edge("data_analyzer", "orchestrator")
    graph.add_edge("synthesizer", "quality_checker")  # Synthesizer → direkt zu QC
    
    # Quality-Check: Terminierung oder Revision
    graph.add_conditional_edges(
        "quality_checker",
        route_after_quality_check,
        {
            "orchestrator": "orchestrator",
            END: END
        }
    )
    
    return graph.compile()


# Main Execution Loop: Hier wird das System tatsächlich ausgeführt.
def run_research_system(question: str):
    """
    Hauptfunktion: Startet das Multi-Agenten-System.
    """
    
    # Graph kompilieren
    app = create_research_graph()
    _export_mermaid_graph(app)
    
    # Initial State
    initial_state: ResearchState = {
        "research_question": question,
        "messages": [],
        "subtasks": [],
        "completed_subtasks": [],
        "research_results": [],
        "next_agent": "orchestrator",
        "quality_score": None,
        "needs_revision": False,
        "final_report": ""
    }
    
    runtime = load_runtime_config()

    print(f"🚀 Starte Research-System für: {question}\n")
    print(f"LLM-Backend: {runtime.backend}")
    if runtime.backend == "ollama":
        print("Connecting to external Ollama server...")
        print(f"Modell: {runtime.ollama_model}")
    else:
        print("Loading Hugging Face model locally on XPU with automatic Ollama fallback...")
        print(f"Lokales Modell: {runtime.hf_model_id} auf Intel XPU")
        print(f"Fallback-Modell: {runtime.ollama_model} via Ollama bei bekanntem Iris-Xe-UR-Fehler")
    print("=" * 60)
    
    # Stream-Execution: Jeden Step live beobachten
    # Das ist KRITISCH für Debugging und Monitoring!
    for step in app.stream(initial_state, config={"recursion_limit": 20}):
        
        for node_name, node_output in step.items():
            print(f"\n📍 Agent: {node_name.upper()}")
            print(f"   Next: {node_output.get('next_agent', 'N/A')}")
            
            # Messages anzeigen
            new_messages = node_output.get("messages", [])
            for msg in new_messages:
                print(f"   💬 [{msg.get('from', '?')} → {msg.get('to', '?')}]: {msg.get('content', '')[:100]}")
            
            # Neue Results
            new_results = node_output.get("research_results", [])
            if new_results:
                print(f"   📊 Neue Findings: {len(new_results)} Ergebnis(se)")
            
            # Completed Tasks
            completed = node_output.get("completed_subtasks", [])
            if completed:
                print(f"   ✅ Abgeschlossen: {completed}")
    
    # Finale Ergebnisse aus komplettem State holen
    final_state = app.invoke(initial_state, config={"recursion_limit": 20})
    
    print("\n" + "=" * 60)
    print("📋 FINAL REPORT:")
    print("=" * 60)
    print(final_state.get("final_report", "Kein Report generiert"))
    print(f"\n⭐ Quality Score: {final_state.get('quality_score', 'N/A')}")
    
    return final_state


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_cli_args(argv)
    apply_runtime_overrides(
        backend=args.backend,
        ollama_model=args.ollama_model,
        hf_model_id=args.hf_model_id,
        load_in_4bit=args.load_in_4bit,
        cpu_embedding=args.cpu_embedding,
        optimize_model=args.optimize_model,
        max_new_tokens=args.max_new_tokens,
    )

    runtime = load_runtime_config()
    self_test_result: Optional[dict[str, Any]] = None
    if args.ipex_smoke_test:
        return run_ipex_smoke_test(runtime)

    if args.self_test or args.self_test_only:
        self_test_result = run_startup_self_test(runtime)

    if args.self_test_only:
        print(format_self_test_summary(self_test_result or {
            "success": False,
            "backend": runtime.backend,
            "model": _selected_model_name(runtime),
            "metric": None,
            "latency_seconds": None,
            "error": "self-test did not run",
        }))
        return 0 if self_test_result and self_test_result["success"] else 1

    run_research_system(args.question)
    return 0


# Ausführung
if __name__ == "__main__":
    raise SystemExit(main())

