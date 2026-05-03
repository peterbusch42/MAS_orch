# MAS_orch

This project can now run with either:

1. Ollama as an external local inference server.
2. A direct Intel GPU backend through `ipex-llm` inside Python.

## Intel GPU recommendation

On this Lenovo T16, the detected GPU is Intel Iris Xe with driver `32.0.101.7026`.
For this hardware, the correct Python package is:

`ipex-llm[xpu_2.6]`

Do not use `ipex-llm[xpu]` unless you explicitly want the older PyTorch 2.1 stack. That path currently resolves to deprecated dependencies and failed during installation in this environment.

## Install base project

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Install Intel XPU backend

```powershell
.\.venv\Scripts\python.exe -m pip install --pre -r requirements-gpu.txt
```

Equivalent direct command:

```powershell
.\.venv\Scripts\python.exe -m pip install --pre --upgrade "ipex-llm[xpu_2.6]" --extra-index-url https://download.pytorch.org/whl/xpu
```

## Backend selection

You can select the runtime either with environment variables or with CLI arguments.

### CLI examples

Ollama:

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ollama --ollama-model qwen2.5:3b
```

Intel GPU:

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ipex --hf-model-id Qwen/Qwen2.5-0.5B-Instruct --self-test
```

Normal workflow with automatic fallback to Ollama if the known Iris Xe `UR error` appears:

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ipex --hf-model-id Qwen/Qwen2.5-0.5B-Instruct --ollama-model qwen2.5:3b
```

IPEX smoke test without LangGraph:

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ipex --hf-model-id Qwen/Qwen2.5-0.5B-Instruct --ipex-smoke-test
```

You can also override generation settings from the CLI:

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ipex --hf-model-id Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 256 --cpu-embedding --load-in-4bit --no-ipex-optimize-model
```

### Startup self-test

Use `--self-test` to run a small startup check before the research graph begins.
Use `--self-test-only` to run the benchmark and exit without launching the research workflow.
In `--self-test-only` mode, the final line is a script-friendly one-line summary such as `PASS ...` or `FAIL ...`.
It reports:

- Whether `torch.xpu` is available.
- The detected Intel XPU device name.
- First-token latency for the selected local model.
- A fallback full-generation latency if token streaming is unstable, which can happen on Iris Xe.
- A final one-line `PASS` or `FAIL` summary with the measured latency when available.

Examples:

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ipex --hf-model-id Qwen/Qwen2.5-0.5B-Instruct --self-test
```

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ipex --hf-model-id Qwen/Qwen2.5-0.5B-Instruct --self-test-only
```

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ollama --ollama-model qwen2.5:3b --self-test
```

### Ollama backend

```powershell
$env:MAS_LLM_BACKEND = "ollama"
$env:MAS_OLLAMA_MODEL = "llama3.1:8b"
.\.venv\Scripts\python.exe .\MAS_orch.py
```

### Proxy-aware Ollama pull on this network

If `ollama pull ...` fails with `lookup registry.ollama.ai: no such host`, start a temporary Ollama server with proxy variables and pull through that server:

Terminal 1:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:HTTP_PROXY = "http://127.0.0.1:3128"
$env:HTTPS_PROXY = "http://127.0.0.1:3128"
$env:NO_PROXY = "127.0.0.1,localhost"
$env:http_proxy = "http://127.0.0.1:3128"
$env:https_proxy = "http://127.0.0.1:3128"
$env:no_proxy = "127.0.0.1,localhost"
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```

Terminal 2:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11435"
ollama pull qwen2.5:3b
```

After the pull succeeds, stop the temporary server in Terminal 1 with `Ctrl+C`. The model is written into the normal shared Ollama model store and remains available to the default Ollama server.

### Intel GPU backend

```powershell
$env:MAS_LLM_BACKEND = "ipex"
$env:MAS_HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
$env:MAS_IPEX_LOAD_IN_4BIT = "true"
$env:MAS_IPEX_CPU_EMBEDDING = "true"
$env:MAS_IPEX_OPTIMIZE_MODEL = "false"
$env:SYCL_CACHE_PERSISTENT = "1"
.\.venv\Scripts\python.exe .\MAS_orch.py
```

## Practical guidance for this T16

- Start with `Qwen/Qwen2.5-0.5B-Instruct` on Iris Xe.
- Move up to `Qwen/Qwen2.5-1.5B-Instruct` only after the smaller model is stable.
- Keep `MAS_IPEX_LOAD_IN_4BIT=true`.
- Keep `MAS_IPEX_CPU_EMBEDDING=true` on Iris Xe to reduce GPU memory pressure.
- Start with `MAS_IPEX_OPTIMIZE_MODEL=false` on this machine; enable it later only if smoke-test runs are stable.
- Normal `--backend ipex` workflow runs now auto-fallback to Ollama if the exact Iris Xe `UR error` happens during generation.
- Expect the first generation to be slower because Intel XPU kernels need warm-up.
- If you want Ollama itself to use Intel GPU, prefer the `ollama-ipex-llm` portable zip release rather than the stock Ollama build.

## IPEX progress messages

When using the local Intel GPU backend, the script now prints progress markers such as:

- `[2026-05-03 16:35:00] IPEX progress: loading tokenizer ...`
- `[2026-05-03 16:35:02] IPEX progress: loading model weights and running low-bit conversion ...`
- `[2026-05-03 16:35:35] IPEX progress: starting warm-up generate on XPU`
- `[2026-05-03 16:36:10] IPEX progress: starting first full generate`

These messages help you distinguish between slow download/load time, low-bit conversion time, warm-up time, and the first actual generation step.

## IPEX smoke test

Use `--ipex-smoke-test` to bypass LangGraph and verify the local Intel GPU path directly with one fixed prompt.

```powershell
.\.venv\Scripts\python.exe .\MAS_orch.py --backend ipex --ipex-smoke-test --no-ipex-optimize-model
```

This is the quickest way to answer the question: "Can this model load on Iris Xe and produce one response at all?"

## What changed in the code

- The app no longer hardcodes `ChatOllama`.
- `MAS_LLM_BACKEND=ipex` loads one shared `ipex-llm` model on Intel XPU and reuses it across all agents.
- Normal `MAS_LLM_BACKEND=ipex` workflow runs now auto-switch to the configured Ollama model if the known Iris Xe `UR error` occurs during generation.
- `MAS_LLM_BACKEND=ollama` preserves the original workflow.
- `--backend`, `--ollama-model`, and `--hf-model-id` let you switch runtimes without exporting environment variables.
- `--self-test-only` runs the benchmark path and exits before the research graph starts.
- `--self-test` checks XPU visibility and benchmarks first-token latency before the main run.
- The default IPEX model is now `Qwen/Qwen2.5-0.5B-Instruct`, which is a safer starting point for Iris Xe.
- `--ipex-smoke-test` runs one fixed local XPU prompt without starting the orchestration workflow.
