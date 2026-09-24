"""Planner backend, vLLM-accelerated — OPT-IN, not the default, run in an
isolated subprocess (mirrors critic_backend.py's architecture exactly).

See docs/review/29_vllm_planner_migration_research.md for the full
research trail. Summary of what was actually verified (via real package
metadata and one independently-reported measured result, not just docs):

- vllm==0.29.0 has native Qwen3.5 (Gated DeltaNet hybrid) support,
  confirmed via its own official PyPI release notes ("New architectures:
  Qwen3.5 (#34110)").
- vllm==0.29.0's real, declared wheel metadata requires
  transformers>=5.10.4 (no ceiling) and an EXACT torch==2.13.0 — a much
  stricter pin than the main venv's torch>=2.6.0, and the reason this
  needs its own venv/subprocess exactly like the Critic tier, not a
  reason to force it into the main venv.
- transformers==5.17.0 specifically (not just the bare floor) is
  independently reported as measured working for Qwen3.5 on vLLM 0.29.0
  on real hardware (community report, RTX PRO 6000).
- The full vllm+torch+transformers dependency set was verified to
  resolve with ZERO conflicts via a real `pip install --dry-run`.
- bitsandbytes is NOT declared anywhere in vllm==0.29.0's metadata, while
  compressed-tensors IS a required core dependency — signaling
  AWQ/GPTQ/compressed-tensors as vLLM's well-supported, first-class
  quantization path. See `model_id`'s docstring below for the
  recommended (not silently assumed) replacement checkpoint.

WHAT IS NOT VERIFIED, stated plainly: this project has no GPU in its own
environment to actually run vLLM's LLM class against these pins, load
the recommended checkpoint, or measure real per-turn latency. Everything
above is verified at the package/metadata level, not the "it actually
loads and generates correctly on this project's target hardware" level.
That is why this is wired as an explicit opt-in
(KRISNA_PLANNER_BACKEND=vllm) rather than replacing the default — see
factory.py.

ARCHITECTURE (found and corrected during this review pass): an earlier
draft of this file called vLLM's Python API directly, in-process — which
would silently do nothing useful, since the actual krisna_inference
service process runs under the MAIN venv's interpreter
(/opt/venv-inference), which does not and should not have vllm or
torch==2.13.0 installed. Fixed to match critic_backend.py's already-
correct pattern exactly: this class manages a subprocess
(planner_worker_vllm.py) running under a SEPARATE interpreter
(/opt/venv-planner), talking over stdin/stdout JSON lines. Subclasses
PlannerBackend (planner_backend.py) and overrides ONLY load(), unload(),
and the _generate_raw() hook that class was refactored to expose
specifically for this purpose — RAG retrieval, system-prompt
construction, and the JSON generate-validate-retry loop are inherited
unchanged, so both backends share exactly one implementation of that
logic and cannot silently drift apart from each other.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import select
import subprocess
import sys
from pathlib import Path

from krisna_inference.backends.planner_backend import PlannerBackend
from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import OOMSimulatedError

log = logging.getLogger("krisna_inference.backends.planner_vllm")

WORKER_SCRIPT = Path(__file__).parent / "planner_worker_vllm.py"


def compute_gpu_memory_utilization_fraction(declared_gb: float, total_gb: float) -> float:
    """Pure, torch-free formula so it has a direct unit test (see
    tests/inference/test_planner_vllm_backend.py) without needing a real
    GPU or vLLM installed in THIS (parent) process — this project's own
    process never imports torch itself, only the subprocess worker does.

    vLLM's gpu_memory_utilization is a fraction of the REAL, live GPU's
    total memory that vLLM greedily reserves up front for its own
    KV-cache pool — completely independent of this project's own
    VRAMLedger (§7.2), which does its admission bookkeeping against
    DECLARED budgets for determinism. Left at vLLM's own default (0.92,
    sized for "vLLM owns the whole GPU" deployments), this backend would
    try to grab ~92% of the entire physical card for itself the moment
    it loads — starving or OOM-crashing the Sketch tier, which per PRD
    §5.3 is ALSO supposed to be resident at the same time as the Planner
    in the idle/conversing state. This formula converts the project's
    own declared vram_gb budget into vLLM's fraction-of-real-memory
    terms, with a ~15% safety margin (matching VRAMLedger's own
    `safety_margin_gb` concept) since vLLM's actual peak usage during
    warmup/CUDA-graph capture can exceed its target utilization briefly.
    Capped at 0.95 — vLLM's own practical ceiling.
    """
    if total_gb <= 0:
        raise ValueError(f"total_gb must be positive, got {total_gb!r}")
    raw_fraction = declared_gb / total_gb
    return min(0.95, raw_fraction * 1.15)


class PlannerBackendVLLM(PlannerBackend):
    def __init__(
        self,
        spec,
        worker_python: str | None = None,
        # NOT the main venv's Qwen/Qwen3.5-9B + bitsandbytes-NF4 checkpoint
        # — see module docstring. RedHatAI/Qwen3.5-9B-quantized.w4a16 is
        # this project's RECOMMENDED candidate (GPTQ W4A16 via
        # llm-compressor, RedHatAI being llm-compressor's maintaining org
        # — the same provenance-quality bar as Unsloth's canonical Gemma-4
        # bnb checkpoint for the Critic tier), but its accuracy has NOT
        # been independently re-verified by this project. Per PRD
        # Appendix C.4's own stated discipline ("verify at the point of
        # adoption, not the point of announcement"), evaluate before
        # relying on it in production, don't just trust this default.
        model_id: str = "RedHatAI/Qwen3.5-9B-quantized.w4a16",
        rag_corpus_dir: str | None = None,
        rag_top_k: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        gpu_memory_utilization: float | None = None,  # None = computed at load() time, see the module-level formula above
        startup_timeout_s: float = 180.0,   # vLLM's own model load (CUDA graph capture,
                                              # kernel compilation) is typically slower
                                              # to first-ready than plain transformers'
                                              # from_pretrained() — critic_backend.py's
                                              # 60s default would be too tight here.
        call_timeout_s: float = 60.0,
    ) -> None:
        # Deliberately does NOT call PlannerBackend.__init__ — that
        # constructor sets up bitsandbytes/dtype/CPU-offload fields this
        # backend doesn't use at all (vLLM manages its own quantization
        # and memory in a separate process). ModelBackend.__init__
        # (PlannerBackend's own parent) is what this class actually needs.
        super(PlannerBackend, self).__init__(spec)
        if worker_python is None:
            worker_python = os.environ.get("KRISNA_PLANNER_VLLM_VENV_PYTHON") or (
                "./venv-planner/Scripts/python.exe" if sys.platform == "win32" else "./venv-planner/bin/python"
            )
        self.worker_python = worker_python
        self.model_id = model_id
        self.rag_corpus_dir = rag_corpus_dir
        self.rag_top_k = rag_top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization
        self.startup_timeout_s = startup_timeout_s
        self.call_timeout_s = call_timeout_s
        self._proc: subprocess.Popen | None = None
        self._rag_index = None

    async def _spawn(self) -> None:
        if not Path(self.worker_python).exists():
            raise BackendLoadError(
                f"Planner (vLLM) worker interpreter not found: {self.worker_python}. "
                "This tier needs its own isolated venv — see "
                "requirements-planner.txt and set KRISNA_PLANNER_VLLM_VENV_PYTHON, "
                "or build the Docker image with --build-arg INSTALL_VLLM_PLANNER=true."
            )

        def _spawn_sync():
            return subprocess.Popen(
                [self.worker_python, str(WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

        self._proc = await asyncio.to_thread(_spawn_sync)

    def _call_sync(self, cmd: str, params: dict | None = None, timeout: float | None = None) -> dict:
        """Blocking subprocess I/O, with NO asyncio wrapper of its own —
        deliberately, so it's directly callable from _generate_raw(),
        which runs inside a worker thread PlannerBackend.run() already
        spawned via asyncio.to_thread (see that method's _run_sync
        closure). Wrapping this in another asyncio.to_thread there would
        be a no-op at best and a source of confusion at worst; async
        callers (load()/unload()) get the async wrapper via _call()
        below instead, which is the one place actually running on the
        event loop.

        Uses select() for a real read timeout on the worker's stdout —
        Popen.stdout.readline() has no built-in per-call timeout, and a
        hung/crashed worker must not block this thread forever. POSIX-
        only (select on a pipe fd) — consistent with this project's
        Linux/WSL2 target platform (see run_data_forge.sh's own header).
        """
        if self._proc is None or self._proc.poll() is not None:
            raise BackendLoadError("Planner (vLLM) worker subprocess is not running")

        request = json.dumps({"cmd": cmd, "params": params or {}}) + "\n"
        effective_timeout = timeout or self.call_timeout_s

        self._proc.stdin.write(request)
        self._proc.stdin.flush()

        if sys.platform != "win32":
            ready, _, _ = select.select([self._proc.stdout], [], [], effective_timeout)
            if not ready:
                raise BackendLoadError(
                    f"Planner (vLLM) worker '{cmd}' timed out after {effective_timeout}s"
                )
        line = self._proc.stdout.readline()
        if not line:
            stderr_tail = self._proc.stderr.read()
            raise BackendLoadError(
                f"Planner (vLLM) worker produced no response (process may have "
                f"crashed). stderr tail:\n{stderr_tail[-4000:]}"
            )
        return json.loads(line)

    async def _call(self, cmd: str, params: dict | None = None, timeout: float | None = None) -> dict:
        return await asyncio.to_thread(self._call_sync, cmd, params, timeout)

    async def load(self) -> None:
        await self._spawn()

        gmu = self.gpu_memory_utilization
        if gmu is None:
            gmu = await asyncio.to_thread(self._probe_gpu_memory_utilization)

        try:
            response = await self._call(
                "load",
                {"model_id": self.model_id, "gpu_memory_utilization": gmu},
                timeout=self.startup_timeout_s,
            )
        except Exception:
            await self._kill()
            raise

        if not response.get("ok"):
            await self._kill()
            if response.get("error_type") == "oom":
                raise OOMSimulatedError(response.get("error", "OOM in planner (vLLM) worker"))
            raise BackendLoadError(f"Planner (vLLM) worker load failed: {response.get('error')}")

        if self.rag_corpus_dir:
            from pathlib import Path as _Path

            from krisna_inference.backends.planner_backend import _RAG_CORPUS_RELATIVE_PATH
            from krisna_inference.backends.planner_rag import UICritRAGIndex

            corpus_path = _Path(self.rag_corpus_dir) / _RAG_CORPUS_RELATIVE_PATH
            self._rag_index = await asyncio.to_thread(UICritRAGIndex.from_jsonl, corpus_path)
        else:
            from krisna_inference.backends.planner_rag import UICritRAGIndex

            log.warning(
                "planner_vllm_no_rag_corpus_configured",
                extra={"note": "No rag_corpus_dir set — planner will run with no retrieval "
                                "context at all. Set KRISNA_PLANNER_RAG_CORPUS_DIR to "
                                "data-forge's model_data/ directory."},
            )
            self._rag_index = UICritRAGIndex()

        self._loaded = True

    def _probe_gpu_memory_utilization(self) -> float:
        """Runs in the PARENT process, which — unlike the vLLM worker
        subprocess — is not guaranteed to have torch installed at all
        (the main venv has its own torch>=2.6.0, but this specific
        function needs to inspect the REAL physical GPU regardless of
        which venv is asking). Falls back to a conservative default if
        torch isn't importable here or no GPU is visible, rather than
        crashing load() over a probe that's allowed to be approximate."""
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device visible")
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception as e:
            log.warning(
                "planner_vllm_gpu_probe_failed",
                extra={"error": str(e), "note": "Falling back to a conservative "
                                                  "gpu_memory_utilization guess (0.3)."},
            )
            return 0.3

        fraction = compute_gpu_memory_utilization_fraction(self.spec.vram_gb, total_gb)
        log.info(
            "planner_vllm_gpu_memory_utilization_computed",
            extra={"declared_vram_gb": self.spec.vram_gb, "real_total_gb": round(total_gb, 2),
                   "computed_fraction": round(fraction, 3)},
        )
        return fraction

    async def unload(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                await self._call("unload", timeout=15.0)
            except Exception as e:
                log.warning("planner_vllm_unload_call_failed", extra={"error": str(e)})
        await self._kill()
        self._rag_index = None
        self._loaded = False

    async def _kill(self) -> None:
        if self._proc is None:
            return

        def _kill_sync():
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

        await asyncio.to_thread(_kill_sync)
        self._proc = None

    def _generate_raw(
        self,
        system: str,
        conversation_history: list[dict] | None,
        message: str,
        extra_instruction: str | None = None,
    ) -> str:
        """Overrides PlannerBackend's transformers-based generation with
        a call to the vLLM subprocess. Everything upstream of this call
        (RAG retrieval, system-prompt construction) and downstream of it
        (JSON extraction, retry loop, response packaging) is inherited
        from PlannerBackend unchanged — see this module's docstring.
        Called from inside the worker thread PlannerBackend.run()'s
        _run_sync closure already spawned via asyncio.to_thread, so this
        method itself must stay synchronous — hence _call_sync rather
        than the async _call."""
        messages = [{"role": "system", "content": system}]
        if conversation_history:
            for turn in conversation_history[-6:]:
                role = "assistant" if turn.get("role") == "planner" else turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        if extra_instruction:
            messages.append({"role": "system", "content": extra_instruction})
        messages.append({"role": "user", "content": message})

        response = self._call_sync(
            "generate",
            {"messages": messages, "max_tokens": self.max_new_tokens, "temperature": self.temperature},
        )
        if not response.get("ok"):
            if response.get("error_type") == "oom":
                raise OOMSimulatedError(response.get("error", "OOM in planner (vLLM) worker"))
            raise BackendLoadError(f"Planner (vLLM) worker generate failed: {response.get('error')}")
        return response["text"]
