"""Critic backend — Gemma 4 31B Dense, run in an isolated subprocess.

See critic_worker.py's module docstring for WHY this is a subprocess at
all: a genuine, unresolvable-in-one-venv `transformers` version conflict
between this tier (needs exactly transformers==5.5.0, per Unsloth's
Gemma-4 pin) and the planner tier (needs transformers built from git main
for Qwen3.5 support). This backend owns the subprocess lifecycle
(spawn/load/run/unload/terminate) and translates the worker's JSON
protocol into the same ModelBackend interface every other tier uses — the
swap orchestrator doesn't know or care that this tier is out-of-process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from krisna_inference.orchestrator.exceptions import BackendLoadError
from krisna_inference.orchestrator.model_registry import ModelBackend, OOMSimulatedError

log = logging.getLogger("krisna_inference.backends.critic")

WORKER_SCRIPT = Path(__file__).parent / "critic_worker.py"


class CriticBackend(ModelBackend):
    def __init__(
        self,
        spec,
        worker_python: str | None = None,
        model_id: str = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
        startup_timeout_s: float = 60.0,
        call_timeout_s: float = 120.0,
        max_gpu_gb: float | None = None,   # low-VRAM mode — see critic_worker.py's
        max_cpu_gb: float | None = None,   # _load() docstring for the real
                                             # unsloth-bypass reasoning and the
                                             # confirmed ~40GB system-RAM cost
    ) -> None:
        super().__init__(spec)
        if worker_python is None:
            worker_python = os.environ.get("KRISNA_CRITIC_VENV_PYTHON") or (
                "./venv-critic/Scripts/python.exe" if sys.platform == "win32" else "./venv-critic/bin/python"
            )
        self.worker_python = worker_python
        self.model_id = model_id
        self.startup_timeout_s = startup_timeout_s
        self.call_timeout_s = call_timeout_s
        self.max_gpu_gb = max_gpu_gb
        self.max_cpu_gb = max_cpu_gb
        self._proc: subprocess.Popen | None = None

    async def _spawn(self) -> None:
        if not Path(self.worker_python).exists():
            raise BackendLoadError(
                f"Critic worker interpreter not found: {self.worker_python}. "
                "This tier needs its own isolated venv — see "
                "requirements-critic.txt and the README's 'Critic tier "
                "isolation' section for setup."
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

    async def _call(self, cmd: str, params: dict | None = None, timeout: float | None = None) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise BackendLoadError("Critic worker subprocess is not running")

        request = json.dumps({"cmd": cmd, "params": params or {}}) + "\n"

        def _io_sync():
            self._proc.stdin.write(request)
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                stderr_tail = self._proc.stderr.read()
                raise BackendLoadError(
                    f"Critic worker produced no response (process may have "
                    f"crashed). stderr tail:\n{stderr_tail[-4000:]}"
                )
            return json.loads(line)

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_io_sync), timeout=timeout or self.call_timeout_s
            )
        except asyncio.TimeoutError as e:
            raise BackendLoadError(f"Critic worker '{cmd}' timed out after {timeout}s") from e

    async def load(self) -> None:
        await self._spawn()
        try:
            load_params = {"model_id": self.model_id}
            if self.max_gpu_gb is not None:
                load_params["max_gpu_gb"] = self.max_gpu_gb
                load_params["max_cpu_gb"] = self.max_cpu_gb
            response = await self._call(
                "load",
                load_params,
                timeout=self.startup_timeout_s,
            )
        except Exception:
            await self._kill()
            raise

        if not response.get("ok"):
            await self._kill()
            if response.get("error_type") == "oom":
                raise OOMSimulatedError(response.get("error", "OOM in critic worker"))
            raise BackendLoadError(f"Critic worker load failed: {response.get('error')}")

        self._loaded = True

    async def unload(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                await self._call("unload", timeout=15.0)
            except Exception as e:
                log.warning("critic_unload_call_failed", extra={"error": str(e)})
        await self._kill()
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

    async def run(self, image_ref: str | None = None, constraints: dict | None = None, **kwargs):
        if not self._loaded:
            raise RuntimeError("Critic backend not loaded")
        if not image_ref:
            raise ValueError("CriticBackend.run() requires image_ref")

        from krisna_inference.backends.blob_store_singleton import get_blob_store

        image_path = str(get_blob_store().path_for(image_ref))
        response = await self._call("run", {"image_path": image_path, "constraints": constraints or {}})

        if not response.get("ok"):
            if response.get("error_type") == "oom":
                raise OOMSimulatedError(response.get("error", "OOM in critic worker"))
            raise BackendLoadError(f"Critic worker run failed: {response.get('error')}")

        return {"tier": self.spec.tier.value, "critique_result": response["critique_result"]}
