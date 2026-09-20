"""Agents package: automated auditing, licensing, toolchain coverage, and VAE checkers."""

from __future__ import annotations

from data_forge.agents.audit_agent import AuditAgent
from data_forge.agents.fp8_eval import FP8EvalHarness
from data_forge.agents.license_agent import LicenseVerificationAgent
from data_forge.agents.toolchain_checker import ToolchainCoverageError, check_unsloth_support
from data_forge.agents.vae_checker import VAEConfigError, check_vae_config

# Aliases for convenience
FP8Evaluator = FP8EvalHarness
LicenseAgent = LicenseVerificationAgent
VAEChecker = check_vae_config

__all__ = [
    "AuditAgent",
    "LicenseVerificationAgent",
    "LicenseAgent",
    "check_unsloth_support",
    "ToolchainCoverageError",
    "check_vae_config",
    "VAEConfigError",
    "VAEChecker",
    "FP8EvalHarness",
    "FP8Evaluator",
]
