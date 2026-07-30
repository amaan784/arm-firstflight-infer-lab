"""Typed loaders for configs/models.yaml, instances.yaml, workloads.yaml.

Light dataclasses over plain dicts — enough structure to catch typos without a heavy
schema dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .util import config_dir

# --- raw yaml -----------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Set FIRSTFLIGHT_CONFIG_DIR or run from the repo root."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top of {path}, got {type(data).__name__}.")
    return data


# --- models -------------------------------------------------------------------


@dataclass(frozen=True)
class ModelVariant:
    name: str
    file: str
    url: str
    sha256: str = ""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_repo: str
    description: str
    variants: dict[str, ModelVariant] = field(default_factory=dict)

    def variant(self, name: str) -> ModelVariant:
        if name not in self.variants:
            raise KeyError(
                f"Model '{self.id}' has no variant '{name}'. "
                f"Available: {', '.join(self.variants) or '(none)'}"
            )
        return self.variants[name]


@dataclass(frozen=True)
class ModelsConfig:
    default_smoke: str
    default_smoke_variant: str
    models: dict[str, ModelSpec]

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self.models:
            raise KeyError(
                f"Unknown model '{model_id}'. Known: {', '.join(self.models) or '(none)'}"
            )
        return self.models[model_id]

    def smoke(self) -> tuple[ModelSpec, ModelVariant]:
        spec = self.get(self.default_smoke)
        return spec, spec.variant(self.default_smoke_variant)


def load_models(path: Path | None = None) -> ModelsConfig:
    data = load_yaml(path or config_dir() / "models.yaml")
    models: dict[str, ModelSpec] = {}
    for model_id, raw in (data.get("models") or {}).items():
        variants = {
            vname: ModelVariant(
                name=vname,
                file=str(vraw.get("file", "")),
                url=str(vraw.get("url", "")),
                sha256=str(vraw.get("sha256", "") or ""),
            )
            for vname, vraw in (raw.get("variants") or {}).items()
        }
        models[model_id] = ModelSpec(
            id=model_id,
            hf_repo=str(raw.get("hf_repo", "")),
            description=str(raw.get("description", "")),
            variants=variants,
        )
    return ModelsConfig(
        default_smoke=str(data.get("default_smoke", "")),
        default_smoke_variant=str(data.get("default_smoke_variant", "")),
        models=models,
    )


# --- instances ----------------------------------------------------------------


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    arch: str
    cpu: str
    vcpus: int
    usd_per_hour: float
    notes: str = ""


@dataclass(frozen=True)
class InstancesConfig:
    default_instance: str
    instances: dict[str, InstanceSpec]

    def get(self, name: str | None = None) -> InstanceSpec:
        name = name or self.default_instance
        if name not in self.instances:
            raise KeyError(
                f"Unknown instance '{name}'. Known: {', '.join(self.instances) or '(none)'}"
            )
        return self.instances[name]


def load_instances(path: Path | None = None) -> InstancesConfig:
    data = load_yaml(path or config_dir() / "instances.yaml")
    instances = {
        name: InstanceSpec(
            name=name,
            arch=str(raw.get("arch", "")),
            cpu=str(raw.get("cpu", "")),
            vcpus=int(raw.get("vcpus", 0) or 0),
            usd_per_hour=float(raw.get("usd_per_hour", 0.0) or 0.0),
            notes=str(raw.get("notes", "")),
        )
        for name, raw in (data.get("instances") or {}).items()
    }
    return InstancesConfig(
        default_instance=str(data.get("default_instance", "")),
        instances=instances,
    )


# --- workloads ----------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    description: str
    prompt_lengths: list[int]
    gen_lengths: list[int]
    repeats: int = 1
    warmup: int = 0


@dataclass(frozen=True)
class WorkloadsConfig:
    default_workload: str
    workloads: dict[str, WorkloadSpec]

    def get(self, name: str | None = None) -> WorkloadSpec:
        name = name or self.default_workload
        if name not in self.workloads:
            raise KeyError(
                f"Unknown workload '{name}'. Known: {', '.join(self.workloads) or '(none)'}"
            )
        return self.workloads[name]


def load_workloads(path: Path | None = None) -> WorkloadsConfig:
    data = load_yaml(path or config_dir() / "workloads.yaml")
    workloads = {
        name: WorkloadSpec(
            name=name,
            description=str(raw.get("description", "")),
            prompt_lengths=[int(x) for x in (raw.get("prompt_lengths") or [])],
            gen_lengths=[int(x) for x in (raw.get("gen_lengths") or [])],
            repeats=int(raw.get("repeats", 1) or 1),
            warmup=int(raw.get("warmup", 0) or 0),
        )
        for name, raw in (data.get("workloads") or {}).items()
    }
    return WorkloadsConfig(
        default_workload=str(data.get("default_workload", "")),
        workloads=workloads,
    )
