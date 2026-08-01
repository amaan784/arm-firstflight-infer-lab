"""Arm Performix wrapper (clean interface, graceful no-op off Arm).

Arm Performix is Arm's performance-analysis toolkit for Neoverse server/cloud (AWS Graviton,
Microsoft Cobalt, Google Axion). Its CLI tool is `apx`.

Real workflow (sourced from Arm's open-source MCP server, github.com/arm/mcp —
mcp-local/utils/apx.py + mcp-local/sql/queries.sql — since the official CLI Reference Guide
doc 111566 is a JS-rendered SPA that a plain fetch can't read):

  1. register a target (once):  apx target add <user>@<host>:22:<key>   (or --target=localhost)
  2. run a recipe:  apx recipe run code_hotspots --workload="<cmd>" --json --target=localhost
  3. render the run:  apx run render <run_id>                  # -> session_id
  4. query hotspots:  apx render query <session_id> "<top-10 self-time SQL>"

The `code_hotspots` recipe's query returns the top-10 self-time functions with columns
function_name, node_type, periodic_samples_self, periodic_samples_self_percent, rendered as a
`┃`-delimited Unicode table (or JSON via --json).

Off Arm / without `apx` / on a failure of the (high-confidence but unverified) invocation,
`profile()` returns a `skipped` ProfileResult with a clear reason and never raises.

TODO(confirm) on the live box / against doc 111566: the exact JSON keys for run_id / session_id
and whether `apx recipe run --json` already embeds the hotspot table (skipping render/query).
The COMMANDS themselves come from Arm's own tooling, not invention.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..util import is_arm_linux

APX_BINARY = "apx"
APX_DEFAULT_PATHS = ["/opt/Arm Performix/assets/apx/apx"]  # TODO(confirm) exact install path
PERFORMIX_DOC = "https://developer.arm.com/documentation/111566"
ARM_MCP = "https://github.com/arm/mcp"
TOP_N = 12

# Recipes (Arm MCP server). code_hotspots needs the fewest PMU counters; the others need all.
RECIPES = ["code_hotspots", "instruction_mix", "cpu_microarchitecture", "memory_access", "all"]
DEFAULT_RECIPE = "code_hotspots"

# The documented top-10 self-time hotspot query (mcp-local/sql/queries.sql). The full query
# builds `agg` from drilldown tables; this is the selecting tail the parser ultimately needs.
HOTSPOTS_SQL = (
    "SELECT a.function_name, a.node_type, a.periodic_samples_self, "
    "a.periodic_samples_self_percent FROM agg a ORDER BY a.periodic_samples_self DESC LIMIT 10;"
)


@dataclass
class Hotspot:
    function: str
    percent: float
    module: str = ""


@dataclass
class ProfileResult:
    skipped: bool
    reason: str = ""
    timestamp: str = ""
    target: str = ""
    tool: str = "arm-performix"
    recipe: str = DEFAULT_RECIPE
    hotspots: list[Hotspot] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> ProfileResult:
        return cls(
            skipped=bool(d.get("skipped", True)),
            reason=str(d.get("reason", "")),
            timestamp=str(d.get("timestamp", "")),
            target=str(d.get("target", "")),
            tool=str(d.get("tool", "arm-performix")),
            recipe=str(d.get("recipe", DEFAULT_RECIPE)),
            hotspots=[Hotspot(**h) for h in d.get("hotspots", [])],
            raw_output=str(d.get("raw_output", "")),
        )

    @classmethod
    def load_json(cls, path: Path) -> ProfileResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def find_apx() -> Path | None:
    """Locate the `apx` CLI on PATH or at the documented install path."""
    found = shutil.which(APX_BINARY)
    if found:
        return Path(found)
    for candidate in APX_DEFAULT_PATHS:
        p = Path(candidate)
        if p.exists():
            return p
    return None


# --- command construction (real apx CLI; source: github.com/arm/mcp) ----------


def build_recipe_run_cmd(
    apx: Path, workload_cmd: list[str], *, recipe: str = DEFAULT_RECIPE, target: str = "localhost"
) -> list[str]:
    cmd = [
        str(apx),
        "recipe",
        "run",
        recipe,
        f"--workload={' '.join(workload_cmd)}",
        "--json",
        f"--target={target}",
        "--deploy-tools",
    ]
    if target == "localhost":
        cmd.append("--host-key-policy=ignore")
    return cmd


def build_render_cmd(apx: Path, run_id: str) -> list[str]:
    return [str(apx), "run", "render", str(run_id)]


def build_query_cmd(apx: Path, session_id: str, sql: str = HOTSPOTS_SQL) -> list[str]:
    return [str(apx), "render", "query", str(session_id), sql]


# --- parsing ------------------------------------------------------------------


def parse_hotspots(text: str, top_n: int = TOP_N) -> list[Hotspot]:
    """Parse apx output into ranked hotspots.

    Handles JSON (list/obj with function_name|function|symbol + percent|self_percent), the
    `┃`-delimited Unicode table apx render emits, and plain CSV. Returns top-N by percent desc.
    """
    text = (text or "").strip()
    if not text:
        return []
    if text[0] in "[{":
        spots = _parse_json_hotspots(text)
    elif "┃" in text:
        spots = _rows_to_hotspots(_box_rows(text))
    else:
        spots = _rows_to_hotspots([r for r in csv.reader(io.StringIO(text)) if r])
    spots.sort(key=lambda h: h.percent, reverse=True)
    return spots[:top_n]


def _to_float(value) -> float:
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return 0.0


def _first(d: dict, keys, default=None):
    """First present (non-None) value among keys — distinguishes a real 0 from missing."""
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


def _parse_json_hotspots(text: str) -> list[Hotspot]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict):
        rows = obj.get("hotspots") or obj.get("rows") or obj.get("functions") or []
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = []
    out: list[Hotspot] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        fn = _first(r, ("function_name", "function", "symbol", "name"), "")
        pct = _first(
            r, ("periodic_samples_self_percent", "percent", "self_percent", "self", "overhead"), 0
        )
        out.append(
            Hotspot(function=str(fn), percent=_to_float(pct), module=str(r.get("module", "")))
        )
    return out


def _box_rows(text: str) -> list[list[str]]:
    rows = []
    for ln in text.splitlines():
        if "┃" not in ln:
            continue
        cells = [c.strip() for c in ln.strip().strip("┃").split("┃")]
        if any(cells):
            rows.append(cells)
    return rows


def _find_col(header: list[str], *names) -> int | None:
    for i, h in enumerate(header):
        if any(n in h for n in names):
            return i
    return None


def _find_percent_col(header: list[str]) -> int | None:
    for i, h in enumerate(header):
        if "percent" in h or "%" in h:
            return i
    return _find_col(header, "self", "overhead", "cpu")


def _rows_to_hotspots(rows: list[list[str]]) -> list[Hotspot]:
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    fi = _find_col(header, "function", "symbol", "name")
    pi = _find_percent_col(header)
    mi = _find_col(header, "module", "object", "dso", "image")
    if fi is None or pi is None:
        return []
    out: list[Hotspot] = []
    for r in rows[1:]:
        if len(r) <= max(fi, pi):
            continue
        out.append(
            Hotspot(
                function=r[fi].strip(),
                percent=_to_float(r[pi]),
                module=r[mi].strip() if mi is not None and len(r) > mi else "",
            )
        )
    return out


def _extract_id(text: str, keys: tuple[str, ...]) -> str | None:
    """Pull a run_id / session_id from apx output. TODO(confirm) exact JSON shape (doc 111566)."""
    text = (text or "").strip()
    if text and text[0] in "[{":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            for k in keys:
                if obj.get(k) not in (None, ""):
                    return str(obj[k])
    for k in keys:
        # Anchor the key so 'id' doesn't match inside 'session_id'/'run_id'.
        m = re.search(rf'(?<![A-Za-z0-9_]){re.escape(k)}"?\s*[:=]\s*"?([\w.-]+)', text)
        if m:
            return m.group(1)
    return None


class PerformixProfiler:
    """Thin, testable wrapper around the `apx` CLI."""

    def __init__(self) -> None:
        self._apx = find_apx()

    def available(self) -> bool:
        """True only when we are on Arm Linux AND `apx` is installed."""
        return is_arm_linux() and self._apx is not None

    def unavailable_reason(self) -> str:
        if not is_arm_linux():
            return "not running on Arm Linux (Performix is an Arm Neoverse tool)"
        if self._apx is None:
            return "`apx` not found (install: https://learn.arm.com/install-guides/performix/)"
        return ""

    def _skip(self, reason: str, target: str, stamp: str, raw: str = "") -> ProfileResult:
        return ProfileResult(
            skipped=True, reason=reason, timestamp=stamp, target=target, raw_output=raw
        )

    def profile(
        self,
        command: list[str],
        *,
        recipe: str = DEFAULT_RECIPE,
        target: str = "localhost",
        timeout: float = 1800.0,
    ) -> ProfileResult:
        """Profile `command` with Performix and return the top hotspots.

        Never raises: off Arm / without `apx` / on an invocation failure it returns a `skipped`
        ProfileResult so the report keeps working.
        """
        target_str = " ".join(command)
        stamp = datetime.now(UTC).isoformat()
        if not self.available():
            return self._skip(self.unavailable_reason(), target_str, stamp)

        apx = self._apx
        assert apx is not None
        try:
            with tempfile.TemporaryDirectory():
                run = subprocess.run(
                    build_recipe_run_cmd(apx, command, recipe=recipe, target=target),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
                hotspots = parse_hotspots(run.stdout)
                raw = run.stdout[:4000]

                # If --json didn't already embed hotspots, do the render -> query flow.
                if not hotspots:
                    run_id = _extract_id(run.stdout, ("run_id", "runId", "id"))
                    if run_id:
                        rnd = subprocess.run(
                            build_render_cmd(apx, run_id),
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                            check=True,
                        )
                        session_id = _extract_id(rnd.stdout, ("session_id", "sessionId", "id"))
                        if session_id:
                            q = subprocess.run(
                                build_query_cmd(apx, session_id),
                                capture_output=True,
                                text=True,
                                timeout=timeout,
                                check=True,
                            )
                            hotspots = parse_hotspots(q.stdout)
                            raw = q.stdout[:4000]

            if not hotspots:
                return self._skip(
                    "apx ran but no hotspots parsed — the run_id/session_id plumbing or output "
                    f"format is TODO(confirm); see {PERFORMIX_DOC} and {ARM_MCP}.",
                    target_str,
                    stamp,
                    raw,
                )
            return ProfileResult(
                skipped=False,
                timestamp=stamp,
                target=target_str,
                recipe=recipe,
                hotspots=hotspots,
                raw_output=raw,
            )
        except Exception as exc:  # noqa: BLE001 - any apx failure degrades to a clear skip
            return self._skip(
                f"apx invocation failed ({type(exc).__name__}: {exc}). Exact apx flow is "
                f"TODO(confirm) — see {PERFORMIX_DOC} and {ARM_MCP}.",
                target_str,
                stamp,
            )


def profile_filename(timestamp: str) -> str:
    safe = timestamp.replace(":", "").replace("-", "").replace(".", "")
    return f"profile_{safe}.json"
