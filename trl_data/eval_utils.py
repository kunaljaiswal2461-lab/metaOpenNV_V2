"""
Pure helpers for ``scripts/eval_llm_on_env.py``.

Kept in :mod:`trl_data` (rather than under ``scripts/``) so that they are
trivially importable from tests and from notebooks. No torch / transformers /
matplotlib dependency; only stdlib.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

# Valid action digits emitted by the env contract.
_VALID_ACTION_CHARS: frozenset[str] = frozenset("012")
# Default chosen when the model produces nothing parsable.
_FALLBACK_ACTION: int = 0


def parse_action(text: str) -> int:
    """
    Pick the first ``0/1/2`` character in ``text``; fall back to ``0`` on miss.

    Robust to leading whitespace, newlines, prefatory natural language
    ("The action is 1"), or noisy decoding ("9 then 1"). Anything not in
    ``{0, 1, 2}`` is ignored, so a stray "3" never accidentally becomes a
    valid action.
    """
    if not text:
        return _FALLBACK_ACTION
    for ch in text:
        if ch in _VALID_ACTION_CHARS:
            return int(ch)
    return _FALLBACK_ACTION


def parse_models(specs: Sequence[str]) -> Dict[str, str]:
    """
    Turn CLI ``--models`` entries into a stable ``{name: identifier}`` map.

    Accepts either ``name=identifier`` (preferred) or just ``identifier``.
    The auto-derived name is the identifier with ``/`` and ``:`` replaced
    by ``_`` so it is filesystem-friendly. Names must be unique.
    """
    out: Dict[str, str] = {}
    for raw in specs:
        if not raw:
            continue
        if "=" in raw:
            name, ident = raw.split("=", 1)
            name = name.strip()
            ident = ident.strip()
        else:
            ident = raw.strip()
            name = ident.replace("/", "_").replace(":", "_")
        if not name or not ident:
            raise ValueError(f"Could not parse model spec: {raw!r}")
        if name in out:
            raise ValueError(f"Duplicate model name: {name!r}")
        out[name] = ident
    if not out:
        raise ValueError("At least one --models entry is required.")
    return out


def format_metrics_md(
    rows: Iterable[Mapping[str, Any]],
    *,
    task_name: str,
    prompt_style: str,
    teacher: str,
    episodes: int,
) -> str:
    """
    Render the per-model metrics table as Markdown identical to the format
    used in :mod:`eval.phase4_benchmark` so judges see consistent numbers.
    """
    rows_list: List[Mapping[str, Any]] = list(rows)
    header = [
        "# Phase 3 - LLM on env evaluation",
        "",
        f"task={task_name} prompt={prompt_style} teacher={teacher} episodes={episodes}",
        "",
        "| Model | Mean episode reward | Std | Last final PV | HOLD | BUY | SELL | Teacher agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    body = []
    for r in rows_list:
        body.append(
            "| {name} | {mean_r:.6f} | {std_r:.6f} | {final_pv:.2f} | "
            "{h_pct:.1f}% | {b_pct:.1f}% | {s_pct:.1f}% | {teacher_agreement:.3f} |".format(
                name=r["name"],
                mean_r=r["mean_r"],
                std_r=r["std_r"],
                final_pv=r["final_pv"],
                h_pct=r["h_pct"] * 100.0,
                b_pct=r["b_pct"] * 100.0,
                s_pct=r["s_pct"] * 100.0,
                teacher_agreement=r["teacher_agreement"],
            )
        )
    return "\n".join(header + body) + "\n"


__all__ = [
    "format_metrics_md",
    "parse_action",
    "parse_models",
]
