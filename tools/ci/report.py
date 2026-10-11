#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Write the GPU CI results issue: one comment per run, and the setup of
each workflow in the issue description; and the comment of a run that a
maintainer started on a pull request (--pr), which carries its own setup.

Everything here is rendered from the files a run leaves in its results
directories (setup.json and summary.json from tests/e2e/run.py and
benchmarks/ci/perf.py), so the comment and the description describe what the
run actually did. A correctness run has one directory per part that ran on
its own GPUs: vllm and sglang for nightly, all otherwise.

Usage:
    report.py correctness --results DIR [DIR ...] --profile nightly --comment c.md \\
        --section s.md --run-url URL --sha SHA [--subject S] [--prev SHA:CONCLUSION] \\
        [--artifact-url URL] [--pr N]
    report.py perf --results DIR --comment c.md --section s.md --run-url URL --sha SHA ...
    report.py splice --body body.md --name nightly --section s.md --out new.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_URL = "https://github.com/ovg-project/kvcached"
# Sections of the issue description, in this order, one per workflow.
SECTIONS = ("nightly", "weekly", "perf")
TITLES = {"nightly": "Nightly correctness", "weekly": "Weekly correctness",
          "smoke": "Smoke correctness", "perf": "Weekly performance"}
SCHEDULES = {"nightly": "Daily at 08:00 UTC", "weekly": "Saturdays at 12:00 UTC",
             "perf": "Sundays at 10:00 UTC"}
WORKFLOWS = {"nightly": "gpu-correctness.yml", "weekly": "gpu-correctness.yml",
             "smoke": "gpu-correctness.yml", "perf": "gpu-perf-weekly.yml"}
ENGINE_NAMES = {"vllm": "vLLM", "sglang": "SGLang"}
LAYOUTS = {"c1": "kvcached, contiguous", "c0": "kvcached, non-contiguous", None: "native"}
MAX_FAILED_ROWS = 30
MAX_COMMENT = 60000  # GitHub's limit is 65536 characters


def load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def short(sha: Optional[str]) -> str:
    return (sha or "unknown")[:7]


def code(text: str) -> str:
    return f"`{text}`" if text else ""


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


# --- setup -----------------------------------------------------------------

def machine_line(setup: dict[str, Any]) -> str:
    """The GPUs of a run, and how its parts shared them."""
    line = gpus_line(setup.get("gpus", []))
    parts = setup.get("parts", [])
    if len(parts) > 1:
        line += f"; the {' and '.join(parts)} cases run at the same time, one GPU each"
    return line


def gpus_line(gpus: list[dict[str, str]]) -> str:
    if not gpus:
        return "GPU unknown"
    g = gpus[0]
    return f"{len(gpus)}x {g.get('name', '?')} ({g.get('memory', '?')}, driver {g.get('driver', '?')})"


def revision_link(hf_id: str, revision: Optional[str]) -> str:
    if not revision:
        return f"`{hf_id}`"
    return f"`{hf_id}` @ [{revision[:7]}](https://huggingface.co/{hf_id}/tree/{revision})"


def model_flags(s: dict[str, Any]) -> str:
    """The model's flags and the case's own, showing only the one that takes
    effect when the case overrides a model flag (the last one wins)."""
    extra = s.get("extra_args", [])
    overridden = {a for a in extra if a.startswith("--")}
    args, i, model_args = [], 0, s.get("model_args", [])
    while i < len(model_args):
        flag = model_args[i]
        has_value = i + 1 < len(model_args) and not model_args[i + 1].startswith("--")
        if flag not in overridden:
            args += model_args[i:i + 1 + has_value]
        i += 1 + has_value
    return code(" ".join([*args, *extra]))


def case_brief(s: dict[str, Any], setup: dict[str, Any]) -> str:
    """One line describing a case, for the table of failed cases."""
    revisions = setup.get("model_revisions", {})
    if s.get("kind") == "elastic":
        a, b = s["a"], s["b"]
        return (f"{ENGINE_NAMES[a['engine']]} {a['hf_id']} + {ENGINE_NAMES[b['engine']]} "
                f"{b['hf_id']} on GPU {s.get('gpu', '0')}, batch limit {s.get('batch')}")
    engine = setup.get("engines", {}).get(s["engine"], {})
    parts = [f"{ENGINE_NAMES[s['engine']]} {engine.get('engine') or ''}".strip(),
             f"{s['hf_id']}@{short(revisions.get(s['hf_id']))}", LAYOUTS[s.get("layout")],
             f"TP {s.get('tp', 1)}", f"KV {s.get('kv_cache_dtype') or 'model dtype'}"]
    if s.get("extra_args"):
        parts.append(f"`{' '.join(s['extra_args'])}`")
    return ", ".join(parts)


def engines_table(setup: dict[str, Any]) -> list[str]:
    lines = ["| engine | image | versions |", "|---|---|---|"]
    for name, info in setup.get("engines", {}).items():
        digest = info.get("digest") or ""
        digest = f" (`{digest.split('@')[-1][:19]}…`)" if "@" in digest else ""
        labels = {"engine": name, "kvcached": "kvcached", "torch": "torch", "cuda": "CUDA"}
        versions = ", ".join(f"{label} {info[k]}" for k, label in labels.items() if info.get(k))
        lines.append(f"| {ENGINE_NAMES.get(name, name)} | `{info.get('image', '?')}`{digest} "
                     f"| {versions} |")
    return lines


def env_lines(setup: dict[str, Any]) -> list[str]:
    """How every server of an engine is started, and what kvcached and
    native cases add to that."""
    lines = []
    cases = setup.get("cases", []) + [s[k] for s in setup.get("elastic", []) for k in "ab"]
    for name, info in setup.get("engines", {}).items():
        engine_env = dict(info.get("env") or {})  # a dict, or [key, value] pairs
        common = " ".join(f"{k}={v}" for k, v in engine_env.items())
        lines.append(f"- Every {ENGINE_NAMES.get(name, name)} server: "
                     f"`{(common + ' ' + info.get('server', '')).strip()}`, "
                     "followed by the case's flags.")
        for kv, label in ((True, "kvcached"), (False, "native")):
            envs = [c.get("env", {}) for c in cases
                    if c["engine"] == name and bool(c.get("layout")) == kv]
            if not envs:
                continue
            shared = {k: v for k, v in envs[0].items()
                      if k not in engine_env and all(e.get(k) == v for e in envs)}
            varying = sorted({k for e in envs for k in e} - set(shared) - set(engine_env))
            text = " ".join(f"{k}={v}" for k, v in shared.items())
            extra = f"; `{'`, `'.join(varying)}` follows the KV layout" if varying else ""
            ipc = " and a per-server `KVCACHED_IPC_NAME`" if kv else ""
            lines.append(f"  - {label} cases add `{text}`{ipc}{extra}.")
    return lines


def cases_table(setup: dict[str, Any]) -> list[str]:
    revisions = setup.get("model_revisions", {})
    lines = ["| case | model | KV cache | TP | KV dtype | case flags |",
             "|---|---|---|---|---|---|"]
    for s in setup.get("cases", []):
        lines.append(f"| {s['name']} | {revision_link(s['hf_id'], revisions.get(s['hf_id']))} "
                     f"| {LAYOUTS[s.get('layout')]} | {s.get('tp', 1)} "
                     f"| {s.get('kv_cache_dtype') or 'model dtype'} | {model_flags(s)} |")
    return lines


def elastic_table(setup: dict[str, Any]) -> list[str]:
    revisions = setup.get("model_revisions", {})
    lines = ["| case | GPU | server A | server B |", "|---|---|---|---|"]
    for s in setup.get("elastic", []):
        servers = [f"{ENGINE_NAMES[x['engine']]}, {revision_link(x['hf_id'], revisions.get(x['hf_id']))}"
                   f", {LAYOUTS[x.get('layout')]}, {model_flags(x)}" for x in (s["a"], s["b"])]
        lines.append(f"| {s['name']} | {s.get('gpu', '0')} | {servers[0]} | {servers[1]} |")
    return lines


def checks_table(checks: dict[str, str]) -> list[str]:
    return ["| check | passes when |", "|---|---|"] + [
        f"| `{name}` | {cell(text)} |" for name, text in checks.items()]


def part_name(results: Path) -> str:
    """The part of a run a results directory holds: <dir>/vllm, sglang or all."""
    return ENGINE_NAMES.get(results.name, results.name)


def merge_setups(parts: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """One setup for a run whose parts ran at the same time on separate GPUs."""
    setups = [s for _, s in parts]
    merged = dict(setups[0])
    merged["gpus"] = [g for s in setups for g in s.get("gpus", [])]
    merged["engines"] = {k: v for s in setups for k, v in s.get("engines", {}).items()}
    merged["model_revisions"] = {k: v for s in setups
                                 for k, v in s.get("model_revisions", {}).items()}
    for key in ("cases", "elastic"):
        merged[key] = [c for s in setups for c in s.get(key, [])]
    merged["parts"] = [name for name, _ in parts] if len(parts) > 1 else []
    return merged


def correctness_section(profile: str, setup: dict[str, Any], args: argparse.Namespace) -> str:
    # Only the setup: nothing that changes from run to run, so that the
    # description is edited only when the setup changes.
    if args.pr:
        when = f"On request in a pull request ([gpu-pr.yml]({REPO_URL}/blob/main/.github/workflows/gpu-pr.yml))"
        source = "the pull request merged into main"
    else:
        when = (f"{SCHEDULES.get(profile, 'On demand')} ([{WORKFLOWS[profile]}]"
                f"({REPO_URL}/blob/main/.github/workflows/{WORKFLOWS[profile]}))")
        source = "the latest main"
    lines = [
        f"## {TITLES[profile]}",
        "",
        f"{when} on {machine_line(setup)}. kvcached is installed from {source} into "
        "each engine's official image; each case starts one server in it.",
        "",
        *engines_table(setup),
        "",
        *env_lines(setup),
        "",
        "### Cases",
        "",
        "Each kvcached case is compared with the native case of the same engine and model.",
        "",
        *cases_table(setup),
        "",
    ]
    # The workload and checks come from the same code for every profile, so
    # only the nightly section of the issue spells them out.
    same = profile != "nightly" and not args.pr
    if same:
        lines += ["Each case runs the workload and checks described under "
                  "[Nightly correctness](#nightly-correctness)."]
    else:
        lines += ["Each case:", "", *[f"- {step}" for step in setup.get("workload", [])], "",
                  *checks_table(setup.get("checks", {}))]
    if setup.get("elastic"):
        lines += ["", "### Elastic tests", "", *elastic_table(setup), ""]
        if same:
            lines += ["Phases and checks as under Nightly correctness."]
        else:
            lines += [*[f"- {step}" for step in setup.get("elastic_workload", [])], "",
                      *checks_table(setup.get("elastic_checks", {}))]
    return "\n".join(lines) + "\n"


# --- comments --------------------------------------------------------------

def header(title: str, verdict: str, args: argparse.Namespace) -> list[str]:
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    subject = f" ({args.subject})" if args.subject else ""
    lines = [f"### {title} · {today} · {verdict}", "",
             f"[Run]({args.run_url}) · kvcached [{short(args.sha)}]({REPO_URL}/commit/{args.sha})"
             f"{subject}"]
    if args.prev:
        prev_sha, _, conclusion = args.prev.partition(":")
        state = {"success": "PASS", "failure": "FAIL"}.get(conclusion, conclusion or "?")
        if prev_sha == args.sha:
            lines[-1] += f" · same commit as the previous run ({state})"
        else:
            lines[-1] += (f" · [changes]({REPO_URL}/compare/{prev_sha}...{args.sha}) since the "
                          f"previous run at {short(prev_sha)} ({state})")
    return lines


def failed_cases(summary: dict[str, Any], setup: dict[str, Any]) -> list[str]:
    briefs = {s["name"]: case_brief(s, setup)
              for s in setup.get("cases", []) + setup.get("elastic", [])}
    failed = [c for c in summary.get("cases", []) if not c.get("ok")]
    lines = ["| case | setup | failed checks |", "|---|---|---|"]
    for case in failed[:MAX_FAILED_ROWS]:
        checks = "; ".join(f"`{c['name']}`" + (f" ({cell(c['detail'][:300])})" if c.get("detail")
                                                else "")
                           for c in case.get("checks", []) if not c.get("ok"))
        lines.append(f"| {case['name']} | {cell(briefs.get(case['name'], ''))} | {checks} |")
    if len(failed) > MAX_FAILED_ROWS:
        lines.append(f"\nand {len(failed) - MAX_FAILED_ROWS} more; see the run's job summary.")
    return lines


def logs_line(args: argparse.Namespace, days: int) -> str:
    where = f"[artifact]({args.artifact_url})" if args.artifact_url else f"[run]({args.run_url})"
    return (f"Logs: {where}, kept {days} days: serve.log and cmd.txt per case, summary.json "
            "with every request's tokens, setup.json, run.log, GPU state at the end.")


def reason(results: Path, gpu: str) -> str:
    """Why a run produced no results, without the provider's details: those
    are in the run log."""
    if (results / "error.txt").exists():
        return f"No {gpu} GPU could be obtained; the run log has the details."
    return "The tests did not finish; the run log has the details."


def correctness_comment(profile: str, dirs: list[Path], args: argparse.Namespace) -> str:
    title = TITLES.get(profile, profile)
    parts = [(d, load(d / "summary.json"), load(d / "setup.json") or {}) for d in dirs]
    ran = [(part_name(d), summary, setup) for d, summary, setup in parts if summary]
    missing = [part_name(d) for d, summary, _ in parts if not summary]
    if not ran:
        return "\n".join(header(title, "DID NOT RUN", args)
                         + ["", reason(dirs[0], "L4")]) + "\n"
    summary = {"cases": [c for _, run, _ in ran for c in run.get("cases", [])]}
    setup = merge_setups([(name, setup) for name, _, setup in ran])
    # The <engine>-setup entries are the kvcached installs, not test cases.
    cases = [c for c in summary["cases"] if not c["name"].endswith("-setup")]
    failed = [c for c in summary["cases"] if not c.get("ok")]
    not_run = "".join(f"; the {name} cases did not finish" for name in missing)
    verdict = (f"FAIL ({len(failed)} of {len(cases)} cases{not_run})" if failed or missing
               else "PASS")
    images = ", ".join(f"`{i.get('image')}`" for i in setup.get("engines", {}).values())
    lines = header(title, verdict, args) + [f"{machine_line(setup)} · {images}", ""]
    if missing:
        lines += [f"The {name} cases did not finish; the run log has the details."
                  for name in missing] + [""]
    lines += failed_cases(summary, setup) if failed else [f"All {len(cases)} cases passed."]
    if args.pr:
        # The pull request may change the cases, so its setup is its own.
        section = correctness_section(profile, setup, args).split("\n", 2)[2]
        lines += ["", "<details><summary>Setup of every case</summary>", "", section.strip(),
                  "", "</details>", "", logs_line(args, 30)]
    else:
        lines += ["", "Setup of every case: the issue description. " + logs_line(args, 30)]
    return "\n".join(lines)[:MAX_COMMENT] + "\n"


def perf_table(summary: dict[str, Any]) -> list[str]:
    table = summary.get("table", [])
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for t in table:
        rows.setdefault((t["model"], t["concurrency"]), {})[t["metric"]] = t

    def ratio(row: dict[str, Any], metric: str) -> str:
        r = row.get(metric, {}).get("ratio")
        return "-" if r is None else f"{r:.2f}"

    def value(row: dict[str, Any], metric: str, arm: str) -> str:
        v = row.get(metric, {}).get(arm)
        return "-" if v is None else f"{v:.0f}"

    lines = ["| model | concurrency | output tok/s, native | kvcached | ratio | mean TPOT ratio "
             "| p99 TPOT ratio | mean TTFT ratio |", "|---|---|---|---|---|---|---|---|"]
    for (model, c), row in rows.items():
        lines.append(f"| {model} | {c} | {value(row, 'output_throughput', 'native')} "
                     f"| {value(row, 'output_throughput', 'kvcached')} "
                     f"| {ratio(row, 'output_throughput')} | {ratio(row, 'mean_tpot_ms')} "
                     f"| {ratio(row, 'p99_tpot_ms')} | {ratio(row, 'mean_ttft_ms')} |")
    return lines


def idle_memory(summary: dict[str, Any]) -> list[str]:
    lines = []
    for model in dict.fromkeys(r["model"] for r in summary.get("runs", [])):
        idle = {arm: sorted(r["idle_gpu_mib"] for r in summary["runs"]
                            if r["model"] == model and r["arm"] == arm and "idle_gpu_mib" in r)
                for arm in ("native", "kvcached")}
        if idle["native"] and idle["kvcached"]:
            lines.append(f"- {model}: native {idle['native'][-1]} MiB, "
                         f"kvcached {idle['kvcached'][-1]} MiB")
    return lines


def perf_comment(results: Path, args: argparse.Namespace) -> str:
    perf = load(results / "perf" / "summary.json")
    title = TITLES["perf"]
    if not perf:
        return "\n".join(header(title, "DID NOT RUN", args)
                         + ["", reason(results, "H100")]) + "\n"
    setup = load(results / "perf" / "setup.json") or {}
    sandbox = load(results / "sandbox.json") or {}
    e2e, e2e_setup = load(results / "e2e" / "summary.json"), load(results / "e2e" / "setup.json")
    e2e_failed = [c for c in (e2e or {}).get("cases", []) if not c.get("ok")]
    errors = perf.get("errors", [])
    bad = len(errors) + len(e2e_failed)
    verdict = "FAIL" if bad or e2e is None else "PASS"
    versions = setup.get("versions", {})
    lines = header(title, verdict, args) + [
        f"{gpus_line(setup.get('gpus', []))} · `{sandbox.get('image', '?')}` "
        f"(vllm {versions.get('vllm', '?')})", "",
        "kvcached / native, the median of the two runs of each arm. A throughput ratio above 1 "
        "and a latency ratio below 1 favour kvcached.", "", *perf_table(perf)]
    memory = idle_memory(perf)
    if memory:
        lines += ["", "GPU memory in use with the server idle (the larger of the two runs):",
                  *memory]
    if errors:
        lines += ["", "Runs that failed:", *[f"- {cell(e[:500])}" for e in errors]]
    if e2e is None:
        lines += ["", "Hopper correctness: no results."]
    elif e2e_failed:
        lines += ["", "Hopper correctness:", "", *failed_cases(e2e, e2e_setup or {})]
    else:
        lines += ["", f"Hopper correctness: all {len(e2e.get('cases', []))} cases passed."]
    lines += ["", "Setup: the issue description. " + logs_line(args, 90)]
    return "\n".join(lines)[:MAX_COMMENT] + "\n"


def perf_section(results: Path, args: argparse.Namespace) -> Optional[str]:
    setup = load(results / "perf" / "setup.json")
    if not setup:
        return None
    sandbox = load(results / "sandbox.json") or {}
    versions = setup.get("versions", {})
    concurrency = setup.get("concurrency", {})
    models = ["| model | server flags |", "|---|---|"] + [
        f"| {revision_link(m['hf_id'], m.get('revision'))} | {code(' '.join(m.get('args', [])))} |"
        for m in setup.get("models", {}).values()]
    lines = [
        f"## {TITLES['perf']}",
        "",
        f"{SCHEDULES['perf']} ([{WORKFLOWS['perf']}]({REPO_URL}/blob/main/.github/workflows/"
        f"{WORKFLOWS['perf']})) on {gpus_line(setup.get('gpus', []))}, "
        f"{sandbox.get('cpu', '?')} CPUs, {sandbox.get('memory_gb', '?')} GB, image "
        f"`{sandbox.get('image', '?')}` (" + ", ".join(
            f"{'CUDA' if k == 'cuda' else k} {v}" for k, v in versions.items() if v)
        + "). kvcached is installed from the latest main.",
        "",
        "vLLM runs with its default configuration, Inductor included. Native and kvcached "
        f"alternate in the order {', '.join(setup.get('arms', []))}, one server start each; "
        "the results are the median of the two starts of each arm.",
        "",
        *models,
        "",
        f"- Server: `{setup.get('server', '')}`, followed by the model's flags; kvcached starts "
        f"add `{' '.join(f'{k}={v}' for k, v in setup.get('kvcached_env', {}).items())}`.",
        f"- Warm-up: {setup.get('warmup', {}).get('prompts')} prompts at concurrency "
        f"{setup.get('warmup', {}).get('concurrency')}, not measured.",
        f"- Measured: `{setup.get('bench', '')}` at concurrency "
        + ", ".join(f"{c} ({n} prompts)" for c, n in concurrency.items()) + ".",
    ]
    e2e_setup = load(results / "e2e" / "setup.json")
    if e2e_setup:
        lines += ["", "### Hopper correctness", "",
                  "The `hopper` profile of the correctness tests, on the same machine after the "
                  "performance runs; workload and checks as in the correctness sections.", "",
                  *cases_table(e2e_setup)]
    return "\n".join(lines) + "\n"


# --- issue description -----------------------------------------------------

def marker(name: str, end: bool = False) -> str:
    return f"<!-- gpu-ci:{name}:{'end' if end else 'start'} -->"


def splice(body: str, name: str, section: str) -> str:
    """Replace section `name` of the description, or insert it in SECTIONS
    order; everything outside the markers is kept."""
    block = f"{marker(name)}\n{section.strip()}\n{marker(name, end=True)}"
    start, end = marker(name), marker(name, end=True)
    if start in body and end in body:
        before, rest = body.split(start, 1)
        _, after = rest.split(end, 1)
        return before + block + after
    later = [n for n in SECTIONS[SECTIONS.index(name) + 1:] if marker(n) in body]
    if later:
        before, after = body.split(marker(later[0]), 1)
        return before + block + "\n\n" + marker(later[0]) + after
    return (body.rstrip() + "\n\n" if body.strip() else "") + block + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("correctness", "perf"):
        p = sub.add_parser(name)
        p.add_argument("--results", required=True, type=Path, nargs="+")
        p.add_argument("--comment", required=True, type=Path)
        p.add_argument("--section", required=True, type=Path)
        p.add_argument("--run-url", required=True)
        p.add_argument("--sha", required=True)
        p.add_argument("--subject", default="")
        p.add_argument("--prev", default="", help="SHA:CONCLUSION of the previous run")
        p.add_argument("--artifact-url", default="")
        if name == "correctness":
            p.add_argument("--profile", required=True)
            p.add_argument("--pr", type=int, help="the pull request the run tested")
    p = sub.add_parser("splice")
    p.add_argument("--body", required=True, type=Path)
    p.add_argument("--name", required=True, choices=SECTIONS)
    p.add_argument("--section", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    if a.cmd == "splice":
        a.out.write_text(splice(a.body.read_text(), a.name, a.section.read_text()))
        return 0
    if a.cmd == "correctness":
        a.comment.write_text(correctness_comment(a.profile, a.results, a))
        setups = [(part_name(d), load(d / "setup.json")) for d in a.results]
        # Only a complete run describes the setup; otherwise keep the last one.
        section = (correctness_section(a.profile, merge_setups(setups), a)
                   if all(s for _, s in setups) else None)
    else:
        a.comment.write_text(perf_comment(a.results[0], a))
        section = perf_section(a.results[0], a)
    # No section when the run produced no setup: the description keeps the last one.
    if section:
        a.section.write_text(section)
    elif a.section.exists():
        a.section.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
