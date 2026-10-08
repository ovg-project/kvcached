#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Progress deadlines and local checkpoints for read-only analysis stages.

Checkpoints detect accidental drift, not malicious replacement. They are local
operator-owned files; never restore sessions from downloaded PR artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from repair_engine_compat import run_command

WALL_SECONDS = 2700


class ProgressBudget:
    def __init__(self, started, *, work=1800, idle=600, recovery=300, wall=WALL_SECONDS):
        self.work, self.idle, self.recovery, self.wall = work, idle, recovery, wall
        self.start(started)

    def start(self, now):
        self.started = self.last_progress = now
        self.recovered = self.progress_recovered = 0.0
        self.reconnecting = None
        self.session = None
        self.events = 0
        self.offset = 0
        self.pending = b""

    def recovery_time(self, now):
        ongoing = 0 if self.reconnecting is None else now - self.reconnecting
        return min(self.recovery, self.recovered + ongoing)

    def feed(self, line, now):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            return
        if not isinstance(value, dict):
            return
        kind = value.get("type")
        if not isinstance(kind, str):
            return
        if kind == "thread.started":
            session = value.get("thread_id")
            try:
                if str(uuid.UUID(session)) != session:
                    raise ValueError
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError("Invalid agent session identity") from exc
            if self.session is not None and self.session != session:
                raise ValueError("Agent session changed during execution")
            self.session = session
        elif kind == "error":
            message = str(value.get("message", "")).lower()
            if "reconnecting" in message or "stream disconnected" in message:
                if self.reconnecting is None:
                    self.reconnecting = now
        elif kind in {"item.started", "item.updated", "item.completed", "turn.completed"}:
            if kind != "turn.completed":
                item = value.get("item")
                if not isinstance(item, dict) or item.get("type") not in (
                    "command_execution", "mcp_tool_call", "agent_message", "reasoning",
                    "file_change", "web_search", "todo_list",
                ):
                    return
            self.recovered = self.recovery_time(now)
            self.reconnecting = None
            self.last_progress = now
            self.progress_recovered = self.recovered
            self.events += 1

    def poll(self, log, now):
        with log.open("rb") as stream:
            stream.seek(self.offset)
            chunk = stream.read(8 * 1024 * 1024)
            self.offset += len(chunk)
        self.pending += chunk
        lines = self.pending.split(b"\n")
        self.pending = lines.pop()
        if len(self.pending) > 8 * 1024 * 1024:
            raise ValueError("Agent event exceeds the log event limit")
        for line in lines:
            self.feed(line, now)

    def reason(self, now):
        elapsed = now - self.started
        recovery = self.recovery_time(now)
        if elapsed >= self.wall:
            return "wall_timeout"
        if elapsed - recovery >= self.work:
            return "work_timeout"
        if now - self.last_progress - (recovery - self.progress_recovered) >= self.idle:
            return "idle_timeout"
        return None

    def report(self, now):
        recovery = self.recovery_time(now)
        return dict(session=self.session, progress_events=self.events,
                    elapsed_seconds=round(now - self.started, 3),
                    recovery_seconds=round(recovery, 3),
                    work_seconds=round(now - self.started - recovery, 3),
                    transport_pending=self.reconnecting is not None)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def implementation_digest():
    root = Path(__file__).parent
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "engine_compat_agent.py", "engine_release_analysis.py", "repair_engine_compat.py")})


def save(path, value):
    if path.is_symlink():
        raise ValueError("Checkpoint must not be a symlink")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def read(path):
    if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Invalid checkpoint artifact")
    return json.loads(path.read_bytes())


def execute_stage(codex, cwd, output, name, inputs, *, resume=False):
    if not re.fullmatch(r"[a-z][a-z-]*", name):
        raise ValueError("Invalid analysis stage name")
    lock = output / f"{name}.lock"
    # Never steal an apparently stale lock: the previous process may still run.
    lock.open("x").close()
    try:
        return _execute_stage(codex, cwd, output, name, inputs, resume)
    finally:
        lock.unlink()


def _execute_stage(codex, cwd, output, name, inputs, resume):
    state_path, reply = output / f"{name}-state.json", output / f"{name}.json"
    implementation = implementation_digest()
    binding = digest(dict(inputs=inputs, implementation=implementation, cwd=str(cwd)))
    state = dict(binding=binding, status="new", attempts=[], session=None)
    if state_path.exists():
        if not resume:
            raise ValueError("Existing stage requires explicit --resume")
        state = read(state_path)
        if state["binding"] != binding:
            raise ValueError("Analysis checkpoint identity changed; start a fresh run")
        if state["status"] == "complete":
            value = read(reply)
            if digest(value) != state["reply_digest"]:
                raise ValueError("Completed agent response changed")
            return value
        if state["status"] != "interrupted":
            raise ValueError("Previous stage is not resumable; preserve it for inspection")
    elif reply.exists() or (output / f"{name}-attempts").exists():
        raise ValueError("Unbound existing analysis artifacts; start a fresh run")
    attempts = output / f"{name}-attempts"
    if attempts.is_symlink():
        raise ValueError("Attempt directory must not be a symlink")
    attempts.mkdir(exist_ok=True)
    number = len(state["attempts"]) + 1
    raw = attempts / f"{number:03d}.json"
    log = attempts / f"{number:03d}.log"
    schema = output / f"{name}-schema.json"
    save(schema, inputs["schema"])
    command = [codex, "exec", "--sandbox", "read-only", "--json", "--color", "never",
               "--output-schema", str(schema), "--output-last-message", str(raw)]
    session = state["session"]
    if session is not None:
        if str(uuid.UUID(session)) != session:
            raise ValueError("Invalid checkpoint session")
        command += ["resume", session]
    command += ["-"]
    prompt = inputs["prompt"]
    if session is not None:
        prompt = ("Continue this interrupted read-only stage using the existing work. "
                  "The source and input identity have been checked. Do not restart completed "
                  "investigation or waive any requirement. Return the requested final JSON.\n" + prompt)
    env = {k: v for k, v in os.environ.items() if k not in {"GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK"}}
    budget = ProgressBudget(0)
    state["status"] = "running"
    save(state_path, state)
    result = run_command(command, cwd, log, WALL_SECONDS, env, prompt, watch=budget)
    state["attempts"].append(result)
    if implementation_digest() != implementation:
        state["status"] = "failed"
        save(state_path, state)
        raise ValueError("Analysis controller changed during execution")
    activity = result.get("activity", {})
    if session is not None and activity.get("session") not in (None, session):
        state["status"] = "failed"
        save(state_path, state)
        raise ValueError("Resumed agent session identity changed")
    state["session"] = activity.get("session") or session
    resumable = result.get("timeout") or activity.get("transport_pending")
    state["status"] = "interrupted" if resumable and result.get("timeout_reason") != "protocol_error" else "failed"
    save(state_path, state)
    if result["exit_code"] != 0:
        raise ValueError(f"{name} {state['status']}: {result.get('timeout_reason') or 'agent exit'}; preserve evidence")
    value = read(raw)
    save(reply, value)
    state.update(status="complete", reply_digest=digest(value))
    save(state_path, state)
    return value
