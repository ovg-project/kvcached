import resource
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


# Ensure this process has a sufficiently high file-descriptor limit.
def set_ulimit(target_soft_limit=1048576):
    try:
        soft_nofile, hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_nofile < target_soft_limit:
            new_soft = min(target_soft_limit, hard_nofile)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard_nofile))
            logger.info("Raised RLIMIT_NOFILE from %s to %s", soft_nofile,
                        new_soft)
    except Exception as _e:
        logger.warning("Could not raise RLIMIT_NOFILE: %s", _e)


def ensure_tmux_session(session: str) -> bool:
    """Ensure a detached tmux session named *session* exists.

    If the session already exists, interactively ask the user whether to kill
    and restart it.  Returns *True* if the session is ready to use, *False* if
    the user chose to skip.
    """

    try:
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Session exists – prompt user
        response = input(
            f"Tmux session '{session}' already exists. Kill it and restart? (y/N): "
        )
        if response.lower() == "y":
            logger.info("Killing existing tmux session: %s", session)
            subprocess.run(["tmux", "kill-session", "-t", session], check=True)
        else:
            logger.info("Skipping launch for session: %s", session)
            return False
    except subprocess.CalledProcessError:
        # Session does not exist – we'll create it below
        pass

    # Create new detached session with a decent window size
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-x",
            "120",
            "-y",
            "30",
        ],
        check=True,
    )

    # Configure helpful defaults
    subprocess.run(
        ["tmux", "set-option", "-t", session, "history-limit", "999999"],
        check=True)
    subprocess.run(["tmux", "set-option", "-t", session, "mouse", "on"],
                   check=True)

    return True


def ensure_tmux_session_auto_kill(session: str) -> bool:
    """Ensure a detached tmux session exists, automatically kill existing one.
    
    This is similar to ensure_tmux_session but automatically kills existing
    sessions without prompting, useful for automated scripts.
    """
    
    try:
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Session exists – automatically kill it
        logger.info("Killing existing tmux session: %s", session)
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    except subprocess.CalledProcessError:
        # Session does not exist – we'll create it below
        pass

    # Create new detached session with a decent window size
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-x",
            "120",
            "-y",
            "30",
        ],
        check=True,
    )

    # Configure helpful defaults with increased history for better log capture
    subprocess.run(
        ["tmux", "set-option", "-t", session, "history-limit", "50000"],
        check=True)
    subprocess.run(["tmux", "set-option", "-t", session, "mouse", "on"],
                   check=True)

    return True


def collect_env_mods(inst: Dict[str, Any]) -> Dict[str, str]:
    """Collect environment overrides from an *instance* configuration dict."""

    env: Dict[str, str] = {}

    for kv in inst.get("engine_env", []) + inst.get("kvcached_env", []):
        if "=" not in kv:
            raise ValueError(f"Invalid env entry (expected KEY=VALUE): {kv}")
        key, value = kv.split("=", 1)
        env[key] = value

    return env


def launch_in_tmux(
    session: str,
    window_name: str,
    cmd: List[str],
    env_mod: Dict[str, str],
    inst: Dict[str, Any],
) -> None:
    """Launch *cmd* inside its own tmux window with optional env overrides."""

    # Prepare environment exports
    env_exports = "".join(f"export {k}={shlex.quote(str(v))}; "
                          for k, v in env_mod.items())

    cmd_str = shlex.join(cmd)

    # If virtualenv activation is requested, prepend the activation command
    if inst.get("using_venv") and inst.get("venv_path"):
        venv_activate = Path(
            inst["venv_path"]).expanduser().resolve() / "bin" / "activate"
        activate_cmd = f"source {shlex.quote(str(venv_activate))}; "
    else:
        activate_cmd = ""

    # Bump the per-process file-descriptor limit inside the tmux pane.  We
    # regularly hit the default soft limit (1024) when the router handles
    # many concurrent streaming connections.
    ulimit_cmd = "ulimit -n 1048576; "

    full_cmd = f"{ulimit_cmd}{activate_cmd}{env_exports}{cmd_str}"

    logger.debug("Command for %s: %s", window_name, full_cmd)

    # Create new window in the session
    subprocess.run(
        [
            "tmux",
            "new-window",
            "-t",
            session,
            "-n",
            window_name,
            "bash",
            "-c",
            f"echo 'Starting {window_name}...'; {full_cmd}; echo 'Press Enter to close...'; read",
        ],
        check=True,
    )

    # Remove the default window (index 0) if it still exists to keep the session tidy
    try:
        subprocess.run(["tmux", "kill-window", "-t", f"{session}:0"],
                       check=True)
    except subprocess.CalledProcessError:
        pass
