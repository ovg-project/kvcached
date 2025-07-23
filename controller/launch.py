import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()
logger.setLevel(logging.INFO)


def _parse_cfg(cfg: Dict[str, Any], config_dir: Path) -> List[Dict[str, Any]]:
    """Parse and validate the YAML configuration.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Raw YAML dict.
    config_dir : Path
        Directory of the YAML file (used for resolving relative paths).

    Returns
    -------
    List[Dict[str, Any]]
        A list of normalized instance configuration dictionaries.
    """
    if "instances" not in cfg or not isinstance(cfg["instances"], list):
        raise ValueError(
            "Configuration must contain a top-level 'instances' list.")

    parsed: List[Dict[str, Any]] = []
    for inst in cfg["instances"]:
        if not isinstance(inst, dict):
            raise ValueError(
                "Each element of 'instances' must be a mapping (dictionary).")

        # Required keys
        missing = {"model", "engine"} - set(inst)
        if missing:
            raise ValueError(
                f"Instance is missing required keys: {', '.join(missing)} - {inst}"
            )

        raw_args = inst.get("engine_args", inst.get("args", []))

        # Parse args: allow each list element to contain spaces which should be split
        if isinstance(raw_args, list):
            args_list: List[str] = []
            for item in raw_args:
                if isinstance(item, (list, tuple)):
                    args_list.extend(map(str, item))
                else:
                    # Use shlex.split to respect quoting
                    args_list.extend(shlex.split(str(item)))
        else:
            # Single string provided
            args_list = shlex.split(str(raw_args))

        inst_cfg: Dict[str, Any] = {
            "name": inst.get("name", inst['model'].split('/')[-1]),
            "model": inst["model"],
            "engine": inst["engine"].lower(),
            "args": args_list,
            "engine_env": inst.get("engine_env", []),
            "kvcached_env": inst.get("kvcached_env", []),
            "using_venv": bool(inst.get("using_venv", False)),
            "venv_path": inst.get("venv_path"),
        }

        # Validate env lists
        for _field in ("engine_env", "kvcached_env"):
            if not isinstance(inst_cfg[_field], list):
                raise ValueError(
                    f"'{_field}' must be a list of KEY=VALUE strings")

        # Validate venv settings
        if inst_cfg["using_venv"]:
            if not inst_cfg["venv_path"]:
                raise ValueError(
                    "'venv_path' must be provided when 'using_venv' is true")
            inst_cfg["venv_path"] = Path(
                inst_cfg["venv_path"]).expanduser().resolve()

        parsed.append(inst_cfg)

    return parsed


def _build_command(inst: Dict[str, Any]) -> List[str]:
    engine = inst["engine"]

    # Determine which virtualenv to use
    if inst.get("using_venv") and inst.get("venv_path"):
        venv_dir = Path(inst["venv_path"]).expanduser().resolve()

    def _vbin(program: str) -> Path:
        return venv_dir / "bin" / program

    if engine == "vllm":
        python_exec = _vbin("python")
        # Always launch via module to avoid relying on the 'vllm' CLI entrypoint script
        cmd = [
            str(python_exec),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            inst["model"],
        ] + inst["args"]
    elif engine in {"sgl", "sglang"}:
        python_exec = "python"  # Use system Python or venv-activated Python from PATH per user preference

        # Pass through all args from YAML config
        sgl_args = inst["args"]

        cmd = [
            str(python_exec),
            "-m",
            "sglang.launch_server",
            "--model-path",
            inst["model"],
        ] + sgl_args
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    return cmd


def _ensure_tmux_session(session: str) -> bool:
    """Ensure a detached tmux session named ``session`` exists.

    Returns:
        bool: True if session is ready for launching, False if user chose to skip
    """
    try:
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Session exists - prompt user
        response = input(
            f"Tmux session '{session}' already exists. Kill it and restart? (y/N): "
        )
        if response.lower() == 'y':
            logger.info("Killing existing tmux session: %s", session)
            subprocess.run(["tmux", "kill-session", "-t", session], check=True)
        else:
            logger.info("Skipping launch for session: %s", session)
            return False
    except subprocess.CalledProcessError:
        # Session does not exist - continue to create it
        pass

    # Create new session detached
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
            "30"  # Set window size
        ],
        check=True)
    # Configure scrollback and other settings
    subprocess.run(
        ["tmux", "set-option", "-t", session, "history-limit", "999999"],
        check=True)
    subprocess.run(["tmux", "set-option", "-t", session, "mouse", "on"],
                   check=True)

    return True


def _collect_env_mods(inst: Dict[str, Any]) -> Dict[str, str]:
    env: Dict[str, str] = {}

    for kv in inst.get("engine_env", []) + inst.get("kvcached_env", []):
        if "=" not in kv:
            raise ValueError(f"Invalid env entry (expected KEY=VALUE): {kv}")
        key, value = kv.split("=", 1)
        env[key] = value

    return env


def _launch_in_tmux(session: str, window_name: str, cmd: List[str],
                    env_mod: Dict[str, str], inst: Dict[str, Any]) -> None:
    """Launch ``cmd`` inside its own tmux window with overridden env vars."""
    # Prepare environment exports
    env_exports = "".join(f"export {k}={shlex.quote(str(v))}; "
                          for k, v in env_mod.items())

    cmd_str = shlex.join(cmd)

    # If virtualenv activation is requested, prepend source command
    if inst.get("using_venv") and inst.get("venv_path"):
        venv_activate = Path(
            inst["venv_path"]).expanduser().resolve() / "bin" / "activate"
        activate_cmd = f"source {shlex.quote(str(venv_activate))}; "
    else:
        activate_cmd = ""

    full_cmd = f"{activate_cmd}{env_exports}{cmd_str}"

    logger.info("Command for %s: %s", window_name, full_cmd)

    # Ensure only one window per session: first create the real window, then remove the default index 0 if it still exists
    subprocess.run([
        "tmux", "new-window", "-t", session, "-n", window_name, "bash", "-c",
        f"echo 'Starting {window_name}...'; {full_cmd}; echo 'Press Enter to close...'; read"
    ],
                   check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KVCached controller launcher")
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    with cfg_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)

    try:
        instances_cfg = _parse_cfg(raw_cfg, cfg_path.parent)
    except Exception as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)

    launched_instances = []
    # Launch each instance in its own tmux session
    for inst in instances_cfg:
        session_name = f"kvcached-{inst['name']}"

        # Ensure tmux session exists (detached)
        if not _ensure_tmux_session(session_name):
            logger.info("Skipping launch for %s", inst["name"])
            continue

        cmd = _build_command(inst)
        env_mod = _collect_env_mods(inst)
        try:
            _launch_in_tmux(session_name, inst["name"], cmd, env_mod, inst)
            logger.info("Launched %s in tmux session '%s'", inst["name"],
                        session_name)
            launched_instances.append(inst)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to launch %s: %s", inst["name"], e)

    if launched_instances:
        logger.info(
            "All instances launched. Attach to individual sessions with:")
        for inst in launched_instances:
            session_name = f"kvcached-{inst['name']}"
            logger.info("  tmux attach -t %s  # for %s", session_name,
                        inst["name"])
    else:
        logger.info("No instances were launched.")


if __name__ == "__main__":
    main()
