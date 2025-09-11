# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from utils import collect_env_mods, ensure_tmux_session, launch_in_tmux

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _parse_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse and validate the YAML configuration.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Raw YAML dict.

    Returns
    -------
    List[Dict[str, Any]]
        A list of instances_config.
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


def _extract_models_mapping(
        raw_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build the model→endpoint mapping consumed by the router frontend."""
    models_mapping: Dict[str, Dict[str, Any]] = {}

    for inst in raw_cfg.get("instances", []):
        model_name = inst["model"]

        # Defaults
        host = "localhost"
        port = None

        raw_args = inst.get("engine_args", inst.get("args", []))
        if isinstance(raw_args, str):
            arg_list = shlex.split(raw_args)
        else:
            arg_list: List[str] = []
            for item in raw_args:
                arg_list.extend(shlex.split(str(item)))

        for idx, token in enumerate(arg_list):
            if token.startswith("--host="):
                host = token.split("=", 1)[1]
            elif token == "--host" and idx + 1 < len(arg_list):
                host = arg_list[idx + 1]
            elif token.startswith("--port="):
                try:
                    port = int(token.split("=", 1)[1])
                except ValueError:
                    pass
            elif token == "--port" and idx + 1 < len(arg_list):
                try:
                    port = int(arg_list[idx + 1])
                except ValueError:
                    pass

        if port is None:
            logger.warning(
                "Could not determine port for model %s – skipping in router mapping",
                model_name)
            continue

        models_mapping[model_name] = {"endpoint": {"host": host, "port": port}}

    return models_mapping


def _launch_instances(instances_cfg: List[Dict[str, Any]],
                      global_env: Dict[str, str]) -> List[Dict[str, Any]]:
    """Launch each configured model instance in its own dedicated tmux session."""
    launched: List[Dict[str, Any]] = []

    for inst in instances_cfg:
        session_name = f"kvcached-{inst['name']}"

        # Ensure tmux session exists (detached)
        if not ensure_tmux_session(session_name):
            logger.info("Skipping launch for %s", inst["name"])
            continue

        cmd = _build_command(inst)
        env_mod = {**global_env, **collect_env_mods(inst)}
        try:
            launch_in_tmux(session_name, inst["name"], cmd, env_mod, inst)
            logger.info(
                "Launched %s in tmux session '%s'. tmux attach -t %s to attach",
                inst["name"], session_name, session_name)
            launched.append(inst)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to launch %s: %s", inst["name"], e)

    return launched


def _maybe_launch_router(router_cfg: Dict[str, Any],
                         config_path: Path) -> None:
    """Conditionally launch the router/frontend if enabled in the config."""
    if not router_cfg.get("enable_router", False):
        logger.info("Router launch disabled via configuration.")
        return

    frontend_session = "kvcached-frontend"
    if not ensure_tmux_session(frontend_session):
        return

    frontend_port = router_cfg.get("router_port", 8080)

    frontend_cmd = [
        "python",
        "-u",
        str(Path(__file__).parent / "frontend.py"),
        "--config_path",
        str(config_path),
        "--port",
        str(frontend_port),
    ]

    try:
        launch_in_tmux(frontend_session, "frontend", frontend_cmd, {}, {})
        logger.info(
            "Launched frontend in tmux session '%s' (port %s). tmux attach -t %s  to attach",
            frontend_session, frontend_port, frontend_session)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to launch frontend: %s", e)


def main() -> None:
    """Entry point for the KVCached controller CLI."""
    parser = argparse.ArgumentParser(
        description="KVCached controller launcher")
    parser.add_argument("--config",
                        type=Path,
                        default="example-config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    with cfg_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)

    # Extract global env and router configuration
    global_env_cfg: Dict[str, Any] = raw_cfg.get("kvcached", {}) or {}
    global_kvcached_env = {
        str(k).upper(): str(v)
        for k, v in global_env_cfg.items()
    }
    router_cfg: Dict[str, Any] = raw_cfg.get("router", {}) or {}

    try:
        instances_cfg = _parse_cfg(raw_cfg)
    except Exception as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)

    _launch_instances(instances_cfg, global_kvcached_env)
    _maybe_launch_router(router_cfg, cfg_path)


if __name__ == "__main__":
    main()
