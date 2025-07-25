import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from utils import collect_env_mods, ensure_tmux_session, launch_in_tmux


def load_config(config_path: str = "example-config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict) or "instances" not in config:
            print(
                "Error: YAML configuration must contain an 'instances' list.")
            sys.exit(1)

        # Build a mapping of model_name -> instance_details
        models = {}
        for instance in config["instances"]:
            model_name = instance.get("model") or instance.get("name")
            if model_name is None:
                continue
            models[model_name] = instance

        router_cfg = config.get("router", {})
        router_port = router_cfg.get("router_port")

        if router_port is None:
            print(
                "Error: 'router_port' not specified under 'router' section in config."
            )
            sys.exit(1)

        return models, router_port

    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(
            f"Error: Invalid YAML in configuration file '{config_path}': {str(e)}"
        )
        sys.exit(1)


def _launch_benchmark_clients_tmux(models_config, router_port: int):
    """Launch each benchmark client in its own tmux session using the router port."""

    script_dir = (Path(__file__).parent /
                  "../engine_integration/benchmark").resolve()

    for model_name, inst in models_config.items():
        inst_name = inst.get("name") or model_name.replace("/", "-")
        session_name = f"benchmark-{inst_name}"

        if not ensure_tmux_session(session_name):
            print(
                f"Skipping {inst_name} - tmux session already running and user chose not to recreate."
            )
            continue

        engine = inst.get("engine", "sglang")

        start_client_sh = script_dir / "start_client.sh"

        cmd = [
            str(start_client_sh),
            engine,
            str(router_port),
            model_name,
        ]

        env_mod = collect_env_mods(inst)

        try:
            launch_in_tmux(session_name, f"{inst_name}-client", cmd, env_mod,
                           inst)
            print(
                f"Launched benchmark client for {model_name} (engine {engine}) in tmux session '{session_name}'."
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to launch benchmark client for {model_name}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Launch benchmark clients in tmux against the router")
    parser.add_argument(
        "--config",
        default="example-config.yaml",
        help="Configuration file path (default: example-config.yaml)",
    )

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    models_config, router_port = load_config(args.config)

    _launch_benchmark_clients_tmux(models_config, router_port)


if __name__ == "__main__":
    main()
