# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration loading for the tests."""

from pathlib import Path

import yaml


def load_example_config():
    """Load the example configuration file in controller folder used by tests.

    Returns:
        Dict containing the parsed YAML configuration from example-config.yaml
    """
    # tests/helpers/config.py -> repository root -> controller/
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "controller" / "example-config.yaml"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config
