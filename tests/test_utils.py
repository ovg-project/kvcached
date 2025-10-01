# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for test files.
"""

from pathlib import Path

import yaml


def load_example_config():
    """Load the example configuration file in controller folder used by tests.

    Returns:
        Dict containing the parsed YAML configuration from example-config.yaml
    """
    # Find the controller directory relative to the test file
    test_dir = Path(__file__).parent
    config_path = test_dir.parent / "controller" / "example-config.yaml"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config
