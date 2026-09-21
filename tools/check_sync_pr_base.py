#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Refuse to update a sync branch proposed against another integration base."""

import argparse
import json
import subprocess


def check_base(repository: str, branch: str, base: str) -> None:
    completed = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repository,
            "--state",
            "open",
            "--head",
            branch,
            "--json",
            "number,baseRefName",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for pull in json.loads(completed.stdout):
        if pull["baseRefName"] != base:
            raise RuntimeError(
                f"PR #{pull['number']} targets {pull['baseRefName']}, not {base}; "
                "leave the existing branch unchanged and resolve the target first"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    check_base(args.repository, args.branch, args.base)
