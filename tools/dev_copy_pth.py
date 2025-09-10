#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 contributors to the kvcached project
#
# SPDX-License-Identifier: Apache-2.0

"""
Copy kvcached_autopatch.pth into the active site-packages.
Used in dev mode when kvcached is installed as editable.

Usage:
    python dev_copy_pth.py
"""

import shutil
import site
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
KVCACHED_DIR = SCRIPT_DIR.parent


def main():
    # Locate site-packages
    site_dirs = site.getsitepackages()
    if not site_dirs:
        # fallback: venvs sometimes only have getusersitepackages
        site_dirs = [site.getusersitepackages()]

    target_dir = Path(site_dirs[0])
    src = KVCACHED_DIR / "kvcached_autopatch.pth"
    dst = target_dir / src.name

    if not src.exists():
        sys.exit(
            f"ERROR: {src} not found. Check where kvcached_autopatch.pth is located."
        )

    print(f"Copying {src} → {dst}")
    shutil.copy2(src, dst)
    print("Done. Restart Python to verify it runs.")


if __name__ == "__main__":
    main()
