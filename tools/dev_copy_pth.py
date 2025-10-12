#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Copy or remove kvcached_autopatch.pth in the active site-packages.
Used in dev mode when kvcached is installed as editable.

Usage:
    python dev_copy_pth.py           # Copy .pth file to site-packages
    python dev_copy_pth.py --remove  # Remove .pth file from site-packages
    python dev_copy_pth.py --check   # Check for stale .pth files
"""

import argparse
import shutil
import site
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
KVCACHED_DIR = SCRIPT_DIR.parent
PTH_FILENAME = "kvcached_autopatch.pth"


def get_site_packages():
    """Get the site-packages directory."""
    site_dirs = site.getsitepackages()
    if not site_dirs:
        # fallback: venvs sometimes only have getusersitepackages
        site_dirs = [site.getusersitepackages()]
    return [Path(d) for d in site_dirs]


def copy_pth():
    """Copy the .pth file to site-packages."""
    target_dirs = get_site_packages()
    target_dir = target_dirs[0]
    src = KVCACHED_DIR / PTH_FILENAME
    dst = target_dir / PTH_FILENAME

    if not src.exists():
        sys.exit(
            f"ERROR: {src} not found. Check where {PTH_FILENAME} is located."
        )

    print(f"Copying {src} → {dst}")
    shutil.copy2(src, dst)
    print("Done. Restart Python to verify it runs.")


def remove_pth():
    """Remove the .pth file from all site-packages directories."""
    target_dirs = get_site_packages()
    removed_any = False

    for target_dir in target_dirs:
        dst = target_dir / PTH_FILENAME
        if dst.exists():
            print(f"Removing {dst}")
            dst.unlink()
            removed_any = True
        else:
            print(f"Not found: {dst}")

    if removed_any:
        print("Done. Restart Python for changes to take effect.")
    else:
        print("No .pth files found to remove.")


def check_pth():
    """Check for .pth files in site-packages and report their status."""
    target_dirs = get_site_packages()
    found_any = False

    print(f"Checking for {PTH_FILENAME} in site-packages directories:")
    for target_dir in target_dirs:
        dst = target_dir / PTH_FILENAME
        if dst.exists():
            print(f"  ✓ Found: {dst}")
            found_any = True
        else:
            print(f"  ✗ Not found: {dst}")

    if found_any:
        print("\nTo remove stale .pth files, run:")
        print(f"  python {Path(__file__).name} --remove")
    else:
        print("\nNo .pth files found.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage kvcached_autopatch.pth file in site-packages"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--remove",
        "--clean",
        action="store_true",
        help="Remove .pth file from site-packages (useful after uninstall)",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Check for .pth files in site-packages",
    )

    args = parser.parse_args()

    if args.remove:
        remove_pth()
    elif args.check:
        check_pth()
    else:
        copy_pth()


if __name__ == "__main__":
    main()
