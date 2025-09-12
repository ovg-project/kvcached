# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import io
import re
import sys


def fix_file(path: str) -> bool:
    try:
        with io.open(path, "r", encoding="utf-8", newline="") as f:
            content = f.read()
    except Exception:
        return False

    # Remove exactly one blank line after the first two comment lines at top
    # that contain SPDX lines.
    pattern = r"^((?:\s*[#/]{1,2}.*\n){2})\s*\n"
    new_content = re.sub(pattern, r"\1", content, count=1, flags=re.M)

    if new_content != content:
        try:
            with io.open(path, "w", encoding="utf-8", newline="") as f:
                f.write(new_content)
            return True
        except Exception:
            return False
    return False

def main() -> int:
    changed = False
    for p in sys.argv[1:]:
        if fix_file(p):
            changed = True
    # Return 1 to indicate files were modified so pre-commit stops and asks to re-run
    return 1 if changed else 0

if __name__ == "__main__":
    sys.exit(main())


