# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0


def kvweb_main() -> None:
    """Console-script entry point for ``kvweb``.

    Kept out of ``kvweb`` itself: that module imports fastapi and uvicorn at
    module scope, so without the ``web`` extra the script would fail with an
    ImportError traceback instead of saying what to install.
    """
    try:
        from kvcached.cli.kvweb import main
    except ImportError as exc:
        raise SystemExit(
            f"kvweb could not start: {exc}\n"
            "Install the optional dependencies with `pip install kvcached[web]`."
        ) from exc

    main()
