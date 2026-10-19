# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU checks that serve real models with and without kvcached.

Run with ``python3 tests/e2e/run.py --help``. The driver and the client use
only the Python standard library, so they run on a bare GPU host that has
Docker and the NVIDIA container toolkit.
"""
