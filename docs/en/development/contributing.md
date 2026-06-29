# Contributing

We welcome contributions of any kind — code, documentation, tests, or bug reports.

## Development Setup

```bash
git clone https://github.com/ovg-project/kvcached.git
cd kvcached
pip install -e . --no-build-isolation --no-cache-dir
python tools/dev_copy_pth.py
pip install pre-commit
pre-commit install
```

## Code Style

We use pre-commit hooks to ensure consistent style:

- **Ruff**: Python linting and formatting
- **isort**: Import sorting
- **clang-format**: C/C++ formatting
- **mypy**: Type checking (Python 3.9–3.13)
- **codespell**: Spelling checks

Run all checks:

```bash
pre-commit run --all-files
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure `pre-commit run --all-files` passes
5. Submit a pull request

## License

All contributions are licensed under Apache 2.0. New files must include the SPDX license header:

```python
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
```
