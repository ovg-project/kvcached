# Test execution boundaries

Every top-level `tests/test_*.py` module must appear in exactly one manifest.
`tools/check_test_classification.py` enforces complete, exclusive coverage and
fails CI when a new test has not been classified.

- `cpu.txt`: dependency-isolated tests that run on a hosted CPU runner without
  PyTorch, a serving engine, the compiled VMM extension, or shared services.
- `gpu.txt`: tests whose primary boundary is CUDA/HIP hardware and the compiled
  extension.
- `integration.txt`: tests that require a serving-engine or broader project
  runtime, shared memory, multiprocessing, external services, or an end-to-end
  environment. Some integration tests may also require a GPU.

The CPU workflow executes only `cpu.txt`. GPU and integration runners may use
their manifests as those CI environments are added.
