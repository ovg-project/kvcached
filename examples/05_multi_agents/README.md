# Multi-Agent System with kvcached
This example demonstrates how to build a multi-agent system with local vLLM/SGLang models, with kvcached for efficient memory sharing between models.

Specifically, it shows how to build a multi-agent system with two specialized agents:
1. Research Agent - Analyzes topics and provides detailed information
2. Writing Agent - Creates clear, structured summaries
to collaborate on a topic.

The multi-agent system can be extended to include more agents with different specialized tasks.

## Quickstart

### Start model servers

```bash
# Start with default models (recommended for multi-agent demo)
bash start_multi_agent_models.sh \
    --research-model meta-llama/Llama-3.2-3B \
    --writing-model Qwen/Qwen3-4B \
    --research-engine vllm \
    --writing-engine vllm \
    --venv-vllm-path ../../engine_integration/vllm-v0.9.2/.venv
```

### Run multi-agent conversations

In a separate terminal, run the multi-agent client:

```bash
# Run default example topics (total 3 topics)
bash start_multi_agent_client.sh --venv-path .../../engine_integration/vllm-v0.9.2/.venv

# Explore specific topic
bash start_multi_agent_client.sh --topic "your topic here"

# Enable streaming mode for real-time responses
bash start_multi_agent_client.sh --topic "blockchain technology" --streaming

# Use custom ports
bash start_multi_agent_client.sh --research-port 12348 --writing-port 12349
```

You should see collaborative conversations where the Research Agent analyzes topics and the Writing Agent creates comprehensive summaries.
