# Multi-Agent Collaboration with LangChain and kvcached

This example demonstrates how to build a multi-agent system using LangChain APIs with local vLLM/SGLang models, with kvcached for efficient memory sharing between models.

Specifically, it shows how to build a multi-agent system with two specialized agents:
1. Research Agent - Analyzes topics and provides detailed information
2. Writing Agent - Creates clear, structured summaries
to collaborate on a topic.

The multi-agent collaboration system uses LangChain for agent orchestration and can be extended to include more agents with different specialized tasks.

## Quickstart

### 1. Setup LangChain Environment

First, set up the LangChain environment:

```bash
bash setup_langchain.sh
```

### 2. Start Model Servers

Start the model servers (these can use the existing vLLM or SGLang environments):

```bash
# Start with default models (recommended for multi-agent demo)
bash start_multi_agent_models.sh \
    --research-model meta-llama/Llama-3.2-3B \
    --writing-model Qwen/Qwen3-4B \
    --research-engine vllm \
    --writing-engine vllm \
    --venv-vllm-path ${VENV_PATH}
```

### 3. Run Multi-Agent Examples with LangChain

In a separate terminal, run the multi-agent client with LangChain:

```bash
cd examples/05_multi_agents
source langchain-venv/bin/activate
```

```bash
# Run default example topics (total 3 topics)
python3 multi_agent_example.py

# Explore specific topic
python3 multi_agent_example.py --topic "your topic here"

# Enable streaming mode for real-time responses
python3 multi_agent_example.py --topic "blockchain technology" --streaming

# Use custom ports
python3 multi_agent_example.py --research-port 12348 --writing-port 12349
```

You should see collaborative conversations where the Research Agent analyzes topics and the Writing Agent creates comprehensive summaries.
