#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Agent System using proper LangChain APIs with local vLLM/SGLang models

This system creates a collaborative conversation between two specialized agents:
1. Research Agent - Analyzes topics and provides detailed information
2. Writing Agent - Creates clear, structured summaries

Usage: python3 multi_agent_system.py "your topic here"
"""

import asyncio
import sys
import time
from typing import Dict

import requests
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


class LangChainAgent:

    def __init__(self, name: str, port: int, model_name: str, system_prompt: str):
        self.name = name
        self.port = port
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.temperature = 0.7
        self.max_tokens = 512

        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key="EMPTY",
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_kwargs={
                "extra_body": {
                    "chat_template": (
                        "{% for message in messages %}"
                        "{% if message['role'] == 'system' %}{{ message['content'] }}\n\n{% endif %}"
                        "{% if message['role'] == 'user' %}### Input:\n{{ message['content'] }}\n\n{% endif %}"
                        "{% if message['role'] == 'assistant' %}### Response:\n{{ message['content'] }}\n\n{% endif %}"
                        "{% endfor %}"
                        "### Response:\n"
                    )
                }
            }
        )

        # Create AgentExecutor
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_openai_tools_agent(self.llm, [], prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[],
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=1
        )

    def as_runnable(self):
        """Return the agent as a LangChain Runnable for chaining."""
        def process(input_dict):
            user_input = input_dict.get("input", "")
            try:
                result = self.agent_executor.invoke({"input": user_input})
                return result.get("output", "")
            except Exception as e:
                print(f"AgentExecutor error in {self.name}: {e}")
                return ""

        return RunnableLambda(process)


class MultiAgentResearchWriter:
    """Multi-agent collaboration between research and writing agents."""

    def __init__(self, research_port: int = 12346, writing_port: int = 12347):
        self.research_port = research_port
        self.writing_port = writing_port

        self._wait_for_servers()
        research_model = self._get_model_name(research_port)
        writing_model = self._get_model_name(writing_port)

        self.research_agent = LangChainAgent(
            name="ResearchAgent",
            port=research_port,
            model_name=research_model,
            system_prompt="""You are a Research Agent specializing in topic analysis.

Your task is to:
1. Analyze the given topic thoroughly
2. Identify key concepts, definitions, and important facts
3. Explain main benefits, advantages, and challenges
4. Provide real-world examples and applications

Be thorough but concise in your analysis."""
        )

        self.writing_agent = LangChainAgent(
            name="WritingAgent",
            port=writing_port,
            model_name=writing_model,
            system_prompt="""You are a Writing Agent specializing in creating clear, structured summaries.

Your task is to:
1. Take research information and create well-structured summaries
2. Format and organize content effectively
3. Make content accessible to general audiences
4. Ensure proper structure and flow

Focus on clarity, organization, and accessibility."""
        )

        self.chain = self._create_agent_chain()

        print("Multi-Agent Research Writer Ready!")
        print(f"Research Agent: {research_model} (port {research_port})")
        print(f"Writing Agent: {writing_model} (port {writing_port})")

    def _create_agent_chain(self):
        """Create a LangChain chain that connects the research and writing agents."""
        research_runnable = self.research_agent.as_runnable()

        # Create a function to print and format research output for writing agent
        def format_for_writing(research_output):
            research_content = str(research_output).strip()

            print("\nResearch Agent Response:")
            print(research_content if research_content else "[EMPTY RESPONSE]")
            print("\n" + "="*60 + "\n")
            print("Writing Agent creating summary...\n")

            return {
                "input": f"""Based on the research analysis below, create a comprehensive summary.

Research Analysis:
{research_content}

Create a well-structured summary for general audiences."""
            }

        writing_runnable = self.writing_agent.as_runnable()

        # Chain them together: input -> research -> format -> writing
        return research_runnable | RunnableLambda(format_for_writing) | writing_runnable

    def _wait_for_servers(self, timeout: int = 60):
        """Wait for both model servers to be ready."""

        for port in [self.research_port, self.writing_port]:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError(f"Model server on port {port} not ready after {timeout}s")

    def _get_model_name(self, port: int) -> str:
        """Get the actual model name from the server."""
        try:
            response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    return data["data"][0]["id"]
        except Exception:
            pass
        return f"model-{port}"

    async def astream(self, topic: str):
        """Stream the multi-agent conversation using LangChain's astream."""
        print(f"\n{'='*60}")
        print(f"Topic: {topic}")
        print(f"{'='*60}\n")

        print("Research Agent analyzing...")
        async for chunk in self.chain.astream({"input": topic}):
            print(chunk, end='', flush=True)
        print("\n")
        print("Multi-agent research and writing collaboration completed\n")

    def invoke(self, topic: str) -> Dict[str, str]:
        """Run the multi-agent conversation using LangChain's invoke."""
        print(f"\n{'='*60}")
        print(f"Topic: {topic}")
        print(f"{'='*60}")
        print("\nResearch Agent analyzing...\n")

        try:
            result = self.chain.invoke({"input": topic})

            final_response = str(result).strip()

            print("Writing Agent Response:")
            print(final_response)
            print("\n" + "="*60)
            print("Multi-agent research and writing collaboration completed")
            print("="*60 + "\n")

            return {
                "topic": topic,
                "summary": final_response
            }

        except Exception as e:
            print(f"Error in multi-agent collaboration: {e}\n")
            return {
                "topic": topic,
                "error": str(e)
            }


def main():
    """Main function for the multi-agent system."""
    import argparse

    parser = argparse.ArgumentParser(description="A Multi-Agent Research and Writing System with Proper LangChain APIs")
    parser.add_argument("--research-port", type=int, default=12346,
                       help="Research Agent port (default: 12346)")
    parser.add_argument("--writing-port", type=int, default=12347,
                       help="Writing Agent port (default: 12347)")
    parser.add_argument("--topic", type=str,
                       help="Topic for collaboration")
    parser.add_argument("--streaming", action="store_true",
                       help="Enable streaming mode (real-time responses)")

    args = parser.parse_args()

    try:
        system = MultiAgentResearchWriter(
            research_port=args.research_port,
            writing_port=args.writing_port
        )

        if args.topic:
            if args.streaming:
                asyncio.run(system.astream(args.topic))
            else:
                system.invoke(args.topic)
        else:
            examples = [
                "artificial intelligence and machine learning",
                "renewable energy technologies",
                "quantum computing basics"
            ]

            print("\nRunning Multi-Agent Examples with LangChain Chains")
            for i, example_topic in enumerate(examples, 1):
                print(f"\nExample {i}/{len(examples)}")
                if args.streaming:
                    asyncio.run(system.astream(example_topic))
                else:
                    system.invoke(example_topic)

                if i < len(examples):
                    input("\nPress Enter to continue to next example...")

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()