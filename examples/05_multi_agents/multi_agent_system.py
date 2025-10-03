#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Agent System using direct API calls to local vLLM/SGLang models

This system creates a collaborative conversation between two specialized agents:
1. Research Agent - Analyzes topics and provides detailed information
2. Writing Agent - Creates clear, structured summaries

Usage: python3 multi_agent_system.py "your topic here"
"""

import asyncio
import json
import sys
import time
from typing import AsyncGenerator, Dict, Optional

import aiohttp
import requests


class Agent:
    """Simple agent that calls a local model via completions API."""

    def __init__(self, name: str, port: int, model_name: str, system_prompt: str):
        self.name = name
        self.port = port
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.base_url = f"http://127.0.0.1:{port}/v1"

    async def generate_response_streaming(self, prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> AsyncGenerator[str, None]:
        """Generate a streaming response using the completions endpoint."""
        try:
            # Combine system prompt with user prompt
            full_prompt = f"{self.system_prompt}\n\nTopic: {prompt}\n\nResponse:"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.9,
                        "stop": ["\nTopic:", "\nUser:"],
                        "stream": True
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:

                    if response.status == 200:
                        full_text = ""
                        async for line in response.content:
                            line_text = line.decode('utf-8').strip()
                            if line_text.startswith('data: '):
                                data_str = line_text[6:]  # Remove 'data: ' prefix
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str) if data_str else {}
                                    if data.get("choices") and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("text", "")
                                        if delta:
                                            full_text += delta
                                            yield delta
                                except json.JSONDecodeError:
                                    continue
                                except Exception:
                                    continue
                    else:
                        error_text = await response.text()
                        print(f"Error from {self.name}: {response.status} - {error_text}")

        except Exception as e:
            print(f"Exception in {self.name}: {e}")

    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> Optional[str]:
        """Generate a non-streaming response (fallback method)."""
        try:
            # Combine system prompt with user prompt
            full_prompt = f"{self.system_prompt}\n\nTopic: {prompt}\n\nResponse:"

            response = requests.post(
                f"{self.base_url}/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["\nTopic:", "\nUser:"]
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("choices") and len(data["choices"]) > 0:
                    return data["choices"][0]["text"].strip()
            else:
                print(f"Error from {self.name}: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Exception in {self.name}: {e}")

        return None


class MultiAgentSystem:
    """Simple multi-agent system with two specialized agents."""

    def __init__(self, research_port: int = 12346, writing_port: int = 12347):
        self.research_port = research_port
        self.writing_port = writing_port

        # Wait for servers and get model names
        self._wait_for_servers()
        research_model = self._get_model_name(research_port)
        writing_model = self._get_model_name(writing_port)

        # Create specialized agents
        self.research_agent = Agent(
            name="ResearchAgent",
            port=research_port,
            model_name=research_model,
            system_prompt="""You are a Research Agent specializing in topic analysis.
Provide detailed information including:
1. Key concepts and definitions
2. Important facts and current information
3. Main benefits or advantages
4. Challenges or limitations
5. Real-world examples or applications

Be thorough but concise in your analysis."""
        )

        self.writing_agent = Agent(
            name="WritingAgent",
            port=writing_port,
            model_name=writing_model,
            system_prompt="""You are a Writing Agent specializing in creating clear summaries.
Based on research information, create a well-structured summary that is:
- Well-organized with clear sections
- Easy to understand for general audiences
- Informative and engaging
- Properly formatted

Focus on clarity and accessibility."""
        )

        print("Multi-Agent System Ready!")
        print(f"Research Agent: {research_model} (port {research_port})")
        print(f"Writing Agent: {writing_model} (port {writing_port})")

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

    async def collaborative_conversation_streaming(self, topic: str) -> Dict[str, str]:
        """Run a collaborative conversation with streaming responses."""
        print(f"Topic: {topic}")

        # Step 1: Research Agent analyzes the topic with streaming
        print("Research Agent analyzing...")
        print("Research Agent Response:")
        research_response = ""
        async for chunk in self.research_agent.generate_response_streaming(topic, temperature=0.7, max_tokens=350):
            print(chunk, end='', flush=True)
            research_response += chunk
        print("\n")

        if not research_response:
            print("Error: Research Agent failed to respond")
            return {"error": "Research Agent failed"}

        # Step 2: Writing Agent creates summary based on research with streaming
        print("Writing Agent creating summary...")
        writing_prompt = f"""Based on the research analysis below, create a comprehensive summary about {topic}.

Research Analysis:
{research_response}

Create a well-structured summary for general audiences."""

        print("Writing Agent Response:")
        writing_response = ""
        async for chunk in self.writing_agent.generate_response_streaming(writing_prompt, temperature=0.5, max_tokens=350):
            print(chunk, end='', flush=True)
            writing_response += chunk
        print("\n")

        if not writing_response:
            print("Error: Writing Agent failed to respond")
            return {"topic": topic, "research": research_response, "error": "Writing Agent failed"}

        print("Multi-agent collaboration completed")

        return {
            "topic": topic,
            "research": research_response,
            "summary": writing_response
        }

    def collaborative_conversation(self, topic: str) -> Dict[str, str]:
        """Run a collaborative conversation (non-streaming fallback)."""
        print(f"Topic: {topic}")

        # Step 1: Research Agent analyzes the topic
        print("Research Agent analyzing...")
        research_response = self.research_agent.generate_response(topic, temperature=0.7, max_tokens=350)

        if not research_response:
            print("Error: Research Agent failed to respond")
            return {"error": "Research Agent failed"}

        print("Research Agent Response:")
        print(f"{research_response}\n")

        # Step 2: Writing Agent creates summary based on research
        print("Writing Agent creating summary...")
        writing_prompt = f"""Based on the research analysis below, create a comprehensive summary about {topic}.

Research Analysis:
{research_response}

Create a well-structured summary for general audiences."""

        writing_response = self.writing_agent.generate_response(writing_prompt, temperature=0.5, max_tokens=350)

        if not writing_response:
            print("Error: Writing Agent failed to respond")
            return {"topic": topic, "research": research_response, "error": "Writing Agent failed"}

        print("Writing Agent Response:")
        print(f"{writing_response}\n")

        print("Multi-agent collaboration completed")

        return {
            "topic": topic,
            "research": research_response,
            "summary": writing_response
        }


async def run_streaming_conversation(system: MultiAgentSystem, topic: str):
    """Run a streaming conversation."""
    try:
        result = await system.collaborative_conversation_streaming(topic)
        return result
    except Exception as e:
        print(f"Streaming failed, falling back to non-streaming: {e}")
        return system.collaborative_conversation(topic)

def main():
    """Main function for the multi-agent system."""
    import argparse

    parser = argparse.ArgumentParser(description="A Multi-Agent Research and Writing System")
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
        # Initialize system
        system = MultiAgentSystem(
            research_port=args.research_port,
            writing_port=args.writing_port
        )

        if args.topic:
            # Collaborative conversation on specific topic
            if args.streaming:
                asyncio.run(run_streaming_conversation(system, args.topic))
            else:
                system.collaborative_conversation(args.topic)
        else:
            examples = [
                "artificial intelligence and machine learning",
                "renewable energy technologies",
                "quantum computing basics"
            ]

            print("Running Multi-Agent Examples")
            for i, example_topic in enumerate(examples, 1):
                print(f"\nExample {i}/{len(examples)}")
                if args.streaming:
                    asyncio.run(run_streaming_conversation(system, example_topic))
                else:
                    system.collaborative_conversation(example_topic)

                if i < len(examples):
                    input("\nPress Enter to continue to next example...")

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()