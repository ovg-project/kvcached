#!/usr/bin/env python3
"""
Comprehensive test script for traffic monitoring and sleep management functionality
"""

import argparse
import asyncio
import random
from typing import Any, Dict, List

import aiohttp


class TrafficMonitorTest:
    """Test suite for traffic monitoring functionality"""

    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.test_models = ["meta-llama/Llama-3.2-1B", "Qwen/Qwen3-0.6B"]

    async def test_endpoint(self,
                            session: aiohttp.ClientSession,
                            endpoint: str,
                            method: str = "GET",
                            json_data: Dict = None) -> Dict[str, Any]:
        """Test a single endpoint and return result"""
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                async with session.get(url) as response:
                    status = response.status
                    data = await response.json(
                    ) if response.content_type == 'application/json' else await response.text(
                    )
            elif method.upper() == "POST":
                async with session.post(url, json=json_data) as response:
                    status = response.status
                    data = await response.json(
                    ) if response.content_type == 'application/json' else await response.text(
                    )
            else:
                return {
                    "error": f"Unsupported method: {method}",
                    "status": 400
                }

            return {"status": status, "data": data, "success": status < 400}

        except Exception as e:
            return {"error": str(e), "status": 0, "success": False}

    async def generate_test_traffic(self,
                                    session: aiohttp.ClientSession,
                                    num_requests: int = 5) -> List[Dict]:
        """Generate test traffic to models"""
        print(f"\n📈 Generating {num_requests} test requests...")

        results = []
        test_prompts = [
            "Hello, how are you today?", "What is the capital of France?",
            "Explain quantum physics in simple terms",
            "Write a short poem about coding",
            "What are the benefits of renewable energy?"
        ]

        for i in range(num_requests):
            model = random.choice(self.test_models)
            prompt = random.choice(test_prompts)

            test_request = {
                "model": model,
                "prompt": prompt,
                "max_tokens": random.randint(10, 50),
                "temperature": 0.7
            }

            print(f"  Request {i+1}: {model} - '{prompt[:30]}...'")

            result = await self.test_endpoint(session, "/v1/completions",
                                              "POST", test_request)
            results.append({
                "request_id": i + 1,
                "model": model,
                "prompt": prompt,
                "result": result
            })

            # Add some delay between requests
            await asyncio.sleep(0.5)

        return results

    async def test_traffic_endpoints(self, session: aiohttp.ClientSession):
        """Test all traffic monitoring endpoints"""
        print("\n🔍 Testing Traffic Monitoring Endpoints")
        print("=" * 50)

        endpoints = [
            "/traffic/stats",
            "/traffic/stats?window=30",
            "/traffic/idle",
            "/traffic/idle?threshold=60",
            "/traffic/active",
            "/traffic/active?threshold=60&window=30",
        ]

        # Test specific model endpoints - encode model names for URL
        import urllib.parse
        for model in self.test_models:
            encoded_model = urllib.parse.quote(model, safe='')
            endpoints.append(f"/traffic/stats/model/{encoded_model}")
            endpoints.append(f"/traffic/stats/model/{encoded_model}?window=30")

        results = {}
        for endpoint in endpoints:
            print(f"\n📊 Testing: {endpoint}")
            result = await self.test_endpoint(session, endpoint)
            results[endpoint] = result

            if result["success"]:
                print(f"  ✅ Status: {result['status']}")
                if isinstance(result["data"], dict):
                    # Pretty print key statistics
                    data = result["data"]
                    if "traffic_stats" in data:
                        stats_count = len(data["traffic_stats"])
                        print(f"  📈 Found stats for {stats_count} models")
                    elif "idle_models" in data:
                        idle_count = len(data["idle_models"])
                        print(f"  😴 Found {idle_count} idle models")
                    elif "active_models" in data:
                        active_count = len(data["active_models"])
                        print(f"  ⚡ Found {active_count} active models")
                    elif "model_stats" in data:
                        model_stats = data["model_stats"]
                        print(
                            f"  📊 Model: {model_stats.get('model_name', 'Unknown')}"
                        )
                        print(
                            f"      Requests: {model_stats.get('total_requests', 0)}"
                        )
                        print(
                            f"      Rate: {model_stats.get('request_rate', 0):.2f} req/s"
                        )
            else:
                print(
                    f"  ❌ Status: {result['status']} - {result.get('error', 'Unknown error')}"
                )

        return results

    async def test_sleep_endpoints(self, session: aiohttp.ClientSession):
        """Test all sleep management endpoints"""
        print("\n😴 Testing Sleep Management Endpoints")
        print("=" * 50)

        results = {}

        # Test sleep status
        print("\n🔍 Testing sleep status...")
        result = await self.test_endpoint(session, "/sleep/status")
        results["sleep_status"] = result

        if result["success"]:
            print("  ✅ Sleep status retrieved")
            data = result["data"]
            print(
                f"  💤 Sleeping models: {len(data.get('sleeping_models', {}))}")
            print(
                f"  🎯 Sleep candidates: {len(data.get('sleep_candidates', []))}"
            )
            print(f"  🔄 Auto-sleep: {data.get('auto_sleep_enabled', False)}")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")

        # Test sleep candidates
        print("\n🎯 Testing sleep candidates...")
        result = await self.test_endpoint(session, "/sleep/candidates")
        results["sleep_candidates"] = result

        if result["success"]:
            candidates = result["data"].get("sleep_candidates", [])
            print(f"  ✅ Found {len(candidates)} sleep candidates")
            for candidate in candidates:
                print(f"    - {candidate}")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")

        # Test manual sleep/wake cycle
        test_model = self.test_models[-1]
        print(f"\n💤 Testing manual sleep for model: {test_model}")

        # URL encode the model name
        import urllib.parse
        encoded_test_model = urllib.parse.quote(test_model, safe='')
        print("!!!", encoded_test_model)
        # Put model to sleep
        result = await self.test_endpoint(
            session, f"/sleep/model/{encoded_test_model}", "POST")
        results["sleep_model"] = result

        if result["success"]:
            print("  ✅ Model put to sleep successfully")
        else:
            print(f"  ❌ Sleep failed: {result.get('error', 'Unknown error')}")

        # Check sleep status again
        await asyncio.sleep(1)
        result = await self.test_endpoint(session, "/sleep/status")
        if result["success"]:
            sleeping = result["data"].get("sleeping_models", {})
            if test_model in sleeping or any(test_model in str(k)
                                             for k in sleeping.keys()):
                print("  ✅ Model confirmed sleeping")
            else:
                print("  ⚠️  Model not found in sleeping list")

        # Wake model up
        print(
            f"\n⏰ Testing wake up for model: {test_model} {encoded_test_model}"
        )
        result = await self.test_endpoint(session,
                                          f"/wake/model/{encoded_test_model}",
                                          "POST")
        results["wake_model"] = result
        if result["success"]:
            print("  ✅ Model woken up successfully")
        else:
            print(f"  ❌ Wake failed: {result.get('error', 'Unknown error')}")

        return results

    async def test_integration_scenarios(self, session: aiohttp.ClientSession):
        """Test realistic integration scenarios"""
        print("\n🎯 Testing Integration Scenarios")
        print("=" * 50)

        # Scenario 1: Generate traffic and monitor activity
        print("\n📋 Scenario 1: Traffic Generation → Monitoring")

        # Generate some traffic
        traffic_results = await self.generate_test_traffic(session, 3)
        successful_requests = sum(1 for r in traffic_results
                                  if r["result"]["success"])
        print(
            f"  Generated traffic: {successful_requests}/{len(traffic_results)} successful requests"
        )

        # Wait a moment for statistics to update
        await asyncio.sleep(2)

        # Check updated statistics
        stats_result = await self.test_endpoint(session, "/traffic/stats")
        if stats_result["success"]:
            stats = stats_result["data"].get("traffic_stats", {})
            print(f"  📊 Current stats show {len(stats)} models with activity")

            for model_name, model_stats in stats.items():
                if model_stats.get("total_requests", 0) > 0:
                    print(
                        f"    - {model_name}: {model_stats['total_requests']} requests, "
                        f"{model_stats['request_rate']:.2f} req/s")

        # Scenario 2: Test idle detection after waiting
        print("\n📋 Scenario 2: Idle Detection")
        print("  Waiting 10 seconds to test idle detection...")
        await asyncio.sleep(10)

        idle_result = await self.test_endpoint(session,
                                               "/traffic/idle?threshold=5")
        if idle_result["success"]:
            idle_models = idle_result["data"].get("idle_models", [])
            print(f"  😴 Found {len(idle_models)} models idle for >5 seconds")

        # Scenario 3: Test wake-on-request
        test_model = self.test_models[0]
        print(f"\n📋 Scenario 3: Wake-on-Request for {test_model}")

        # URL encode the model name
        import urllib.parse
        encoded_test_model = urllib.parse.quote(test_model, safe='')

        # Put model to sleep
        await self.test_endpoint(session, f"/sleep/model/{encoded_test_model}",
                                 "POST")
        await asyncio.sleep(2)

        # Send a request to sleeping model
        print("  Sending request to sleeping model...")
        test_request = {
            "model": test_model,
            "prompt": "Wake up test",
            "max_tokens": 10
        }
        request_result = await self.test_endpoint(session, "/v1/completions",
                                                  "POST", test_request)

        if request_result["success"]:
            print("  ✅ Request succeeded - wake-on-request working")
        else:
            print(
                f"  ❌ Request failed: {request_result.get('error', 'Unknown error')}"
            )

    async def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting Traffic Monitor & Sleep Management Tests")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"Test Models: {', '.join(self.test_models)}")
        print("=" * 60)

        async with aiohttp.ClientSession() as session:
            try:
                # Test basic connectivity
                print("\n🔌 Testing server connectivity...")
                health_result = await self.test_endpoint(session, "/health")
                if not health_result["success"]:
                    print(
                        f"❌ Server not responding: {health_result.get('error', 'Unknown error')}"
                    )
                    return
                print("✅ Server is responding")

                # Test models endpoint
                models_result = await self.test_endpoint(session, "/models")
                if models_result["success"]:
                    models_data = models_result["data"]
                    available_models = list(
                        models_data.get("models", {}).keys())
                    print(f"📋 Available models: {', '.join(available_models)}")

                    # Update test models to only use available ones
                    self.test_models = [
                        m for m in self.test_models if m in available_models
                    ]
                    if not self.test_models:
                        print(
                            "⚠️  No test models available, using first available model"
                        )
                        self.test_models = available_models[:1] if available_models else []

                if not self.test_models:
                    print("❌ No models available for testing")
                    return

                # Run test suites
                results = {}

                # Generate initial traffic
                traffic_results = await self.generate_test_traffic(session, 4)
                results["initial_traffic"] = traffic_results

                # Test traffic monitoring
                traffic_test_results = await self.test_traffic_endpoints(
                    session)
                results["traffic_tests"] = traffic_test_results

                # Test sleep management
                sleep_test_results = await self.test_sleep_endpoints(session)
                results["sleep_tests"] = sleep_test_results

                # Test integration scenarios
                await self.test_integration_scenarios(session)

                # Summary
                print("\n📊 Test Summary")
                print("=" * 30)

                total_traffic_tests = len(traffic_test_results)
                successful_traffic_tests = sum(
                    1 for r in traffic_test_results.values() if r["success"])

                total_sleep_tests = len(sleep_test_results)
                successful_sleep_tests = sum(
                    1 for r in sleep_test_results.values() if r["success"])

                print(
                    f"Traffic Tests: {successful_traffic_tests}/{total_traffic_tests} passed"
                )
                print(
                    f"Sleep Tests: {successful_sleep_tests}/{total_sleep_tests} passed"
                )

                if successful_traffic_tests == total_traffic_tests and successful_sleep_tests == total_sleep_tests:
                    print("🎉 All tests passed!")
                else:
                    print("⚠️  Some tests failed - check output above")

            except Exception as e:
                print(f"❌ Test suite failed: {e}")


async def main():
    parser = argparse.ArgumentParser(description='Traffic Monitor Test Suite')
    parser.add_argument(
        '--url',
        default='http://localhost:8081',
        help=
        'Base URL for the controller server (default: http://localhost:8081)')
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip integration scenarios)')

    args = parser.parse_args()

    print("Traffic Monitor & Sleep Management Test Suite")
    print("============================================")
    print("Make sure the controller server is running!")
    print(f"Expected server URL: {args.url}")
    print()
    print("To start the server:")
    print("  cd /workspace/kvcached/controller")
    print("  python frontend.py --config example-config.yaml --port 8081")
    print()

    tester = TrafficMonitorTest(args.url)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
