#!/usr/bin/env python3
"""
Simple test script for the traffic monitoring functionality
"""

import asyncio
import json
import time
import aiohttp
from pathlib import Path


async def test_traffic_monitoring():
    """Test the traffic monitoring endpoints"""
    
    base_url = "http://localhost:8081"
    
    # Test endpoints to check
    endpoints = [
        "/traffic/stats",
        "/traffic/stats/meta-llama/Llama-3.2-1B",
        "/traffic/idle",
        "/traffic/active",
        "/sleep/status",
        "/sleep/candidates"
    ]
    
    async with aiohttp.ClientSession() as session:
        print("Testing traffic monitoring endpoints...")
        print("=" * 50)
        
        for endpoint in endpoints:
            try:
                async with session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✓ {endpoint}")
                        print(f"  Response: {json.dumps(data, indent=2)}")
                    else:
                        print(f"✗ {endpoint} - Status: {response.status}")
                        print(f"  Error: {await response.text()}")
            except Exception as e:
                print(f"✗ {endpoint} - Exception: {e}")
            print()
        
        # Test a completion request to generate traffic
        print("Testing completion request to generate traffic...")
        test_request = {
            "model": "meta-llama/Llama-3.2-1B",
            "prompt": "Hello, how are you?",
            "max_tokens": 10
        }
        
        try:
            async with session.post(f"{base_url}/v1/completions", json=test_request) as response:
                print(f"Completion request status: {response.status}")
                if response.status != 200:
                    print(f"Response: {await response.text()}")
        except Exception as e:
            print(f"Completion request error: {e}")
        
        print("\nWaiting 2 seconds then checking traffic stats again...")
        await asyncio.sleep(2)
        
        # Check traffic stats after the request
        try:
            async with session.get(f"{base_url}/traffic/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    print("Updated traffic stats:")
                    print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error checking updated stats: {e}")


if __name__ == "__main__":
    print("Traffic Monitor Test Script")
    print("Make sure the controller server is running on port 8081")
    print("You can start it with: python frontend.py --config example-config.yaml --port 8081")
    print()
    
    asyncio.run(test_traffic_monitoring())
    
    print("\nTest complete!")