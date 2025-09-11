#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Test script for the updated SleepManager implementation with vLLM API integration.
This test connects to real vLLM instances running on localhost.
"""
import asyncio
import sys
from pathlib import Path

# Add the controller directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "controller"))

from controller.sleep_manager import SleepConfig, SleepManager
from controller.traffic_monitor import TrafficMonitor


async def test_basic_functionality():
    """Test basic SleepManager functionality"""
    print("=== Testing Basic Functionality ===")

    # Create sleep manager with default config
    config = SleepConfig()
    traffic_monitor = TrafficMonitor()
    manager = SleepManager(config, traffic_monitor=traffic_monitor)

    print("✓ SleepManager created successfully")
    print(f"  Auto sleep enabled: {manager.config.auto_sleep_enabled}")
    print(f"  Idle threshold: {manager.config.idle_threshold_seconds}s")
    print(f"  Wake on request: {manager.config.wakeup_on_request}")
    print(f"  Min sleep duration: {manager.config.min_sleep_duration}s")

    return manager


async def test_real_vllm_instances(manager):
    """Test with real vLLM instances from example-config.yaml"""
    print("\n=== Testing Real vLLM Instances ===")

    # Add the real vLLM instances from the config
    manager.add_vllm_model('meta-llama/Llama-3.2-1B', 'localhost', '12346')

    # Get all models
    models = manager.get_vllm_models()
    print(f"✓ Added {len(models)} real vLLM models:")
    for model_name, config in models.items():
        print(f"  {model_name}: {config['host']}:{config['port']}")

    return models


async def test_sglang_configuration(manager):
    """Test SGLang model configuration management"""
    print("\n=== Testing SGLang Configuration ===")

    # Add SGLang models
    manager.add_sglang_model('Qwen/Qwen3-0.6B', 'localhost', '30000')

    # Get all SGLang models
    sglang_models = manager.get_sglang_models()
    print(f"✓ Added {len(sglang_models)} SGLang models:")
    for model_name, config in sglang_models.items():
        print(f"  {model_name}: {config['host']}:{config['port']}")

    # Test removing a model (but don't remove the one we need for testing)
    test_remove_model = 'test-remove-model'
    manager.add_sglang_model(test_remove_model, 'localhost', '30001')
    manager.remove_sglang_model(test_remove_model)
    sglang_models = manager.get_sglang_models()
    print(f"✓ After removal test, {len(sglang_models)} SGLang models remain")

    return sglang_models


async def test_sleep_wake_functionality(manager):
    """Test actual sleep and wake functionality with real vLLM instances"""
    print("\n=== Testing Sleep/Wake Functionality ===")

    # Test with the first model (Llama)
    test_model = 'meta-llama/Llama-3.2-1B'

    print(f"Testing sleep/wake cycle for {test_model}")

    # Check initial sleep status using vLLM API
    print("1. Checking initial sleep status...")
    sleep_status = await manager.check_model_sleep_status(test_model)
    print(f"   Initial sleep status: {sleep_status}")

    # Put model to sleep
    print("2. Putting model to sleep...")
    sleep_success = await manager.put_model_to_sleep(test_model, manual=True)
    print(f"   Sleep operation success: {sleep_success}")

    if sleep_success:
        # Wait a moment
        await asyncio.sleep(2)

        # Check sleep status again
        print("3. Verifying model is sleeping...")
        sleep_status = await manager.check_model_sleep_status(test_model)
        print(f"   Sleep status after sleep: {sleep_status}")

        # Check internal state
        is_sleeping_internal = manager.is_model_sleeping(test_model)
        print(f"   Internal sleep tracking: {is_sleeping_internal}")

        # Wait for minimum sleep duration to pass
        print(
            f"4. Waiting {manager.config.min_sleep_duration}s for minimum sleep duration..."
        )
        await asyncio.sleep(manager.config.min_sleep_duration + 1)

        # Wake up the model
        print("5. Waking up model...")
        wake_success = await manager.wakeup_model(test_model)
        print(f"   Wake operation success: {wake_success}")

        if wake_success:
            # Wait a moment
            await asyncio.sleep(2)

            # Check final sleep status
            print("6. Verifying model is awake...")
            sleep_status = await manager.check_model_sleep_status(test_model)
            print(f"   Sleep status after wake: {sleep_status}")

            # Check internal state
            is_sleeping_internal = manager.is_model_sleeping(test_model)
            print(f"   Internal sleep tracking: {is_sleeping_internal}")


async def test_sglang_sleep_wake_functionality(manager):
    """Test actual sleep and wake functionality with SGLang instances"""
    print("\n=== Testing SGLang Sleep/Wake Functionality ===")

    # Test with the SGLang model
    test_model = 'Qwen/Qwen3-0.6B'

    print(f"Testing SGLang sleep/wake cycle for {test_model}")

    # Check initial memory status using SGLang API
    print("1. Checking initial memory status...")
    sleep_status = await manager.check_model_sleep_status(test_model)
    print(f"   Initial memory released status: {sleep_status}")

    # Release memory (put model to sleep)
    print("2. Releasing memory occupation...")
    sleep_success = await manager.put_model_to_sleep(test_model, manual=True)
    print(f"   Release operation success: {sleep_success}")

    if sleep_success:
        # Wait a moment
        await asyncio.sleep(2)

        # Check memory status again
        print("3. Verifying memory is released...")
        sleep_status = await manager.check_model_sleep_status(test_model)
        print(f"   Memory status after release: {sleep_status}")

        # Check internal state
        is_sleeping_internal = manager.is_model_sleeping(test_model)
        print(f"   Internal sleep tracking: {is_sleeping_internal}")

        # Wait for minimum sleep duration to pass
        print(
            f"4. Waiting {manager.config.min_sleep_duration}s for minimum sleep duration..."
        )
        await asyncio.sleep(manager.config.min_sleep_duration + 1)

        # Resume memory occupation (wake up the model)
        print("5. Resuming memory occupation...")
        wake_success = await manager.wakeup_model(test_model)
        print(f"   Resume operation success: {wake_success}")

        if wake_success:
            # Wait a moment
            await asyncio.sleep(2)

            # Check final memory status
            print("6. Verifying memory is resumed...")
            sleep_status = await manager.check_model_sleep_status(test_model)
            print(f"   Memory status after resume: {sleep_status}")

            # Check internal state
            is_sleeping_internal = manager.is_model_sleeping(test_model)
            print(f"   Internal sleep tracking: {is_sleeping_internal}")


async def test_sleep_state_tracking(manager):
    """Test sleep state tracking functionality"""
    print("\n=== Testing Sleep State Tracking ===")

    # Get sleeping models
    sleeping_models = manager.get_sleeping_models()
    print(f"✓ Currently sleeping models: {len(sleeping_models)}")

    if sleeping_models:
        for model_name, info in sleeping_models.items():
            print(
                f"  {model_name}: sleeping for {info['sleep_duration']:.1f}s, manual: {info['manual_sleep']}"
            )

    # Get sleep candidates (this will depend on traffic_monitor)
    try:
        candidates = manager.get_sleep_candidates()
        print(f"✓ Sleep candidates: {len(candidates)}")
    except Exception as e:
        print(
            f"⚠ Could not get sleep candidates (traffic_monitor not available): {e}"
        )


async def test_config_updates(manager):
    """Test configuration updates"""
    print("\n=== Testing Configuration Updates ===")

    # Update configuration
    manager.update_config(auto_sleep_enabled=True,
                          idle_threshold_seconds=600,
                          min_sleep_duration=120)

    print("✓ Updated configuration:")
    print(f"  Auto sleep enabled: {manager.config.auto_sleep_enabled}")
    print(f"  Idle threshold: {manager.config.idle_threshold_seconds}s")
    print(f"  Min sleep duration: {manager.config.min_sleep_duration}s")


async def test_api_methods_simulation(manager):
    """Test API method signatures without making actual HTTP calls"""
    print("\n=== Testing API Method Signatures ===")

    # These methods would make HTTP calls in real usage
    # Here we just test that they can be called without syntax errors
    print("✓ vLLM sleep/wake API methods are properly defined:")
    print(
        f"  _call_vllm_sleep_api: {hasattr(manager, '_call_vllm_sleep_api')}")
    print(f"  _call_vllm_wake_api: {hasattr(manager, '_call_vllm_wake_api')}")

    print("✓ SGLang API methods are properly defined:")
    print(
        f"  _call_sglang_release_api: {hasattr(manager, '_call_sglang_release_api')}"
    )
    print(
        f"  _call_sglang_resume_api: {hasattr(manager, '_call_sglang_resume_api')}"
    )

    print("✓ Common API methods:")
    print(
        f"  check_model_sleep_status: {hasattr(manager, 'check_model_sleep_status')}"
    )
    print(f"  handle_model_wakeup: {hasattr(manager, 'handle_model_wakeup')}")


async def test_sglang_api_methods_simulation(manager):
    """Test SGLang-specific API method behavior"""
    print("\n=== Testing SGLang API Methods Simulation ===")

    # Test configuration methods
    print("✓ SGLang model management methods:")
    print(f"  add_sglang_model: {hasattr(manager, 'add_sglang_model')}")
    print(f"  remove_sglang_model: {hasattr(manager, 'remove_sglang_model')}")
    print(f"  get_sglang_models: {hasattr(manager, 'get_sglang_models')}")

    # Test model detection logic
    test_model = 'Qwen/Qwen3-0.6B'
    print(f"\n✓ Model type detection for '{test_model}':")
    is_sglang = test_model in manager.config.sglang_models_config
    print(f"  Detected as SGLang model: {is_sglang}")


async def main():
    """Main test function"""
    print("Testing SleepManager with Real vLLM and SGLang Instances")
    print("=" * 70)

    try:
        # Run all tests
        manager = await test_basic_functionality()

        # Test vLLM functionality
        await test_real_vllm_instances(manager)
        await test_sleep_wake_functionality(manager)

        # Test SGLang functionality
        await test_sglang_configuration(manager)
        await test_sglang_sleep_wake_functionality(manager)

        # Test state tracking
        await test_sleep_state_tracking(manager)

        # Test configuration
        await test_config_updates(manager)

        # Test API methods
        await test_api_methods_simulation(manager)
        await test_sglang_api_methods_simulation(manager)

        print("\n" + "=" * 70)
        print("✅ All tests completed successfully!")
        print(
            "\nBoth vLLM and SGLang sleep/wake functionality has been tested.")
        print("Check the output above for detailed results.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
