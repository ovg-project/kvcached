import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import aiohttp
from traffic_monitor import TrafficMonitor

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


@dataclass
class SleepConfig:
    """Configuration for sleep mode management"""
    idle_threshold_seconds: int = 300  # 5 minutes
    check_interval_seconds: int = 60  # Check every minute
    auto_sleep_enabled: bool = False  # Whether to automatically put models to sleep
    wakeup_on_request: bool = True  # Whether to automatically wake models on request
    min_sleep_duration: int = 60  # Minimum time to keep model asleep (seconds)
    vllm_models_config: Dict[str, Dict[
        str,
        str]] = None  # model_name -> {"host": "localhost", "port": "8000"}
    sglang_models_config: Dict[str, Dict[
        str,
        str]] = None  # model_name -> {"host": "localhost", "port": "8000"}


class SleepManager:
    """Manages sleep mode for idle models to save resources"""

    def __init__(self,
                 config: Optional[SleepConfig] = None,
                 traffic_monitor: Optional[TrafficMonitor] = None):
        self.config = config or SleepConfig()
        self.traffic_monitor = traffic_monitor  # Injected dependency
        self.sleeping_models: Dict[str, float] = {
        }  # model_name -> sleep_start_time
        self.manual_sleep_models: Set[str] = set(
        )  # Models manually put to sleep
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        # Initialize default vLLM models config if not provided
        if self.config.vllm_models_config is None:
            self.config.vllm_models_config = {}
        # Initialize default SGLang models config if not provided
        if self.config.sglang_models_config is None:
            self.config.sglang_models_config = {}

    async def start(self):
        """Start the sleep manager"""
        self._running = True
        if self.config.auto_sleep_enabled:
            self._monitor_task = asyncio.create_task(
                self._monitor_idle_models())
        logger.info(
            f"Sleep manager started (auto_sleep: {self.config.auto_sleep_enabled})"
        )

    async def stop(self):
        """Stop the sleep manager"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Sleep manager stopped")

    async def put_model_to_sleep(self,
                                 model_name: str,
                                 manual: bool = False) -> bool:
        """
        Put a specific model to sleep
        
        Args:
            model_name: Name of the model to put to sleep
            manual: Whether this is a manual sleep request
        
        Returns:
            True if model was put to sleep, False if already sleeping or error
        """
        if model_name in self.sleeping_models:
            logger.info(f"Model {model_name} is already sleeping")
            return False

        try:
            # Use vLLM Python API to put model to sleep
            if model_name in self.config.vllm_models_config:
                model_config = self.config.vllm_models_config[model_name]
                host = model_config.get("host", "localhost")
                port = model_config.get("port", "8000")

                success = await self._call_vllm_sleep_api(host, port, level=1)
                if not success:
                    logger.error(
                        f"Failed to call vLLM sleep API for model {model_name}"
                    )
                    return False
            # Use SGLang Python API to release memory occupation
            elif model_name in self.config.sglang_models_config:
                model_config = self.config.sglang_models_config[model_name]
                host = model_config.get("host", "localhost")
                port = model_config.get("port", "8000")

                success = await self._call_sglang_release_api(host, port)
                if not success:
                    logger.error(
                        f"Failed to call SGLang release API for model {model_name}"
                    )
                    return False
            else:
                logger.warning(
                    f"No vLLM or SGLang configuration found for model {model_name}, using fallback behavior"
                )

            self.sleeping_models[model_name] = time.time()
            if manual:
                self.manual_sleep_models.add(model_name)

            logger.info(f"Put model {model_name} to sleep (manual: {manual})")
            return True

        except Exception as e:
            logger.error(f"Failed to put model {model_name} to sleep: {e}")
            return False

    async def wakeup_model(self, model_name: str) -> bool:
        """
        Wake up a sleeping model
        
        Args:
            model_name: Name of the model to wake up
        
        Returns:
            True if model was woken up, False if not sleeping or error
        """
        if model_name not in self.sleeping_models:
            logger.info(f"Model {model_name} is not sleeping")
            return False

        try:
            # Check minimum sleep duration
            sleep_start_time = self.sleeping_models[model_name]
            sleep_duration = time.time() - sleep_start_time
            logger.info(
                f"Model {model_name} has only been sleeping for {sleep_duration:.1f}s, "
                f"minimum is {self.config.min_sleep_duration}s")
            if sleep_duration < self.config.min_sleep_duration:
                logger.info(
                    f"Model {model_name} has only been sleeping for {sleep_duration:.1f}s, "
                    f"minimum is {self.config.min_sleep_duration}s")
                return False

            # Use vLLM Python API to wake up the model
            if model_name in self.config.vllm_models_config:
                model_config = self.config.vllm_models_config[model_name]
                host = model_config.get("host", "localhost")
                port = model_config.get("port", "8000")

                success = await self._call_vllm_wakeup_api(host, port)
                if not success:
                    logger.error(
                        f"Failed to call vLLM wake API for model {model_name}")
                    return False
            # Use SGLang Python API to resume memory occupation
            elif model_name in self.config.sglang_models_config:
                model_config = self.config.sglang_models_config[model_name]
                host = model_config.get("host", "localhost")
                port = model_config.get("port", "8000")

                success = await self._call_sglang_resume_api(host, port)
                if not success:
                    logger.error(
                        f"Failed to call SGLang resume API for model {model_name}"
                    )
                    return False
            else:
                logger.warning(
                    f"No vLLM or SGLang configuration found for model {model_name}, using fallback behavior"
                )

            del self.sleeping_models[model_name]
            self.manual_sleep_models.discard(model_name)

            logger.info(
                f"Woke up model {model_name} after {sleep_duration:.1f}s of sleep"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to wake up model {model_name}: {e}")
            return False

    def is_model_sleeping(self, model_name: str) -> bool:
        """Check if a model is currently sleeping"""
        return model_name in self.sleeping_models

    def get_sleeping_models(self) -> Dict[str, Dict]:
        """Get information about all sleeping models"""
        current_time = time.time()
        return {
            model_name: {
                'sleep_start_time': sleep_start_time,
                'sleep_duration': current_time - sleep_start_time,
                'manual_sleep': model_name in self.manual_sleep_models
            }
            for model_name, sleep_start_time in self.sleeping_models.items()
        }

    def get_sleep_candidates(self) -> List[str]:
        """Get models that are candidates for sleep mode based on activity"""
        if self.traffic_monitor is None:
            logger.warning(
                "TrafficMonitor not provided; cannot determine sleep candidates"
            )
            return []
        idle_models = self.traffic_monitor.get_idle_models(
            self.config.idle_threshold_seconds)
        # Filter out already sleeping models
        return [
            model for model in idle_models if model not in self.sleeping_models
        ]

    async def _monitor_idle_models(self):
        """Background task to automatically put idle models to sleep"""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval_seconds)

                if not self.config.auto_sleep_enabled:
                    continue

                # Get models that should be put to sleep
                candidates = self.get_sleep_candidates()

                for model_name in candidates:
                    # Don't auto-sleep manually controlled models
                    if model_name not in self.manual_sleep_models:
                        await self.put_model_to_sleep(model_name, manual=False)

                if candidates:
                    logger.info(
                        f"Auto-sleep check: put {len(candidates)} models to sleep: {candidates}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sleep monitor: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def handle_model_wakeup_on_request(self, model_name: str) -> bool:
        """
        Handle wake-up request when a request comes for a sleeping model
        
        Args:
            model_name: Name of the model that needs to be woken up
        
        Returns:
            True if model was woken up or already awake, False if wake failed
        """
        if not self.config.wakeup_on_request:
            return False

        if model_name not in self.sleeping_models:
            return True  # Already awake

        logger.info(
            f"Incoming request for sleeping model {model_name}, attempting to wake up"
        )
        return await self.wakeup_model(model_name)

    def update_config(self, **kwargs):
        """Update sleep manager configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated sleep config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")

    async def _call_vllm_sleep_api(self,
                                   host: str,
                                   port: str,
                                   level: int = 1) -> bool:
        """Call vLLM's sleep API endpoint"""
        url = f"http://{host}:{port}/sleep"
        payload = {"level": str(level)}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        logger.info(
                            f"Successfully called vLLM sleep API at {url} with level {level}"
                        )
                        return True
                    else:
                        logger.error(
                            f"vLLM sleep API returned status {response.status}: {await response.text()}"
                        )
                        return False
        except Exception as e:
            logger.error(f"Error calling vLLM sleep API at {url}: {e}")
            return False

    async def _call_vllm_wakeup_api(self, host: str, port: str) -> bool:
        """Call vLLM's wake_up API endpoint"""
        url = f"http://{host}:{port}/wake_up"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url,
                        timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        logger.info(
                            f"Successfully called vLLM wake_up API at {url}")
                        return True
                    else:
                        logger.error(
                            f"vLLM wake_up API returned status {response.status}: {await response.text()}"
                        )
                        return False
        except Exception as e:
            logger.error(f"Error calling vLLM wake_up API at {url}: {e}")
            return False

    async def check_model_sleep_status(self,
                                       model_name: str) -> Optional[bool]:
        """Check if a model is currently sleeping using vLLM's or SGLang's API
        
        Returns:
            True if sleeping, False if awake, None if unable to determine
        """
        if model_name in self.config.vllm_models_config:
            model_config = self.config.vllm_models_config[model_name]
            host = model_config.get("host", "localhost")
            port = model_config.get("port", "8000")
            url = f"http://{host}:{port}/is_sleeping"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url,
                                           timeout=aiohttp.ClientTimeout(
                                               total=10)) as response:
                        if response.status == 200:
                            result = await response.json()
                            is_sleeping = result.get("is_sleeping", False)
                            logger.debug(
                                f"Model {model_name} sleep status: {is_sleeping}"
                            )
                            return is_sleeping
                        else:
                            logger.error(
                                f"vLLM is_sleeping API returned status {response.status}: {await response.text()}"
                            )
                            return None
            except Exception as e:
                logger.error(
                    f"Error checking model {model_name} sleep status: {e}")
                return None
        elif model_name in self.config.sglang_models_config:
            # For SGLang, we cannot check memory status via API
            logger.warning(
                f"Cannot check memory status for SGLang model {model_name}")
            return None
        else:
            logger.warning(
                f"No vLLM or SGLang configuration found for model {model_name}"
            )
            return None

    def add_vllm_model(self,
                       model_name: str,
                       host: str = "localhost",
                       port: str = "8000"):
        """Add a vLLM model configuration for sleep/wake operations"""
        self.config.vllm_models_config[model_name] = {
            "host": host,
            "port": port
        }
        logger.info(
            f"Added vLLM model configuration: {model_name} at {host}:{port}")

    def remove_vllm_model(self, model_name: str):
        """Remove a vLLM model configuration"""
        if model_name in self.config.vllm_models_config:
            del self.config.vllm_models_config[model_name]
            logger.info(f"Removed vLLM model configuration: {model_name}")
        else:
            logger.warning(
                f"No vLLM configuration found for model {model_name}")

    def get_vllm_models(self) -> Dict[str, Dict[str, str]]:
        """Get all configured vLLM models"""
        return self.config.vllm_models_config.copy()

    async def _call_sglang_release_api(self, host: str, port: str) -> bool:
        """Call SGLang's release_memory_occupation API endpoint"""
        url = f"http://{host}:{port}/release_memory_occupation"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url,
                        timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        logger.info(
                            f"Successfully called SGLang release_memory_occupation API at {url}"
                        )
                        return True
                    else:
                        logger.error(
                            f"SGLang release API returned status {response.status}: {await response.text()}"
                        )
                        return False
        except Exception as e:
            logger.error(f"Error calling SGLang release API at {url}: {e}")
            return False

    async def _call_sglang_resume_api(self, host: str, port: str) -> bool:
        """Call SGLang's resume_memory_occupation API endpoint"""
        url = f"http://{host}:{port}/resume_memory_occupation"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url,
                        timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        logger.info(
                            f"Successfully called SGLang resume_memory_occupation API at {url}"
                        )
                        return True
                    else:
                        logger.error(
                            f"SGLang resume API returned status {response.status}: {await response.text()}"
                        )
                        return False
        except Exception as e:
            logger.error(f"Error calling SGLang resume API at {url}: {e}")
            return False

    def add_sglang_model(self,
                         model_name: str,
                         host: str = "localhost",
                         port: str = "8000"):
        """Add a SGLang model configuration for sleep/wake operations"""
        self.config.sglang_models_config[model_name] = {
            "host": host,
            "port": port
        }
        logger.info(
            f"Added SGLang model configuration: {model_name} at {host}:{port}")

    def remove_sglang_model(self, model_name: str):
        """Remove a SGLang model configuration"""
        if model_name in self.config.sglang_models_config:
            del self.config.sglang_models_config[model_name]
            logger.info(f"Removed SGLang model configuration: {model_name}")
        else:
            logger.warning(
                f"No SGLang configuration found for model {model_name}")

    def get_sglang_models(self) -> Dict[str, Dict[str, str]]:
        """Get all configured SGLang models"""
        return self.config.sglang_models_config.copy()
