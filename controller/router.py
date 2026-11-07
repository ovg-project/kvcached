# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import aiohttp
from sleep_manager import SleepManager
from traffic_monitor import RequestStats, TrafficMonitor

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


@dataclass
class Endpoint:
    host: str
    port: int
    health_check_path: str = "/health"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ModelConfig:
    model_name: str
    endpoint: Endpoint

    def __post_init__(self):
        if not self.endpoint:
            raise ValueError(
                f"No endpoint configured for model {self.model_name}")


class LLMRouter:

    def __init__(self, models_config: Dict[str, Any], *,
                 sleep_manager: SleepManager, traffic_monitor: TrafficMonitor):
        """Create a router.

        Args:
            models_config: Mapping of model name to endpoint configuration.
            sleep_manager: SleepManager instance to manage model sleep/wake.
            traffic_monitor: TrafficMonitor instance for request statistics.
        """
        self.sleep_manager = sleep_manager
        self.traffic_monitor = traffic_monitor

        self.models: Dict[str, ModelConfig] = {}

        # Use a connector with *no* limit to avoid the default 100 concurrent
        # connections cap in aiohttp.
        self._connector = aiohttp.TCPConnector(limit=0)
        self.session: aiohttp.ClientSession = aiohttp.ClientSession(
            connector=self._connector,
            timeout=aiohttp.ClientTimeout(total=3000),  # sensible default
        )

        self._streams: Dict[aiohttp.ClientResponse, RequestStats] = {}
        self._drains: Dict[asyncio.Task, Any] = {}
        self.load_config_from_dict(models_config)

    def load_config_from_dict(self, config_data: Dict[str, Any]):
        """Load router configuration from an in-memory dictionary.

        This supports two shapes:

        1. The original JSON shape::

               {"models": {
                   "m1": {"endpoint": {"host": "h", "port": 123}},
                   ...
               }}

        2. A flattened variant::

               {"m1": {"host": "h", "port": 123}, ...}
        """

        try:
            self.models = {}

            # Accept either top-level "models" or direct mapping
            models_section = config_data.get("models", config_data)

            for model_name, model_cfg in models_section.items():
                # Support flattened or nested "endpoint" key
                if "endpoint" in model_cfg:
                    ep_cfg = model_cfg["endpoint"]
                else:
                    ep_cfg = model_cfg

                endpoint = Endpoint(
                    host=ep_cfg["host"],
                    port=int(ep_cfg["port"]),
                    health_check_path=ep_cfg.get("health_check_path",
                                                 "/health"),
                )

                self.models[model_name] = ModelConfig(model_name, endpoint)

            logger.info("Loaded configuration for %s models from dict",
                        len(self.models))
        except Exception as e:
            logger.error("Error loading configuration from dict: %s", e)
            raise

    def add_model(self, model_config: ModelConfig):
        """Add a model configuration programmatically"""
        self.models[model_config.model_name] = model_config
        logger.info(f"Added model configuration for {model_config.model_name}")

    def get_endpoint_for_model(self, model_name: str) -> Optional[Endpoint]:
        """Get the endpoint for a given model"""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found in configuration")
            return None

        model_config = self.models[model_name]

        return model_config.endpoint

    async def route_request(
        self,
        model_name: str,
        request_data: Dict[str, Any],
        endpoint_path: str = "/v1/completions"
    ) -> Optional[Union[Dict[str, Any], aiohttp.ClientResponse]]:
        """Route a request to the appropriate endpoint"""
        request_stats = None
        response = None
        handed_off = False
        backend_drained = True
        success = False
        error = None
        try:
            # Sleep must not slip between the awake check and registering work.
            async with self.sleep_manager.request_admission(model_name) as allowed:
                request_stats = self.traffic_monitor.record_request_start(
                    model_name, endpoint_path)
                if not allowed:
                    error = f"Failed to wake up sleeping model {model_name}"
                    return None

            endpoint = self.get_endpoint_for_model(model_name)
            if endpoint is None:
                error = f"Model {model_name} not found"
                return None
            if self.session is None or self.session.closed:
                error = "Session not initialized"
                return None

            url = f"{endpoint.base_url}{endpoint_path}"
            timeout = aiohttp.ClientTimeout(total=300)
            backend_drained = False
            response = await self.session.post(url, json=request_data, timeout=timeout)
            if response.status != 200:
                error = f"HTTP {response.status}: {await response.text()}"
                return None
            if request_data.get('stream', False):
                self._streams[response] = request_stats
                handed_off = True
                return response
            result = await response.json()
            success = True
            return result
        except asyncio.CancelledError:
            error = "Request cancelled before response completion"
            raise
        except Exception as exc:
            # Failure to establish a connection cannot leave engine work behind.
            if isinstance(exc, aiohttp.ClientConnectorError):
                backend_drained = True
            error = str(exc)
            logger.error("Error routing request for %s: %s", model_name, exc)
            return None
        finally:
            if request_stats is not None and not handed_off:
                if response is not None:
                    backend_drained = response.content.at_eof()
                    response.release()
                self.traffic_monitor.record_request_end(
                    request_stats, success=success, error_message=error,
                    backend_drained=backend_drained)

    def finish_stream(self, response: aiohttp.ClientResponse, *, success: bool):
        """Finish a relay, retaining ownership while an upstream body drains."""
        request_stats = self._streams.pop(response, None)
        if request_stats is None:
            return
        if response.content.at_eof():
            response.release()
            self.traffic_monitor.record_request_end(request_stats, success=success)
            return

        # A disconnected client need not have cancelled work in the engine.
        # Discard the remaining body under the existing upstream request timeout.
        task = asyncio.create_task(self._drain_stream(response))
        self._drains[task] = (response, request_stats)
        task.add_done_callback(self._finish_drain)

    async def _drain_stream(self, response: aiohttp.ClientResponse) -> bool:
        try:
            async for _ in response.content.iter_chunked(65536):
                pass
            return True
        except Exception as exc:
            logger.warning("Cannot confirm backend stream completion: %s", exc)
            return False

    def _finish_drain(self, task: asyncio.Task):
        response, request_stats = self._drains.pop(task)
        drained = False if task.cancelled() else task.result()
        response.release()
        self.traffic_monitor.record_request_end(
            request_stats, success=False, error_message="Stream relay interrupted",
            backend_drained=drained)

    async def health_check(self, model_name: str) -> Dict[str, bool]:
        """Check health of all endpoints for a model"""
        if model_name == "all":
            health_status = {}
            for model_name in self.models:
                health_status[model_name] = await self.health_check(model_name)
            return health_status

        if model_name not in self.models:
            return {}

        model_config = self.models[model_name]
        health_status = {}

        if self.session is None or self.session.closed:
            raise Exception("Session not initialized")

        endpoint = model_config.endpoint
        try:
            url = f"{endpoint.base_url}{endpoint.health_check_path}"
            timeout = aiohttp.ClientTimeout(total=5)

            async with self.session.get(url, timeout=timeout) as response:
                health_status[endpoint.base_url] = response.status == 200

        except Exception as e:
            logger.warning(f"Health check failed for {endpoint.base_url}: {e}")
            health_status[endpoint.base_url] = False

        return health_status

    async def close(self):
        """Close the HTTP session and finalize every retained stream owner."""
        for response in list(self._streams):
            self.finish_stream(response, success=False)
        drains = list(self._drains)
        for task in drains:
            task.cancel()
        if drains:
            await asyncio.gather(*drains, return_exceptions=True)
        if self.session and not self.session.closed:
            await self.session.close()
        if not self._connector.closed:
            await self._connector.close()

    def list_models(self) -> List[str]:
        """List all configured models"""
        return list(self.models.keys())

    def get_model_endpoint(self, model_name: str) -> Optional[str]:
        """Get the endpoint for a specific model"""
        if model_name not in self.models:
            return None

        return self.models[model_name].endpoint.base_url
