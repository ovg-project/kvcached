import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import aiohttp

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

    def __init__(self, models_config: Dict[str, Any]):
        """Create a router.

        Parameters
        ----------
        models_config : Optional[Dict[str, Any]]
            In-memory dict mapping model names to endpoint definitions. If
            provided, this takes precedence over *config_path* so no extra
            file is required when the controller already knows endpoint
            details at runtime.
        """

        self.models: Dict[str, ModelConfig] = {}

        # Use a connector with *no* limit to avoid the default 100 concurrent
        # connections cap in aiohttp.
        self._connector = aiohttp.TCPConnector(limit=0)
        self.session: aiohttp.ClientSession = aiohttp.ClientSession(
            connector=self._connector,
            timeout=aiohttp.ClientTimeout(total=3000),  # sensible default
        )

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
        endpoint = self.get_endpoint_for_model(model_name)
        if not endpoint:
            return None

        if self.session is None or self.session.closed:
            raise Exception("Session not initialised")

        try:
            url = f"{endpoint.base_url}{endpoint_path}"

            # Add timeout configuration
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes

            # Check if this is a streaming request
            is_streaming = request_data.get('stream', False)

            if is_streaming:
                # For streaming requests, return the response object directly
                response = await self.session.post(url,
                                                   json=request_data,
                                                   timeout=timeout)
                if response.status == 200:
                    logger.info(
                        f"Successfully routed streaming request for model {model_name} to {endpoint.base_url}"
                    )
                    return response
                else:
                    logger.error(
                        f"Streaming request failed with status {response.status}: {await response.text()}"
                    )
                    await response.release()
                    return None
            else:
                # For non-streaming requests, handle as before
                async with self.session.post(url,
                                             json=request_data,
                                             timeout=timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f"Successfully routed request for model {model_name} to {endpoint.base_url}"
                        )
                        return result
                    else:
                        logger.error(
                            f"Request failed with status {response.status}: {await response.text()}"
                        )
                        return None

        except asyncio.TimeoutError:
            logger.error(
                f"Request timeout for model {model_name} at {endpoint.base_url}"
            )
            return None
        except Exception as e:
            logger.error(f"Error routing request for model {model_name}: {e}")
            return None

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
            raise Exception("Session not initialised")

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
        """Close the HTTP session"""
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
