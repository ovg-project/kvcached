import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import aiohttp

logger = logging.getLogger(__name__)


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

    def __init__(self, config_path: Optional[str] = None):
        self.models: Dict[str, ModelConfig] = {}
        self.session: Optional[aiohttp.ClientSession] = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """Load router configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            self.models = {}
            for model_name, model_config in config_data.get("models",
                                                            {}).items():
                endpoint = Endpoint(
                    host=model_config["endpoint"]["host"],
                    port=model_config["endpoint"]["port"],
                    health_check_path=model_config["endpoint"].get(
                        "health_check_path", "/health"))

                self.models[model_name] = ModelConfig(
                    model_name=model_name,
                    endpoint=endpoint,
                )

            logger.info(f"Loaded configuration for {len(self.models)} models")

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
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

        if not self.session:
            self.session = aiohttp.ClientSession()

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

        if not self.session:
            self.session = aiohttp.ClientSession()

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
        if self.session:
            await self.session.close()
            self.session = None

    def list_models(self) -> List[str]:
        """List all configured models"""
        return list(self.models.keys())

    def get_model_endpoint(self, model_name: str) -> Optional[str]:
        """Get the endpoint for a specific model"""
        if model_name not in self.models:
            return None

        return self.models[model_name].endpoint.base_url
