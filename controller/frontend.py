import asyncio
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from aiohttp import web
from router import LLMRouter
from utils import set_ulimit

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


class MultiLLMFrontend:

    def __init__(self, port: int, model_config_json: Dict[str, Any]):
        self.router = LLMRouter(models_config=model_config_json)
        self.port = port
        self.app = web.Application()
        self.configure_endpoints()
        set_ulimit()

    def configure_endpoints(self):
        """Configure HTTP endpoints"""
        self.app.router.add_post('/v1/completions', self.handle_completion)
        self.app.router.add_post('/v1/chat/completions',
                                 self.handle_chat_completion)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/models', self.handle_list_models)
        self.app.router.add_get('/health/{model_name}',
                                self.handle_model_health)
        self.app.router.add_get('/get_server_info',
                                self.handle_get_server_info)

    async def handle_completion(self, request: web.Request) -> web.Response:
        """Handle completion requests"""
        try:
            request_data = await request.json()
            model_name = request_data.get('model')

            if not model_name:
                return web.Response(text=json.dumps(
                    {"error": "model parameter is required"}),
                                    status=400,
                                    content_type='application/json')

            # Route the request
            result = await self.router.route_request(model_name, request_data,
                                                     "/v1/completions")

            if result is None:
                return web.Response(text=json.dumps({
                    "error":
                    f"Failed to route request for model {model_name}"
                }),
                                    status=500,
                                    content_type='application/json')

            # Check if this is a streaming response
            if hasattr(result, 'content') and hasattr(result, 'headers'):
                # This is a streaming response (aiohttp.ClientResponse)
                response = web.StreamResponse(status=result.status,
                                              headers={
                                                  'Content-Type':
                                                  result.headers.get(
                                                      'Content-Type',
                                                      'text/event-stream'),
                                                  'Cache-Control':
                                                  'no-cache',
                                                  'Connection':
                                                  'keep-alive'
                                              })
                await response.prepare(request)

                try:
                    async for chunk in result.content.iter_chunked(1024):
                        await response.write(chunk)

                    # Explicitly close the stream to make sure the
                    # connection/FIFO is released promptly.
                    await response.write_eof()

                    return response
                finally:
                    await result.release()
            else:
                # This is a regular JSON response
                return web.Response(text=json.dumps(result),
                                    status=200,
                                    content_type='application/json')

        except Exception as e:
            logger.error(f"Error handling completion request: {str(e)}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_chat_completion(self,
                                     request: web.Request) -> web.Response:
        """Handle chat completion requests"""
        try:
            request_data = await request.json()
            model_name = request_data.get('model')

            if not model_name:
                return web.Response(text=json.dumps(
                    {"error": "model parameter is required"}),
                                    status=400,
                                    content_type='application/json')

            # Route the request
            result = await self.router.route_request(model_name, request_data,
                                                     "/v1/chat/completions")

            if result is None:
                return web.Response(text=json.dumps({
                    "error":
                    f"Failed to route request for model {model_name}"
                }),
                                    status=500,
                                    content_type='application/json')

            # Check if this is a streaming response
            if hasattr(result, 'content') and hasattr(result, 'headers'):
                # This is a streaming response (aiohttp.ClientResponse)
                response = web.StreamResponse(status=result.status,
                                              headers={
                                                  'Content-Type':
                                                  result.headers.get(
                                                      'Content-Type',
                                                      'text/event-stream'),
                                                  'Cache-Control':
                                                  'no-cache',
                                                  'Connection':
                                                  'keep-alive'
                                              })
                await response.prepare(request)

                try:
                    async for chunk in result.content.iter_chunked(1024):
                        await response.write(chunk)

                    # Ensure the downstream connection is properly closed.
                    await response.write_eof()

                    return response
                finally:
                    await result.release()
            else:
                # This is a regular JSON response
                return web.Response(text=json.dumps(result),
                                    status=200,
                                    content_type='application/json')

        except Exception as e:
            logger.error(f"Error handling chat completion request: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests"""
        return web.Response(text=json.dumps({"status": "healthy"}),
                            status=200,
                            content_type='application/json')

    async def handle_list_models(self, request: web.Request) -> web.Response:
        """Handle list models requests"""
        try:
            models = self.router.list_models()
            model_info = {}

            for model_name in models:
                endpoint = self.router.get_model_endpoint(model_name)
                model_info[model_name] = {"endpoint": endpoint}

            return web.Response(text=json.dumps({"models": model_info}),
                                status=200,
                                content_type='application/json')

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_model_health(self, request: web.Request) -> web.Response:
        """Handle model health check requests"""
        try:
            model_name = request.match_info['model_name']
            health_status = await self.router.health_check(model_name)

            return web.Response(text=json.dumps({
                "model": model_name,
                "health": health_status
            }),
                                status=200,
                                content_type='application/json')

        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_get_server_info(self,
                                     request: web.Request) -> web.Response:
        """Handle get server info requests"""
        # return dummy data
        return web.Response(text=json.dumps({"status": "dummy"}),
                            status=200,
                            content_type='application/json')

    async def start(self):
        """Start the router server"""
        logger.info(f"Starting router server on port {self.port}")

        # Start LLM servers based on the configuration
        # await self.router.start_llm_servers()

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logger.info(f"Router server started on http://0.0.0.0:{self.port}")

        # Keep the server running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down router server...")
        finally:
            await self.router.close()
            await runner.cleanup()


def _extract_models_mapping(
        raw_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build the model→endpoint mapping consumed by the router frontend.

    This mirrors the logic in controller.launch so that the frontend can run
    standalone given only the master YAML config file.
    """

    models_mapping: Dict[str, Dict[str, Any]] = {}

    for inst in raw_cfg.get("instances", []):
        model_name = inst["model"]
        inst_name = inst.get("name", model_name)

        # Defaults
        host: str = "localhost"
        port: Optional[int] = None

        raw_args = inst.get("engine_args", inst.get("args", []))

        # Normalize args to a flat list of strings
        if isinstance(raw_args, str):
            arg_list: List[str] = shlex.split(raw_args)
        else:
            arg_list: List[str] = []
            for item in raw_args:
                arg_list.extend(shlex.split(str(item)))

        # Detect --host / --port options
        for idx, token in enumerate(arg_list):
            if token.startswith("--host="):
                host = token.split("=", 1)[1]
            elif token == "--host" and idx + 1 < len(arg_list):
                host = arg_list[idx + 1]
            elif token.startswith("--port="):
                try:
                    port = int(token.split("=", 1)[1])
                except ValueError:
                    pass
            elif token == "--port" and idx + 1 < len(arg_list):
                try:
                    port = int(arg_list[idx + 1])
                except ValueError:
                    pass

        if port is None:
            logger.warning(
                "Could not determine port for instance %s – skipping in router mapping",
                inst_name,
            )
            continue

        # Use instance name as key to support multiple instances of same model
        models_mapping[inst_name] = {
            "endpoint": {"host": host, "port": port},
            "actual_model_name": model_name  # Keep track of actual model name
        }

    return models_mapping


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='LLM Router Server')
    parser.add_argument(
        '--config',
        required=True,
        help='Path to YAML configuration file (e.g. example-config.yaml)')
    parser.add_argument('--port',
                        type=int,
                        default=8080,
                        help='Port to run the server on')

    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"YAML config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)

    models_mapping = _extract_models_mapping(raw_cfg)
    models_config = {"models": models_mapping}

    server = MultiLLMFrontend(
        port=args.port,
        model_config_json=models_config,
    )
    await server.start()


if __name__ == '__main__':
    asyncio.run(main())
