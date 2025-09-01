import asyncio
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from aiohttp import web
from router import LLMRouter
from sleep_manager import SleepManager
from traffic_monitor import TrafficMonitor
from utils import set_ulimit

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


class MultiLLMFrontend:

    def __init__(self, port: int, model_config_json: Dict[str, Any]):
        self.traffic_monitor = TrafficMonitor()
        self.sleep_manager = SleepManager(traffic_monitor=self.traffic_monitor)
        self.router = LLMRouter(models_config=model_config_json,
                                sleep_manager=self.sleep_manager,
                                traffic_monitor=self.traffic_monitor)
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

        # Traffic monitoring endpoints
        self.app.router.add_get('/traffic/stats', self.handle_traffic_stats)
        self.app.router.add_get('/traffic/stats/{model_name}',
                                self.handle_model_traffic_stats)

        # Model idle/active status check endpoints
        self.app.router.add_get('/models/idle', self.handle_list_idle_models)
        self.app.router.add_get('/models/active',
                                self.handle_list_active_models)

        # Sleep management endpoints
        self.app.router.add_get('/sleep/status', self.handle_sleep_status)
        self.app.router.add_get('/sleep/candidates',
                                self.handle_sleep_candidates)
        self.app.router.add_post('/action/sleep/{model_name}',
                                 self.handle_model_sleep)
        self.app.router.add_post('/action/wakeup/{model_name}',
                                 self.handle_model_wakeup)

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

    async def handle_traffic_stats(self, request: web.Request) -> web.Response:
        """Handle traffic statistics requests for all models"""
        try:
            # Get optional query parameters
            window_seconds = int(request.query.get('window', 60))

            stats = self.traffic_monitor.get_traffic_summary(window_seconds)

            return web.Response(text=json.dumps({
                "traffic_stats":
                stats,
                "window_seconds":
                window_seconds
            }),
                                status=200,
                                content_type='application/json')
        except Exception as e:
            logger.error(f"Error getting traffic stats: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_model_traffic_stats(self,
                                         request: web.Request) -> web.Response:
        """Handle traffic statistics requests for a specific model"""
        try:
            import urllib.parse
            model_name = urllib.parse.unquote(request.match_info['model_name'])
            window_seconds = int(request.query.get('window', 60))

            model_stats = self.traffic_monitor.get_model_stats(model_name)
            if not model_stats:
                return web.Response(text=json.dumps(
                    {"error":
                     f"No traffic data found for model {model_name}"}),
                                    status=404,
                                    content_type='application/json')

            stats = {
                'model_name': model_name,
                'total_requests': model_stats.total_requests,
                'successful_requests': model_stats.successful_requests,
                'failed_requests': model_stats.failed_requests,
                'request_rate': model_stats.get_request_rate(window_seconds),
                'avg_response_time': model_stats.avg_response_time,
                'last_activity_time': model_stats.last_activity_time,
                'idle_time_seconds': model_stats.get_idle_time(),
                'is_idle': model_stats.is_idle()
            }

            return web.Response(text=json.dumps({
                "model_stats":
                stats,
                "window_seconds":
                window_seconds
            }),
                                status=200,
                                content_type='application/json')
        except Exception as e:
            logger.error(f"Error getting model traffic stats: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_list_idle_models(self,
                                      request: web.Request) -> web.Response:
        """Handle requests for idle models that could be put to sleep"""
        try:
            # Get optional threshold parameter
            idle_threshold = int(request.query.get('threshold',
                                                   300))  # Default 5 minutes

            idle_models = self.traffic_monitor.get_idle_models(idle_threshold)

            # Get detailed stats for idle models
            idle_model_stats = {}
            for model_name in idle_models:
                model_stats = self.traffic_monitor.get_model_stats(model_name)
                if model_stats:
                    idle_model_stats[model_name] = {
                        'idle_time_seconds': model_stats.get_idle_time(),
                        'total_requests': model_stats.total_requests,
                        'last_activity_time': model_stats.last_activity_time
                    }

            return web.Response(text=json.dumps({
                "idle_models":
                idle_models,
                "idle_threshold_seconds":
                idle_threshold,
                "idle_model_details":
                idle_model_stats,
                "idle_models_count":
                len(idle_models)
            }),
                                status=200,
                                content_type='application/json')
        except Exception as e:
            logger.error(f"Error getting idle models: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_list_active_models(self,
                                        request: web.Request) -> web.Response:
        """Handle requests for active models"""
        try:
            # Get optional threshold parameter
            idle_threshold = int(request.query.get('threshold',
                                                   300))  # Default 5 minutes
            window_seconds = int(request.query.get('window', 60))

            active_models = self.traffic_monitor.get_active_models(
                idle_threshold)

            # Get detailed stats for active models
            active_model_stats = {}
            for model_name in active_models:
                model_stats = self.traffic_monitor.get_model_stats(model_name)
                if model_stats:
                    active_model_stats[model_name] = {
                        'request_rate':
                        model_stats.get_request_rate(window_seconds),
                        'total_requests':
                        model_stats.total_requests,
                        'avg_response_time':
                        model_stats.avg_response_time,
                        'last_activity_time':
                        model_stats.last_activity_time
                    }

            return web.Response(text=json.dumps({
                "active_models":
                active_models,
                "idle_threshold_seconds":
                idle_threshold,
                "window_seconds":
                window_seconds,
                "active_model_details":
                active_model_stats,
                "active_models_count":
                len(active_models)
            }),
                                status=200,
                                content_type='application/json')
        except Exception as e:
            logger.error(f"Error getting active models: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_sleep_status(self, request: web.Request) -> web.Response:
        """Handle sleep status requests"""
        try:
            sleeping_models = self.sleep_manager.get_sleeping_models()
            candidates = self.sleep_manager.get_sleep_candidates()

            return web.Response(text=json.dumps({
                "sleeping_models":
                sleeping_models,
                "sleep_candidates":
                candidates,
                "auto_sleep_enabled":
                self.sleep_manager.config.auto_sleep_enabled,
                "idle_threshold_seconds":
                self.sleep_manager.config.idle_threshold_seconds,
                "wakeup_on_request":
                self.sleep_manager.config.wake_on_request
            }),
                                status=200,
                                content_type='application/json')
        except Exception as e:
            logger.error(f"Error getting sleep status: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_sleep_candidates(self,
                                      request: web.Request) -> web.Response:
        """Handle requests for models that are candidates for sleep mode"""
        try:
            candidates = self.sleep_manager.get_sleep_candidates()

            # Get detailed info for each candidate
            candidate_details = {}
            for model_name in candidates:
                model_stats = self.traffic_monitor.get_model_stats(model_name)
                if model_stats:
                    candidate_details[model_name] = {
                        'idle_time_seconds': model_stats.get_idle_time(),
                        'total_requests': model_stats.total_requests,
                        'last_activity_time': model_stats.last_activity_time,
                        'can_sleep': True
                    }

            return web.Response(text=json.dumps({
                "sleep_candidates":
                candidates,
                "candidate_details":
                candidate_details,
                "idle_threshold_seconds":
                self.sleep_manager.config.idle_threshold_seconds,
                "auto_sleep_enabled":
                self.sleep_manager.config.auto_sleep_enabled
            }),
                                status=200,
                                content_type='application/json')
        except Exception as e:
            logger.error(f"Error getting sleep candidates: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_model_sleep(self, request: web.Request) -> web.Response:
        """Handle requests to put a model to sleep"""
        try:
            import urllib.parse
            model_name = urllib.parse.unquote(request.match_info['model_name'])

            # Check if model exists
            if model_name not in self.router.models:
                return web.Response(text=json.dumps(
                    {"error": f"Model {model_name} not found"}),
                                    status=404,
                                    content_type='application/json')

            success = await self.sleep_manager.put_model_to_sleep(model_name,
                                                                  manual=True)

            return web.Response(text=json.dumps({
                "model_name":
                model_name,
                "success":
                success,
                "message":
                f"Model {model_name} sleep request {'successful' if success else 'failed'}"
            }),
                                status=200 if success else 400,
                                content_type='application/json')
        except Exception as e:
            logger.error(f"Error putting model to sleep: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def handle_model_wakeup(self, request: web.Request) -> web.Response:
        """Handle requests to wake up a sleeping model"""
        try:
            import urllib.parse
            model_name = urllib.parse.unquote(request.match_info['model_name'])
            # Check if model exists
            if model_name not in self.router.models:
                return web.Response(text=json.dumps(
                    {"error": f"Model {model_name} not found"}),
                                    status=404,
                                    content_type='application/json')

            success = await self.sleep_manager.wake_model(model_name)

            return web.Response(text=json.dumps({
                "model_name":
                model_name,
                "success":
                success,
                "message":
                f"Model {model_name} wake request {'successful' if success else 'failed'}"
            }),
                                status=200 if success else 400,
                                content_type='application/json')
        except Exception as e:
            logger.error(f"Error waking up model: {e}")
            return web.Response(text=json.dumps({"error": str(e)}),
                                status=500,
                                content_type='application/json')

    async def start(self):
        """Start the router server"""
        logger.info(f"Starting router server on port {self.port}")

        # Start traffic monitor and sleep manager
        await self.traffic_monitor.start()
        await self.sleep_manager.start()

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
            await self.traffic_monitor.stop()
            await self.sleep_manager.stop()
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
                "Could not determine port for model %s – skipping in router mapping",
                model_name,
            )
            continue

        models_mapping[model_name] = {"endpoint": {"host": host, "port": port}}

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
