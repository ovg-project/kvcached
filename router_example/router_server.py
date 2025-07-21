#!/usr/bin/env python3

import asyncio
import logging
import json
from typing import Dict, Any
from aiohttp import web, ClientSession
from .router import LLMRouter

logger = logging.getLogger(__name__)


class RouterServer:
    def __init__(self, config_path: str, port: int = 8080):
        self.router = LLMRouter(config_path)
        self.port = port
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_post('/v1/completions', self.handle_completion)
        self.app.router.add_post('/v1/chat/completions', self.handle_chat_completion)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/models', self.handle_list_models)
        self.app.router.add_get('/health/{model_name}', self.handle_model_health)
    
    async def handle_completion(self, request: web.Request) -> web.Response:
        """Handle completion requests"""
        try:
            request_data = await request.json()
            model_name = request_data.get('model')
            
            if not model_name:
                return web.Response(
                    text=json.dumps({"error": "model parameter is required"}),
                    status=400,
                    content_type='application/json'
                )
            
            # Route the request
            result = await self.router.route_request(model_name, request_data, "/v1/completions")
            
            if result is None:
                return web.Response(
                    text=json.dumps({"error": f"Failed to route request for model {model_name}"}),
                    status=500,
                    content_type='application/json'
                )
            
            # Check if this is a streaming response
            if hasattr(result, 'content') and hasattr(result, 'headers'):
                # This is a streaming response (aiohttp.ClientResponse)
                response = web.StreamResponse(
                    status=result.status,
                    headers={
                        'Content-Type': result.headers.get('Content-Type', 'text/event-stream'),
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive'
                    }
                )
                await response.prepare(request)
                
                try:
                    async for chunk in result.content.iter_chunked(1024):
                        await response.write(chunk)
                    return response
                finally:
                    await result.release()
            else:
                # This is a regular JSON response
                return web.Response(
                    text=json.dumps(result),
                    status=200,
                    content_type='application/json'
                )
            
        except Exception as e:
            logger.error(f"Error handling completion request: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_chat_completion(self, request: web.Request) -> web.Response:
        """Handle chat completion requests"""
        try:
            request_data = await request.json()
            model_name = request_data.get('model')
            
            if not model_name:
                return web.Response(
                    text=json.dumps({"error": "model parameter is required"}),
                    status=400,
                    content_type='application/json'
                )
            
            # Route the request
            result = await self.router.route_request(model_name, request_data, "/v1/chat/completions")
            
            if result is None:
                return web.Response(
                    text=json.dumps({"error": f"Failed to route request for model {model_name}"}),
                    status=500,
                    content_type='application/json'
                )
            
            # Check if this is a streaming response
            if hasattr(result, 'content') and hasattr(result, 'headers'):
                # This is a streaming response (aiohttp.ClientResponse)
                response = web.StreamResponse(
                    status=result.status,
                    headers={
                        'Content-Type': result.headers.get('Content-Type', 'text/event-stream'),
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive'
                    }
                )
                await response.prepare(request)
                
                try:
                    async for chunk in result.content.iter_chunked(1024):
                        await response.write(chunk)
                    return response
                finally:
                    await result.release()
            else:
                # This is a regular JSON response
                return web.Response(
                    text=json.dumps(result),
                    status=200,
                    content_type='application/json'
                )
            
        except Exception as e:
            logger.error(f"Error handling chat completion request: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests"""
        return web.Response(
            text=json.dumps({"status": "healthy"}),
            status=200,
            content_type='application/json'
        )
    
    async def handle_list_models(self, request: web.Request) -> web.Response:
        """Handle list models requests"""
        try:
            models = self.router.list_models()
            model_info = {}
            
            for model_name in models:
                endpoint = self.router.get_model_endpoint(model_name)
                model_info[model_name] = {
                    "endpoint": endpoint
                }
            
            return web.Response(
                text=json.dumps({"models": model_info}),
                status=200,
                content_type='application/json'
            )
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_model_health(self, request: web.Request) -> web.Response:
        """Handle model health check requests"""
        try:
            model_name = request.match_info['model_name']
            health_status = await self.router.health_check(model_name)
            
            return web.Response(
                text=json.dumps({"model": model_name, "health": health_status}),
                status=200,
                content_type='application/json'
            )
            
        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def start(self):
        """Start the router server"""
        logger.info(f"Starting router server on port {self.port}")

        # Start LLM servers based on the configuration
        await self.router.start_llm_servers()
        
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


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Router Server')
    parser.add_argument('--config', required=True, help='Path to router configuration file')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start the server
    server = RouterServer(args.config, args.port)
    await server.start()


if __name__ == '__main__':
    asyncio.run(main())