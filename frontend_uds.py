#!/usr/bin/env python3
"""frontend_uds.py -- UDS version of the mainline kvcached controller
frontend.

Subclasses `LLMRouter` and `SleepManager` from /root/kvcached/controller/ so
that every HTTP call (inference routing, sleep, wake, is_sleeping) goes over
a Unix Domain Socket at /tmp/vllm_<port>.sock instead of TCP.

Everything else -- TrafficMonitor, MultiLLMFrontend routes, sleep/wake
decision logic, auto-sleep of idle models -- is reused as-is.

Usage:
    python3 frontend_uds.py --config_path controller-config.yaml --port 8080

The config YAML uses the same shape as controller/example-config.yaml.  The
host/port in `engine_args` are interpreted as the TCP port ONLY to derive
the UDS path (/tmp/vllm_<port>.sock).
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import aiohttp
import yaml

# Make main-branch controller/ importable
KVCACHED_ROOT = Path("/root/kvcached")
CONTROLLER_DIR = KVCACHED_ROOT / "controller"
sys.path.insert(0, str(KVCACHED_ROOT))
sys.path.insert(0, str(CONTROLLER_DIR))

from frontend import MultiLLMFrontend, _extract_sleep_config  # noqa: E402
from router import LLMRouter, ModelConfig, Endpoint  # noqa: E402
from sleep_manager import SleepManager  # noqa: E402
from traffic_monitor import TrafficMonitor  # noqa: E402
from utils import extract_models_mapping, set_ulimit  # noqa: E402


def uds_path_for_port(port: int) -> str:
    return f"/tmp/vllm_{port}.sock"


class UDSLLMRouter(LLMRouter):
    """LLMRouter that talks to each engine over /tmp/vllm_<port>.sock."""

    def __init__(self, models_config, *, sleep_manager, traffic_monitor,
                 known_ports: Optional[set] = None):
        # Skip parent __init__ and build our own state so we don't open a
        # TCPConnector-backed session that we never use.
        self.sleep_manager = sleep_manager
        self.traffic_monitor = traffic_monitor
        self.models = {}
        self._connector = None
        self.session = None
        # Set of every engine port we know about, even if multiple engines
        # share the same model name.  The mainline router's models_config
        # dict can only hold one endpoint per model, so without this we'd
        # ignore the bench script's "port" hint and funnel all traffic to
        # the single deduped endpoint.
        self.known_ports: set = set(known_ports or [])

        # --- LRU eviction bookkeeping ---
        # engine state: port -> {"sleeping", "timestamp", "mem_mb_needed"}
        # mem_mb_needed is learned from the first successful wake (peak
        # GPU delta) and used to decide whether another engine must be
        # evicted before waking this one.
        self._engine_state: Dict[int, Dict[str, Any]] = {
            p: {"sleeping": True, "timestamp": 0.0, "mem_mb_needed": None}
            for p in self.known_ports
        }
        # Serialize all wake/sleep decisions so concurrent requests don't
        # race to wake multiple engines and OOM the GPU.
        self._sleep_wake_lock = asyncio.Lock()
        # Conservative fallback for the first wake of an engine we haven't
        # measured yet.  Overestimates a bit so the first wake rarely OOMs.
        self._default_mem_need_mb = 30 * 1024
        self._safety_margin_mb = 1024

        self.load_config_from_dict(models_config)

    # --- LRU eviction helpers -------------------------------------------------

    @staticmethod
    def _get_free_gpu_mb() -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=3)
            return float(out.strip().split("\n")[0])
        except Exception:
            return 0.0

    def _pick_lru_awake(self, except_port: int) -> Optional[int]:
        """Return the port of the oldest-used awake engine (LRU)."""
        awake = [(p, s["timestamp"]) for p, s in self._engine_state.items()
                 if not s["sleeping"] and p != except_port]
        if not awake:
            return None
        return min(awake, key=lambda x: x[1])[0]

    async def _drain_and_sleep(self, port: int) -> bool:
        """Wait for in-flight requests, then POST /sleep?level=2 over UDS."""
        uds = uds_path_for_port(port)
        deadline = time.monotonic() + 10.0
        # Drain: poll vllm:num_requests_running until 0 (10s max)
        while time.monotonic() < deadline:
            try:
                connector = aiohttp.UnixConnector(path=uds)
                async with aiohttp.ClientSession(connector=connector) as s:
                    async with s.get("http://localhost/metrics",
                                     timeout=aiohttp.ClientTimeout(total=3)) as r:
                        if r.status == 200:
                            text = await r.text()
                            running = 0.0
                            for line in text.split("\n"):
                                if line.startswith("vllm:num_requests_running"):
                                    try:
                                        running = float(line.split()[-1])
                                    except Exception:
                                        pass
                                    break
                            if running == 0:
                                break
            except Exception:
                break
            await asyncio.sleep(0.3)

        try:
            connector = aiohttp.UnixConnector(path=uds)
            async with aiohttp.ClientSession(connector=connector) as s:
                async with s.post("http://localhost/sleep?level=2",
                                  timeout=aiohttp.ClientTimeout(total=15)) as r:
                    if r.status == 200:
                        self._engine_state[port]["sleeping"] = True
                        return True
        except Exception:
            pass
        return False

    async def _ensure_capacity(self, target_port: int):
        """Sleep LRU awake engines until there's room for target_port."""
        need_mb = (self._engine_state[target_port].get("mem_mb_needed")
                   or self._default_mem_need_mb) + self._safety_margin_mb
        while self._get_free_gpu_mb() < need_mb:
            victim = self._pick_lru_awake(except_port=target_port)
            if victim is None:
                return  # nothing more we can evict
            print(f"[uds-router] evicting LRU engine {victim} "
                  f"(free={self._get_free_gpu_mb():.0f} MB, need={need_mb:.0f})")
            await self._drain_and_sleep(victim)

    async def route_request(
        self,
        model_name: str,
        request_data: Dict[str, Any],
        endpoint_path: str = "/v1/completions",
    ) -> Optional[Any]:
        stats = self.traffic_monitor.record_request_start(model_name, endpoint_path)

        # Honor the bench script's "port" hint if present.  The mainline
        # router only stores one endpoint per model name, so without this
        # we'd ignore the distribution choice and funnel everything to a
        # single engine.  The hint may live at the top level of the request
        # body or inside extra_body.
        hinted_port = request_data.get("port")
        if hinted_port is None and isinstance(request_data.get("extra_body"), dict):
            hinted_port = request_data["extra_body"].get("port")
        if hinted_port is not None:
            try:
                hinted_port = int(hinted_port)
            except (TypeError, ValueError):
                hinted_port = None

        if hinted_port is not None and hinted_port in self.known_ports:
            uds = uds_path_for_port(hinted_port)
            # Strip "port" from the body so vLLM doesn't choke on it.
            request_data = {k: v for k, v in request_data.items() if k != "port"}
            if isinstance(request_data.get("extra_body"), dict):
                eb = {k: v for k, v in request_data["extra_body"].items()
                      if k != "port"}
                request_data["extra_body"] = eb if eb else None
        else:
            endpoint = self.get_endpoint_for_model(model_name)
            if endpoint is None:
                self.traffic_monitor.record_request_end(
                    stats, success=False, error_message=f"Model {model_name} not found")
                return None
            uds = uds_path_for_port(endpoint.port)

        # Derive the numeric port we're routing to (for engine-state bookkeeping).
        target_port = hinted_port if (hinted_port is not None
                                      and hinted_port in self.known_ports) else None
        if target_port is None:
            # uds was derived from endpoint.port above
            target_port = endpoint.port if endpoint is not None else None

        # Robust wake check + LRU eviction: serialized under _sleep_wake_lock so
        # concurrent requests don't race to wake multiple engines and OOM the
        # GPU (which is exactly what led to 85% request drops in round-robin).
        actual_sleeping = await self._is_engine_sleeping(uds)
        if actual_sleeping:
            async with self._sleep_wake_lock:
                # Re-check inside the lock: another coroutine may have just woken
                # this engine.
                actual_sleeping = await self._is_engine_sleeping(uds)
                if actual_sleeping:
                    if target_port is not None:
                        await self._ensure_capacity(target_port)

                    mem_before = self._get_free_gpu_mb()
                    woke = await self._wake_engine(uds)
                    mem_after = self._get_free_gpu_mb()
                    if not woke:
                        self.traffic_monitor.record_request_end(
                            stats, success=False,
                            error_message=f"Failed to wake {model_name} on UDS {uds}")
                        return None

                    if target_port is not None:
                        st = self._engine_state.setdefault(
                            target_port,
                            {"sleeping": False, "timestamp": 0.0,
                             "mem_mb_needed": None})
                        st["sleeping"] = False
                        st["timestamp"] = time.time()
                        # Learn the actual memory footprint (once) so future
                        # eviction decisions are accurate.
                        if st.get("mem_mb_needed") is None:
                            delta = mem_before - mem_after
                            if delta > 1024:  # ignore noise
                                st["mem_mb_needed"] = delta
                    # Keep the inherited SleepManager state in sync.
                    self.sleep_manager.sleeping_models.pop(model_name, None)

        # Touch the LRU timestamp for the engine we're about to hit so it's
        # less likely to be evicted while this request is in flight.
        if target_port is not None and target_port in self._engine_state:
            self._engine_state[target_port]["sleeping"] = False
            self._engine_state[target_port]["timestamp"] = time.time()
        is_streaming = bool(request_data.get("stream", False))
        timeout = aiohttp.ClientTimeout(total=300)
        url = f"http://localhost{endpoint_path}"

        # Per-request session bound to this engine's UDS
        connector = aiohttp.UnixConnector(path=uds)
        session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        try:
            if is_streaming:
                resp = await session.post(url, json=request_data)
                if resp.status == 200:
                    self.traffic_monitor.record_request_end(stats, success=True)
                    # Stash session on the response so the caller can close it
                    resp._uds_session = session  # type: ignore[attr-defined]
                    return resp
                text = await resp.text()
                await resp.release()
                await session.close()
                self.traffic_monitor.record_request_end(
                    stats, success=False, error_message=f"HTTP {resp.status}: {text}")
                return None

            async with session.post(url, json=request_data) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    self.traffic_monitor.record_request_end(stats, success=True)
                    return body
                text = await resp.text()
                self.traffic_monitor.record_request_end(
                    stats, success=False, error_message=f"HTTP {resp.status}: {text}")
                return None
        except Exception as e:
            self.traffic_monitor.record_request_end(
                stats, success=False, error_message=str(e))
            return None
        finally:
            if not is_streaming:
                await session.close()

    @staticmethod
    async def _is_engine_sleeping(uds: str) -> bool:
        """Query an engine's /is_sleeping over UDS. Returns False on error so we
        don't block requests when the engine is reachable but not in sleep mode
        (or when /is_sleeping isn't implemented)."""
        connector = aiohttp.UnixConnector(path=uds)
        try:
            async with aiohttp.ClientSession(connector=connector) as s:
                async with s.get("http://localhost/is_sleeping",
                                 timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status != 200:
                        return False
                    return bool((await r.json()).get("is_sleeping", False))
        except Exception:
            return False

    @staticmethod
    async def _wake_engine(uds: str) -> bool:
        """Issue /wake_up over UDS. Bypasses SleepManager.min_sleep_duration so
        that engines pre-slept by start_vllm_servers.sh wake immediately."""
        connector = aiohttp.UnixConnector(path=uds)
        try:
            async with aiohttp.ClientSession(connector=connector) as s:
                async with s.post("http://localhost/wake_up",
                                  timeout=aiohttp.ClientTimeout(total=120)) as r:
                    return r.status == 200
        except Exception:
            return False

    async def health_check(self, model_name: str) -> Dict[str, bool]:
        if model_name == "all":
            return {m: await self.health_check(m) for m in self.models}
        if model_name not in self.models:
            return {}
        endpoint = self.models[model_name].endpoint
        uds = uds_path_for_port(endpoint.port)
        connector = aiohttp.UnixConnector(path=uds)
        async with aiohttp.ClientSession(connector=connector) as s:
            try:
                async with s.get(f"http://localhost{endpoint.health_check_path}",
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    return {endpoint.base_url: r.status == 200}
            except Exception:
                return {endpoint.base_url: False}

    async def close(self):
        # Nothing persistent to close; per-request sessions self-clean
        return


class UDSSleepManager(SleepManager):
    """SleepManager that issues /sleep, /wake_up, /is_sleeping over UDS."""

    @staticmethod
    async def _uds_post(port: str, path: str, payload=None, timeout=30) -> bool:
        uds = uds_path_for_port(int(port))
        connector = aiohttp.UnixConnector(path=uds)
        async with aiohttp.ClientSession(connector=connector) as s:
            try:
                async with s.post(f"http://localhost{path}",
                                  json=payload or {},
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                    return r.status == 200
            except Exception:
                return False

    async def _call_vllm_sleep_api(self, host, port, level=1):
        return await self._uds_post(port, "/sleep", {"level": str(level)})

    async def _call_vllm_wakeup_api(self, host, port):
        return await self._uds_post(port, "/wake_up")

    async def check_model_sleep_status(self, model_name):
        if model_name not in self.config.vllm_models_config:
            return None
        port = self.config.vllm_models_config[model_name].get("port", "8000")
        uds = uds_path_for_port(int(port))
        connector = aiohttp.UnixConnector(path=uds)
        async with aiohttp.ClientSession(connector=connector) as s:
            try:
                async with s.get("http://localhost/is_sleeping",
                                 timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        return None
                    return (await r.json()).get("is_sleeping", False)
            except Exception:
                return None


class UDSMultiLLMFrontend(MultiLLMFrontend):
    """MultiLLMFrontend wired to the UDS-backed router and sleep manager."""

    def __init__(self, port, model_config_json, sleep_config,
                 known_ports: Optional[set] = None):
        from aiohttp import web  # noqa: F401 -- parent uses it

        self.traffic_monitor = TrafficMonitor()
        self.sleep_manager = UDSSleepManager(
            config=sleep_config, traffic_monitor=self.traffic_monitor)
        self.router = UDSLLMRouter(
            models_config=model_config_json,
            sleep_manager=self.sleep_manager,
            traffic_monitor=self.traffic_monitor,
            known_ports=known_ports)
        self.port = port

        # Recreate the parent's app / route table
        import aiohttp.web as aio_web
        self.app = aio_web.Application()
        self.configure_endpoints()
        # --- Admin endpoints for live engine registration ---
        self.app.router.add_post('/admin/engines', self.handle_register_engine)
        self.app.router.add_delete('/admin/engines/{port}',
                                   self.handle_deregister_engine)
        self.app.router.add_get('/admin/engines', self.handle_list_engines)
        set_ulimit()

    @staticmethod
    async def _probe_engine(uds: str) -> bool:
        """Quick health probe so we don't register dead engines."""
        connector = aiohttp.UnixConnector(path=uds)
        try:
            async with aiohttp.ClientSession(connector=connector) as s:
                async with s.get("http://localhost/health",
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    return r.status == 200
        except Exception:
            return False

    async def handle_register_engine(self, request):
        """POST /admin/engines  {"model": str, "port": int, "host": "localhost"}
        Hook an already-running vLLM engine into the router, LRU bookkeeping,
        and sleep manager.  Does NOT launch the engine itself.
        """
        import aiohttp.web as aio_web
        try:
            body = await request.json()
        except Exception:
            return aio_web.json_response({"error": "invalid JSON"}, status=400)

        model = body.get("model")
        port = body.get("port")
        host = body.get("host", "localhost")
        if not model or port is None:
            return aio_web.json_response(
                {"error": "both 'model' and 'port' are required"}, status=400)
        try:
            port = int(port)
        except (TypeError, ValueError):
            return aio_web.json_response(
                {"error": "'port' must be an integer"}, status=400)

        uds = uds_path_for_port(port)
        if not await self._probe_engine(uds):
            return aio_web.json_response(
                {"error": f"engine at {uds} is not reachable"}, status=503)

        # Detect actual sleep state so the LRU bookkeeping starts honest
        sleeping = await self.router._is_engine_sleeping(uds)

        # 1. Router: add to models dict (lookups by model name)
        self.router.add_model(ModelConfig(
            model, Endpoint(host=host, port=port)))

        # 2. Router: known_ports (enables port-hint routing) + engine state
        self.router.known_ports.add(port)
        self.router._engine_state[port] = {
            "sleeping": bool(sleeping),
            "timestamp": time.time(),
            "mem_mb_needed": None,
        }

        # 3. SleepManager: vllm_models_config (so sleep/wake can be driven by model name)
        self.sleep_manager.add_vllm_model(model, host, str(port))

        print(f"[admin] registered engine model={model} port={port} "
              f"uds={uds} sleeping={sleeping}")
        return aio_web.json_response({
            "status": "ok",
            "model": model,
            "port": port,
            "sleeping": bool(sleeping),
            "total_known_ports": len(self.router.known_ports),
        })

    async def handle_deregister_engine(self, request):
        """DELETE /admin/engines/{port}  Remove engine bookkeeping.
        Does NOT stop the engine process."""
        import aiohttp.web as aio_web
        try:
            port = int(request.match_info["port"])
        except (TypeError, ValueError):
            return aio_web.json_response(
                {"error": "bad port path segment"}, status=400)

        removed_model = None
        # Remove port from router state
        self.router.known_ports.discard(port)
        self.router._engine_state.pop(port, None)

        # Find the model name(s) mapped to this port and remove them
        to_remove = [name for name, mc in self.router.models.items()
                     if mc.endpoint.port == port]
        for name in to_remove:
            self.router.models.pop(name, None)
            self.sleep_manager.remove_vllm_model(name)
            self.sleep_manager.sleeping_models.pop(name, None)
            removed_model = name

        print(f"[admin] deregistered port={port} model={removed_model}")
        return aio_web.json_response({
            "status": "ok",
            "port": port,
            "removed_model": removed_model,
            "total_known_ports": len(self.router.known_ports),
        })

    async def handle_list_engines(self, request):
        """GET /admin/engines  Inspect current bookkeeping."""
        import aiohttp.web as aio_web
        engines = []
        for port in sorted(self.router.known_ports):
            model_for_port = next(
                (name for name, mc in self.router.models.items()
                 if mc.endpoint.port == port),
                None)
            state = self.router._engine_state.get(port, {})
            engines.append({
                "port": port,
                "model": model_for_port,
                "sleeping": state.get("sleeping"),
                "last_used": state.get("timestamp"),
                "mem_mb_needed": state.get("mem_mb_needed"),
            })
        return aio_web.json_response({
            "engines": engines,
            "total": len(engines),
        })


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg_path = args.config_path.expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    with cfg_path.open() as f:
        raw_cfg = yaml.safe_load(f)

    models_mapping = extract_models_mapping(raw_cfg)
    sleep_config = _extract_sleep_config(raw_cfg)

    # Walk every instance in the YAML to collect ALL engine ports, even
    # when several instances share the same model name (mainline's
    # extract_models_mapping dedupes by model name).
    known_ports: set = set()
    import shlex as _shlex, argparse as _argparse
    _p = _argparse.ArgumentParser(add_help=False)
    _p.add_argument("--port", type=int)
    for inst in raw_cfg.get("instances", []):
        raw_args = inst.get("engine_args", inst.get("args", []))
        if isinstance(raw_args, str):
            arg_list = _shlex.split(raw_args)
        else:
            arg_list = []
            for item in raw_args:
                arg_list.extend(_shlex.split(str(item)))
        ka, _ = _p.parse_known_args(arg_list)
        if ka.port is not None:
            known_ports.add(int(ka.port))
    print(f"Known engine ports: {sorted(known_ports)}")

    server = UDSMultiLLMFrontend(
        port=args.port,
        model_config_json={"models": models_mapping},
        sleep_config=sleep_config,
        known_ports=known_ports,
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
