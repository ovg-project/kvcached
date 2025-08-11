#!/usr/bin/env python3
"""
直接进程方式的自动化benchmark脚本
不使用tmux，直接启动进程并实时记录日志
"""

import argparse
import json
import shutil
import subprocess
import time
import yaml
from datetime import datetime
from pathlib import Path
import os
import sys


def run_cmd(cmd, timeout=None, cwd=None):
    """运行命令并返回结果"""
    try:
        if isinstance(cmd, str):
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, 
                timeout=timeout, cwd=cwd
            )
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                timeout=timeout, cwd=cwd
            )
        return result
    except subprocess.TimeoutExpired:
        print(f"命令超时: {cmd}")
        return subprocess.CompletedProcess(cmd, -1, "", "Timeout")
    except Exception as e:
        print(f"命令执行失败: {cmd}, 错误: {e}")
        return subprocess.CompletedProcess(cmd, -1, "", str(e))

class ProcessLogger:
    """进程日志记录器"""
    def __init__(self, name, log_file):
        self.name = name
        self.log_file = log_file
        self.process = None
        
    def start_process(self, cmd, cwd=None, env=None):
        """启动进程并记录日志"""
        print(f"🚀 启动 {self.name}...")
        print(f"💻 命令: {cmd}")
        print(f"📝 日志文件: {self.log_file}")
        
        try:
            # 创建日志文件目录
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 启动进程，将stdout和stderr都重定向到文件
            with open(self.log_file, 'w') as f:
                f.write(f"=== {self.name} 启动日志 ===\n")
                f.write(f"启动时间: {datetime.now()}\n")
                f.write(f"工作目录: {cwd}\n")
                f.write(f"命令: {cmd}\n")
                f.write("=" * 50 + "\n\n")
                f.flush()
                
                if isinstance(cmd, str):
                    self.process = subprocess.Popen(
                        cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                        cwd=cwd, env=env, text=True, bufsize=1
                    )
                else:
                    self.process = subprocess.Popen(
                        cmd, stdout=f, stderr=subprocess.STDOUT,
                        cwd=cwd, env=env, text=True, bufsize=1
                    )
                    
            print(f"✅ {self.name} 已启动 (PID: {self.process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ 启动 {self.name} 失败: {e}")
            return False
    
    def is_running(self):
        """检查进程是否在运行"""
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def stop(self):
        """停止进程"""
        if self.process and self.is_running():
            print(f"🛑 停止 {self.name} (PID: {self.process.pid})")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"强制杀死 {self.name}")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"停止 {self.name} 时出错: {e}")

def kill_port_processes(ports):
    """杀死占用指定端口的进程"""
    print("🔪 清理端口占用的进程...")
    
    for port in ports:
        try:
            result = run_cmd(["lsof", "-ti", f":{port}"])
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        print(f"  杀死端口 {port} 的进程 PID: {pid}")
                        run_cmd(["kill", "-9", pid])
            else:
                print(f"  端口 {port} 未被占用")
        except Exception as e:
            print(f"  清理端口 {port} 时出错: {e}")

def wait_for_service(port, max_wait=60):
    """等待服务启动"""
    print(f"⏳ 等待服务启动 (端口 {port})...")
    
    for i in range(max_wait):
        try:
            result = run_cmd(["curl", "-s", f"http://localhost:{port}/health"], timeout=3)
            if result.returncode == 0:
                print(f"✅ 服务已启动 (端口 {port}，等待了 {i+1}s)")
                return True
        except Exception:
            pass
        
        if i < 5 or (i + 1) % 5 == 0:
            print(f"  等待中... ({i+1}s/{max_wait}s)")
        time.sleep(1)
    
    print(f"❌ 服务启动超时 (端口 {port})")
    return False

def check_instance_health(port, instance_name, max_wait=45):
    """检查单个实例健康状态"""
    print(f"🏥 检查实例 {instance_name} (端口 {port})...")
    
    for i in range(max_wait):
        try:
            result = run_cmd([
                "curl", "-s", f"http://localhost:{port}/get_model_info"
            ], timeout=3)
            if result.returncode == 0 and "model_path" in result.stdout:
                print(f"  ✅ 实例 {instance_name} 健康 (耗时 {i+1}s)")
                return True
        except Exception:
            pass
        
        if (i + 1) % 5 == 0:
            print(f"  ⏳ 实例 {instance_name} 启动中... ({i+1}s/{max_wait}s)")
        time.sleep(1)
    
    print(f"  ❌ 实例 {instance_name} 启动超时 ({max_wait}s)")
    return False

def main():
    parser = argparse.ArgumentParser(description="直接进程方式的自动化benchmark")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--task-name", required=True, help="任务名称")
    parser.add_argument("--request-rate", type=float, default=10, 
                        help="每秒请求数 (默认: 10)")
    parser.add_argument("--num-prompts", type=int, default=1000, 
                        help="总请求数 (默认: 1000)")
    parser.add_argument("--max-concurrency", type=int, 
                        help="最大并发数 (可选)")
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    router_port = config.get("router", {}).get("port", 8080)
    instances = config.get("instances", [])
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{timestamp}_{args.task_name}"
    log_dir = Path("/workspace/kvcached/yangmin_logs") / log_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 开始直接进程方式的benchmark: {args.task_name}")
    print(f"📅 时间戳: {timestamp}")
    print(f"⚙️  配置文件: {config_path}")
    print(f"📁 日志目录: {log_dir}")
    print("-" * 60)
    
    # 保存配置文件副本
    config_copy = log_dir / config_path.name
    shutil.copy2(config_path, config_copy)
    print(f"✅ 配置文件已保存到: {config_copy}")
    
    # 活跃进程列表
    active_processes = []
    
    try:
        # 1. 清理端口进程
        instance_ports = []
        for instance in instances:
            for arg in instance.get("engine_args", []):
                if arg.startswith("--port="):
                    instance_ports.append(int(arg.split("=")[1]))
        
        ports_to_clear = [router_port] + instance_ports
        kill_port_processes(ports_to_clear)
        
        # 2. 准备环境变量
        env = os.environ.copy()
        venv_bin = "/workspace/kvcached/engine_integration/sglang-v0.4.9/.venv/bin"
        venv_dir = "/workspace/kvcached/engine_integration/sglang-v0.4.9/.venv"
        env['PATH'] = f"{venv_bin}:{env['PATH']}"
        env['VIRTUAL_ENV'] = venv_dir
        env['CUDA_VISIBLE_DEVICES'] = "1"  # 只使用GPU1
        
        cwd = "/workspace/kvcached/controller"
        
        # 3. 逐个启动模型实例（避免并发冲突）
        for i, instance in enumerate(instances):
            instance_name = instance.get("name", "unknown")
            model_name = instance.get("model", "")
            engine_args = instance.get("engine_args", [])
            
            # 构建启动命令
            cmd_parts = ["python3", "-m", "sglang.launch_server"]
            cmd_parts.extend(["--model-path", model_name])
            cmd_parts.extend(engine_args)
            
            # 创建进程记录器
            log_file = log_dir / f"{instance_name}_server.log"
            process_logger = ProcessLogger(f"Instance-{instance_name}", log_file)
            
            # 启动进程
            if process_logger.start_process(cmd_parts, cwd=cwd, env=env):
                active_processes.append(process_logger)
                
                # 如果是第一个实例，等待它完全启动后再启动第二个
                if i == 0:
                    print(f"⏳ 等待第一个实例 {instance_name} 完全启动...")
                    port = None
                    for arg in instance.get("engine_args", []):
                        if arg.startswith("--port="):
                            port = int(arg.split("=")[1])
                            break
                    
                    if port and check_instance_health(port, instance_name, max_wait=60):
                        print(f"✅ 第一个实例 {instance_name} 启动完成，现在启动第二个实例")
                        time.sleep(2)  # 额外缓冲时间
                    else:
                        print(f"❌ 第一个实例 {instance_name} 启动失败")
                        return False
                else:
                    time.sleep(2)  # 其他实例的基本等待时间
            else:
                print(f"❌ 启动实例 {instance_name} 失败")
                return False
        
        # 4. 启动路由器
        router_cmd = ["python3", "frontend.py", "--config", config_path.name]
        router_logger = ProcessLogger("Router", log_dir / "router.log")
        if router_logger.start_process(router_cmd, cwd=cwd, env=env):
            active_processes.append(router_logger)
            time.sleep(3)  # 给路由器时间启动
        else:
            print("❌ 启动路由器失败")
            return False
        
        # 5. 等待路由器启动
        if not wait_for_service(router_port):
            print("❌ 路由器启动失败")
            return False
        
        # 6. 检查所有实例健康状态
        print("🏥 检查所有实例健康状态...")
        all_healthy = True
        for instance in instances:
            instance_name = instance.get("name", "unknown")
            port = None
            for arg in instance.get("engine_args", []):
                if arg.startswith("--port="):
                    port = int(arg.split("=")[1])
                    break
            
            if port and not check_instance_health(port, instance_name):
                all_healthy = False
                break
        
        if not all_healthy:
            print("❌ 实例健康检查失败")
            return False
        
        print("✅ 所有服务启动成功，开始benchmark测试...")
        
        # 7. 运行benchmark (使用直接调用方式，不使用tmux)
        print("🚀 开始运行benchmark测试...")
        print(f"📊 请求速率: {args.request_rate} req/s, 总请求数: {args.num_prompts}")
        
        benchmark_cmd = [
            "python3", "benchmark_direct.py", 
            "--config", config_path.name,
            "--log-dir", str(log_dir),
            "--request-rate", str(args.request_rate),
            "--num-prompts", str(args.num_prompts)
        ]
        
        if args.max_concurrency:
            benchmark_cmd.extend(["--max-concurrency", str(args.max_concurrency)])
        
        try:
            result = subprocess.run(
                benchmark_cmd, 
                cwd=cwd, 
                env=env, 
                capture_output=True, 
                text=True,
                timeout=600  # 10分钟超时
            )
            
            # 保存benchmark执行日志
            with open(log_dir / "benchmark_execution.log", 'w') as f:
                f.write("=== Benchmark 执行日志 ===\n")
                f.write(f"命令: {' '.join(benchmark_cmd)}\n")
                f.write(f"返回代码: {result.returncode}\n")
                f.write("=" * 50 + "\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            if result.returncode == 0:
                print("✅ Benchmark完成")
                print(result.stdout)
            else:
                print("❌ Benchmark失败")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Benchmark超时")
            return False
        except Exception as e:
            print(f"❌ 运行benchmark时出错: {e}")
            return False
        
        # 8. 保存最终状态
        summary = {
            "task_name": args.task_name,
            "timestamp": timestamp,
            "config_file": str(config_path),
            "log_directory": str(log_dir),
            "router_port": router_port,
            "instances": [
                {
                    "name": inst.get("name"),
                    "model": inst.get("model"),
                    "port": next(
                        (int(arg.split("=")[1]) for arg in inst.get("engine_args", []) 
                         if arg.startswith("--port=")), None
                    )
                }
                for inst in instances
            ],
            "status": "completed"
        }
        
        with open(log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 完整日志保存在: {log_dir}")
        print("🎉 自动化测试完成！")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号，正在清理...")
        return False
    finally:
        # 清理所有进程
        print("🧹 清理所有进程...")
        for process_logger in active_processes:
            process_logger.stop()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 