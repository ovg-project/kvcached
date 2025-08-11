#!/usr/bin/env python3
"""
简化的启动脚本，不依赖tmux，直接在后台运行SGLang实例
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import yaml
import os
import signal

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_sglang_command(instance_config):
    """构建SGLang启动命令"""
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", instance_config["model"]
    ]
    
    # 添加engine_args
    cmd.extend(instance_config.get("engine_args", []))
    
    return cmd

def build_env(instance_config, global_env):
    """构建环境变量"""
    env = os.environ.copy()
    
    # 添加全局环境变量
    for key, value in global_env.items():
        env[key] = str(value)
    
    # 添加kvcached环境变量
    for env_var in instance_config.get("kvcached_env", []):
        key, value = env_var.split("=", 1)
        env[key] = value
    
    # 添加engine环境变量
    for env_var in instance_config.get("engine_env", []):
        key, value = env_var.split("=", 1)
        env[key] = value
    
    return env

def main():
    parser = argparse.ArgumentParser(description="直接启动SGLang实例（无tmux）")
    parser.add_argument("--config", type=Path, default="dual-sglang-config.yaml",
                       help="配置文件路径")
    parser.add_argument("--instance", type=str, help="启动特定实例（可选）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 提取全局环境变量
    global_kvcached_env = {}
    if "kvcached" in config:
        for key, value in config["kvcached"].items():
            global_kvcached_env[f"KVCACHED_{key.upper()}"] = str(value)
    
    processes = []
    
    try:
        for instance in config["instances"]:
            instance_name = instance["name"]
            
            # 如果指定了特定实例，只启动该实例
            if args.instance and instance_name != args.instance:
                continue
            
            print(f"启动实例: {instance_name}")
            
            # 构建命令和环境
            cmd = build_sglang_command(instance)
            env = build_env(instance, global_kvcached_env)
            
            print(f"命令: {' '.join(cmd)}")
            print(f"端口: {[arg for arg in cmd if arg.startswith('--port') or (cmd.index(arg) > 0 and cmd[cmd.index(arg)-1] == '--port')]}")
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            processes.append((instance_name, process))
            print(f"✓ {instance_name} 已启动 (PID: {process.pid})")
            
            # 等待一下让进程启动
            time.sleep(2)
        
        print(f"\n所有实例已启动。按 Ctrl+C 停止所有实例。")
        print("实例状态:")
        for name, process in processes:
            if process.poll() is None:
                print(f"  ✓ {name}: 运行中 (PID: {process.pid})")
            else:
                print(f"  ✗ {name}: 已退出")
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
                # 检查进程状态
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"警告: {name} 进程已退出")
        except KeyboardInterrupt:
            print("\n正在停止所有实例...")
            
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 清理所有进程
        for name, process in processes:
            try:
                if process.poll() is None:
                    print(f"停止 {name}...")
                    process.terminate()
                    process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"强制终止 {name}...")
                process.kill()
            except Exception as e:
                print(f"停止 {name} 时出错: {e}")

if __name__ == "__main__":
    main() 