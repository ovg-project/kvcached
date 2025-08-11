#!/usr/bin/env python3
"""
独立的批量benchmark脚本
直接内嵌所有功能，不调用其他脚本，支持实时输出
"""

import argparse
import json
import subprocess
import time
import sys
import yaml
import os
import threading
import requests
from datetime import datetime
from pathlib import Path
import psutil

def wait_for_health_check(port, timeout=120):
    """等待实例健康检查通过"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/get_model_info", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def load_experiments_config(config_file):
    """加载实验配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_sglang_config(config_file):
    """加载SGLang配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def kill_port_processes(ports):
    """杀死占用指定端口的进程"""
    print("🔪 清理端口占用的进程...")
    
    for port in ports:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], 
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        print(f"  杀死端口 {port} 的进程 PID: {pid}")
                        subprocess.run(["kill", "-9", pid])
            else:
                print(f"  端口 {port} 未被占用")
        except Exception as e:
            print(f"  清理端口 {port} 时出错: {e}")

def start_model_instances(config, log_dir, env_vars):
    """启动模型实例"""
    print("🚀 启动模型实例...")
    
    instances = config.get("instances", [])
    processes = []
    
    for i, instance in enumerate(instances):
        instance_name = instance.get("name", "unknown")
        model_name = instance.get("model", "")
        engine_args = instance.get("engine_args", [])
        
        print(f"  启动实例: {instance_name}")
        
        # 构建启动命令
        cmd = ["python3", "-m", "sglang.launch_server", "--model-path", model_name]
        cmd.extend(engine_args)
        
        # 准备环境变量
        instance_env = env_vars.copy()
        
        # 添加实例特定的环境变量
        for env_item in instance.get("kvcached_env", []):
            if "=" in env_item:
                key, value = env_item.split("=", 1)
                instance_env[key] = value
        
        for env_item in instance.get("engine_env", []):
            if "=" in env_item:
                key, value = env_item.split("=", 1)
                instance_env[key] = value
        
        # 启动进程
        log_file = log_dir / f"{instance_name}_server.log"
        with open(log_file, 'w') as f:
            f.write(f"=== {instance_name} 服务器日志 ===\n")
            f.write(f"启动时间: {datetime.now()}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write("=" * 50 + "\n\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd="/workspace/kvcached/controller",
                env=instance_env,
                text=True
            )
            
        processes.append((instance_name, process, log_file))
        print(f"    PID: {process.pid}")
        
        # 如果是第一个实例，等待其健康检查通过再启动下一个
        if i == 0 and len(instances) > 1:
            port = None
            for arg in engine_args:
                if arg.startswith("--port="):
                    port = int(arg.split("=")[1])
                    break
            
            if port:
                print(f"    等待实例 {instance_name} (端口:{port}) 健康检查...")
                if wait_for_health_check(port, timeout=120):
                    print(f"    ✅ 实例 {instance_name} 已就绪，继续启动下一个实例")
                else:
                    print(f"    ❌ 实例 {instance_name} 健康检查失败")
                    break
        elif i > 0:
            time.sleep(5)  # 非第一个实例间的小延迟
    
    return processes

def start_router(config, log_dir, env_vars):
    """启动路由器"""
    router_config = config.get("router", {})
    if not router_config.get("enable_router", True):
        return None
    
    print("🚀 启动路由器...")
    
    config_file = "temp_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    cmd = ["python3", "frontend.py", "--config", config_file]
    
    log_file = log_dir / "router.log"
    with open(log_file, 'w') as f:
        f.write("=== 路由器日志 ===\n")
        f.write(f"启动时间: {datetime.now()}\n")
        f.write(f"命令: {' '.join(cmd)}\n")
        f.write("=" * 50 + "\n\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd="/workspace/kvcached/controller",
            env=env_vars,
            text=True
        )
    
    print(f"  路由器PID: {process.pid}")
    return process, log_file

def wait_for_service_health(port, instance_name, max_wait=60):
    """等待服务健康"""
    print(f"🏥 等待实例 {instance_name} 健康 (端口 {port})...")
    
    for i in range(max_wait):
        try:
            result = subprocess.run([
                "curl", "-s", f"http://localhost:{port}/get_model_info"
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and "model_path" in result.stdout:
                print(f"  ✅ 实例 {instance_name} 健康 (耗时 {i+1}s)")
                return True
        except Exception:
            pass
        
        if (i + 1) % 10 == 0:
            print(f"  ⏳ 等待中... ({i+1}s/{max_wait}s)")
        time.sleep(1)
    
    print(f"  ❌ 实例 {instance_name} 健康检查超时")
    return False

def run_benchmark_for_instance(instance_name, model_name, port, log_file, 
                               request_rate, num_prompts, max_concurrency=None):
    """为单个实例运行benchmark"""
    print(f"\n🧪 开始实例 {instance_name} 的benchmark...")
    print(f"  端口: {port}, 速率: {request_rate} req/s, 请求数: {num_prompts}")
    
    # 准备环境变量
    env = os.environ.copy()
    venv_bin = "/workspace/kvcached/engine_integration/sglang-v0.4.9/.venv/bin"
    env['PATH'] = f"{venv_bin}:{env['PATH']}"
    env['VIRTUAL_ENV'] = "/workspace/kvcached/engine_integration/sglang-v0.4.9/.venv"
    
    # 构建benchmark命令
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang-oai",
        "--model", model_name,
        "--dataset-name", "sharegpt",
        "--dataset-path", "/workspace/kvcached/engine_integration/benchmark/ShareGPT_V3_unfiltered_cleaned_split.json",
        "--request-rate", str(request_rate),
        "--num-prompts", str(num_prompts),
        "--port", str(port),
        "--host", "localhost"
    ]
    
    if max_concurrency:
        cmd.extend(["--max-concurrency", str(max_concurrency)])
    
    print(f"  💻 命令: {' '.join(cmd)}")
    
    # 运行benchmark，实时显示输出
    with open(log_file, 'w') as f:
        f.write(f"=== {instance_name} Benchmark 日志 ===\n")
        f.write(f"实例: {instance_name}\n")
        f.write(f"端口: {port}\n")
        f.write(f"请求速率: {request_rate} req/s\n")
        f.write(f"总请求数: {num_prompts}\n")
        f.write(f"命令: {' '.join(cmd)}\n")
        f.write("=" * 50 + "\n\n")
        f.flush()
        
        # 启动进程，并实时显示输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd="/workspace/kvcached/controller",
            env=env,
            text=True,
            bufsize=1
        )
        
        # 实时读取和显示输出
        while True:
            line = process.stdout.readline()
            if line:
                # 同时写入文件和显示到控制台
                f.write(line)
                f.flush()
                print(f"[{instance_name}] {line.rstrip()}")
            elif process.poll() is not None:
                break
        
        # 获取剩余输出
        remaining = process.stdout.read()
        if remaining:
            f.write(remaining)
            print(f"[{instance_name}] {remaining.rstrip()}")
        
        return_code = process.wait()
        
        f.write(f"\n=== Benchmark 完成 ===\n")
        f.write(f"返回代码: {return_code}\n")
    
    success = return_code == 0
    if success:
        print(f"✅ 实例 {instance_name} benchmark完成")
    else:
        print(f"❌ 实例 {instance_name} benchmark失败 (代码: {return_code})")
    
    return success

def run_parallel_benchmarks(instances, request_rate, num_prompts, log_dir, max_concurrency=None):
    """并行运行多个实例的benchmark"""
    print(f"\n🔄 开始并行benchmark测试...")
    print(f"📊 将同时测试 {len(instances)} 个实例")
    
    threads = []
    results = {}
    
    def benchmark_worker(instance_name, model_name, port, log_file):
        success = run_benchmark_for_instance(
            instance_name, model_name, port, log_file,
            request_rate, num_prompts, max_concurrency
        )
        results[instance_name] = success
    
    # 启动所有benchmark线程
    for instance in instances:
        instance_name = instance.get("name", "unknown")
        model_name = instance.get("model", "")
        
        # 从engine_args中提取端口
        port = None
        for arg in instance.get("engine_args", []):
            if arg.startswith("--port="):
                port = int(arg.split("=")[1])
                break
        
        if not port:
            print(f"❌ 无法获取实例 {instance_name} 的端口")
            results[instance_name] = False
            continue
        
        log_file = log_dir / f"{instance_name}_benchmark.log"
        
        thread = threading.Thread(
            target=benchmark_worker,
            args=(instance_name, model_name, port, log_file)
        )
        thread.start()
        threads.append(thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    return results

def kill_gpu_processes_by_nvidia_smi():
    """通过nvidia-smi查询并终止使用GPU的进程"""
    print("🔍 通过 nvidia-smi 查询使用 GPU 的进程...")
    try:
        # 获取 pid, process_name, used_gpu_memory 等信息
        output = subprocess.check_output([
            "nvidia-smi", 
            "--query-compute-apps=pid,process_name,used_gpu_memory", 
            "--format=csv,noheader"
        ]).decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        print(f"调用 nvidia-smi 出错: {e}")
        return
    except FileNotFoundError:
        print("nvidia-smi 命令未找到，跳过GPU进程清理")
        return

    lines = output.split("\n")
    if not lines or not lines[0]:
        print("✅ 当前没有进程在使用 GPU")
        return

    # 循环解析并 kill
    killed_count = 0
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        pid_str = parts[0].strip()
        proc_name = parts[1].strip()
        mem_usage = parts[2].strip()
        if not pid_str.isdigit():
            continue

        pid = int(pid_str)
        print(f"🎯 找到 GPU 占用进程: PID={pid}, Name={proc_name}, Memory={mem_usage} MiB")
        try:
            p = psutil.Process(pid)
            # 先尝试正常终止
            p.terminate()
            try:
                p.wait(timeout=5)  # 等待 5 秒
                print(f"✅ 已正常终止进程 {pid}")
                killed_count += 1
            except psutil.TimeoutExpired:
                print(f"⚡ 进程 {pid} 未在规定时间内退出，尝试强制 kill...")
                p.kill()
                print(f"💀 已强制终止进程 {pid}")
                killed_count += 1
        except psutil.NoSuchProcess:
            print(f"⚠️  进程 {pid} 已不存在")
        except psutil.AccessDenied:
            print(f"🚫 无权限终止进程 {pid}")
        except Exception as e:
            print(f"❌ 终止进程 {pid} 出错: {e}")

    if killed_count > 0:
        print(f"🧹 GPU 进程清理完毕，共终止了 {killed_count} 个进程")
    else:
        print("✅ GPU 进程清理完毕")


def cleanup_processes(processes):
    """清理进程"""
    print("\n🧹 清理所有进程...")
    
    if isinstance(processes, list):
        for name, process, log_file in processes:
            if process and process.poll() is None:
                print(f"  停止 {name} (PID: {process.pid})")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    elif processes:
        process, log_file = processes
        if process and process.poll() is None:
            print(f"  停止路由器 (PID: {process.pid})")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

def run_single_experiment(experiment, global_settings, base_timestamp):
    """运行单个实验"""
    exp_name = experiment["name"]
    config_file = experiment["config_file"]
    request_rate = experiment["request_rate"]
    num_prompts = experiment["num_prompts"]
    max_concurrency = experiment.get("max_concurrency")
    kvcached_enabled = experiment.get("kvcached_enabled", True)
    
    print("=" * 80)
    print(f"🧪 开始实验: {exp_name}")
    print(f"📋 描述: {experiment['description']}")
    print(f"⚙️  配置文件: {config_file}")
    print(f"📊 请求速率: {request_rate} req/s")
    print(f"📝 总请求数: {num_prompts}")
    print(f"🔧 KVCached: {'启用' if kvcached_enabled else '禁用'}")
    print("=" * 80)
    
    # 创建日志目录
    task_name = f"{base_timestamp}_{exp_name}"
    log_dir = Path(f"/workspace/kvcached/yangmin_logs/{task_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    model_processes = []
    router_process = None
    
    try:
        # 加载配置
        config = load_sglang_config(config_file)
        instances = config.get("instances", [])
        
        # 准备环境变量
        env_vars = os.environ.copy()
        venv_bin = "/workspace/kvcached/engine_integration/sglang-v0.4.9/.venv/bin"
        env_vars['PATH'] = f"{venv_bin}:{env_vars['PATH']}"
        env_vars['VIRTUAL_ENV'] = "/workspace/kvcached/engine_integration/sglang-v0.4.9/.venv"
        env_vars['CUDA_VISIBLE_DEVICES'] = "0"
        
        # 1. 清理端口
        ports_to_clear = [8080]  # 路由器端口
        for instance in instances:
            for arg in instance.get("engine_args", []):
                if arg.startswith("--port="):
                    ports_to_clear.append(int(arg.split("=")[1]))
        
        kill_port_processes(ports_to_clear)
        
        # 2. 启动模型实例
        model_processes = start_model_instances(config, log_dir, env_vars)
        
        # 3. 等待实例健康
        all_healthy = True
        for instance in instances:
            instance_name = instance.get("name", "unknown")
            port = None
            for arg in instance.get("engine_args", []):
                if arg.startswith("--port="):
                    port = int(arg.split("=")[1])
                    break
            
            if port and not wait_for_service_health(port, instance_name):
                all_healthy = False
                break
        
        if not all_healthy:
            print("❌ 实例健康检查失败")
            return {"name": exp_name, "success": False, "error": "health_check_failed"}
        
        # 4. 运行并行benchmark
        benchmark_results = run_parallel_benchmarks(
            instances, request_rate, num_prompts, log_dir, max_concurrency
        )
        
        # 计算结果
        successful_benchmarks = sum(1 for success in benchmark_results.values() if success)
        total_benchmarks = len(benchmark_results)
        
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        
        overall_success = successful_benchmarks == total_benchmarks
        
        result = {
            "name": exp_name,
            "success": overall_success,
            "duration_minutes": round(duration_minutes, 2),
            "task_name": task_name,
            "benchmark_results": benchmark_results,
            "successful_benchmarks": successful_benchmarks,
            "total_benchmarks": total_benchmarks,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat()
        }
        
        if overall_success:
            print(f"\n✅ 实验 {exp_name} 完成 ({successful_benchmarks}/{total_benchmarks} 成功, 耗时: {duration_minutes:.1f} 分钟)")
        else:
            print(f"\n❌ 实验 {exp_name} 部分失败 ({successful_benchmarks}/{total_benchmarks} 成功, 耗时: {duration_minutes:.1f} 分钟)")
        
        return result
        
    except Exception as e:
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        print(f"\n💥 实验 {exp_name} 出错: {e}")
        
        return {
            "name": exp_name,
            "success": False,
            "duration_minutes": round(duration_minutes, 2),
            "task_name": task_name,
            "error": str(e),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat()
        }
    
    finally:
        # 清理进程
        cleanup_processes(model_processes)
        if router_process:
            cleanup_processes(router_process)
        
        # 清理GPU进程
        kill_gpu_processes_by_nvidia_smi()
        
        # 等待一段时间确保资源被释放
        time.sleep(3)

def main():
    parser = argparse.ArgumentParser(description="独立批量benchmark实验")
    parser.add_argument("--config", default="benchmark_experiments.json",
                        help="实验配置文件")
    parser.add_argument("--start-from", type=int, default=0,
                        help="从第几个实验开始")
    parser.add_argument("--max-experiments", type=int,
                        help="最多运行几个实验")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示实验列表")
    args = parser.parse_args()
    
    # 加载配置
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return 1
    
    experiments_config = load_experiments_config(config_file)
    experiments = experiments_config["experiments"]
    global_settings = experiments_config.get("global_settings", {})
    
    # 处理实验范围
    start_idx = args.start_from
    if args.max_experiments:
        end_idx = min(start_idx + args.max_experiments, len(experiments))
    else:
        end_idx = len(experiments)
    
    selected_experiments = experiments[start_idx:end_idx]
    
    # 创建批次时间戳
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🚀 独立批量Benchmark实验开始")
    print(f"📅 批次时间戳: {base_timestamp}")
    print(f"📁 配置文件: {config_file}")
    print(f"🧪 总实验数: {len(experiments)}")
    print(f"📊 将执行: 第{start_idx}到{end_idx-1}个实验 (共{len(selected_experiments)}个)")
    print("=" * 80)
    
    if args.dry_run:
        print("🔍 Dry-run模式，将要执行的实验:")
        for i, exp in enumerate(selected_experiments, start_idx):
            print(f"  {i}: {exp['name']} - {exp['description']}")
        return 0
    
    # 执行实验
    results = []
    total_start_time = time.time()
    
    for i, experiment in enumerate(selected_experiments, start_idx):
        print(f"\n📊 进度: {i-start_idx+1}/{len(selected_experiments)}")
        
        result = run_single_experiment(experiment, global_settings, base_timestamp)
        results.append(result)
        
        # 实验间清理
        if i < len(selected_experiments) - 1:
            print("🧹 实验间清理...")
            time.sleep(5)
    
    total_end_time = time.time()
    total_duration = (total_end_time - total_start_time) / 60
    
    # 生成汇总
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print("\n" + "=" * 80)
    print("🎉 批量实验完成！")
    print(f"📊 总结果: {successful}/{len(results)} 成功, {failed}/{len(results)} 失败")
    print(f"⏱️  总耗时: {total_duration:.1f} 分钟")
    
    # 保存汇总报告
    report_dir = Path(f"/workspace/kvcached/yangmin_logs/{base_timestamp}_batch_summary")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_dir / "batch_report.json", 'w', encoding='utf-8') as f:
        json.dump({
            "batch_timestamp": base_timestamp,
            "total_experiments": len(results),
            "successful_experiments": successful,
            "failed_experiments": failed,
            "total_duration_minutes": total_duration,
            "experiments": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📊 详细报告已保存到: {report_dir}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 