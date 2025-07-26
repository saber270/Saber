#!/usr/bin/env python3
import subprocess
import time
import os
import signal
import sys
import requests
import json
import psutil  # Requires installation: pip install psutil
from datetime import datetime

# Set output directory
log_dir = "experiment_logs"
os.makedirs(log_dir, exist_ok=True)

# Create main log file
main_log_file = f"{log_dir}/main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    """Log message to file and console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    
    # Write to log file
    with open(main_log_file, 'a') as f:
        f.write(log_message + "\n")
        f.flush()  # Ensure immediate writing
    
    # Output to console
    print(log_message)
    sys.stdout.flush()  # Ensure immediate display

def check_server_status(timeout=300):
    """Check server status until successful or timeout"""
    url = 'http://localhost:8000/generate'
    headers = {'Content-Type': 'application/json'}
    test_data = {
        'prompt': "hello world",
        'max_tokens': 10,
        'min_tokens': 1,
        'temperature': 0.1,
        'stream': False
    }
    
    log("Checking server status...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.post(url, headers=headers, json=test_data, timeout=10)
            if response.status_code == 200:
                log(f"Server is ready! Response time: {response.elapsed.total_seconds():.2f}s")
                return True
        except requests.exceptions.RequestException:
            pass
        
        # Check every 10 seconds
        time.sleep(10)
        log(f"Server still loading... Waited {int(time.time() - start_time)} seconds")
    
    log(f"Server startup timeout ({timeout} seconds)")
    return False

def start_vllm_server(max_num_seqs):
    """Start VLLM server"""
    log(f"Starting VLLM server with max_num_seqs={max_num_seqs}...")
    
    # Create server log file
    server_log = f"{log_dir}/server_max_num_seqs_{max_num_seqs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Build server startup command
    cmd = [
        "python", "-m", "vllm.entrypoints.api_server",
        "--model", "/home/model_registry_storage/Qwen2.5-Coder-3B",
        "--max-model-len=2048",
        "--dtype=float16",
        "--gpu-memory-utilization=0.8",
        "--trust-remote-code",
        f"--max-num-seqs={max_num_seqs}"
    ]
    
    cmd_str = " ".join(cmd)
    log(f"Executing command: {cmd_str}")
    
    # Start server process and redirect output to log file
    with open(server_log, 'w') as f:
        server_process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    
    log(f"VLLM server process started, PID: {server_process.pid}")
    log(f"Server log file: {server_log}")
    
    # Check if server started successfully
    if check_server_status(timeout=300):
        return server_process, server_log
    else:
        log("Server failed to start, attempting to terminate process...")
        try:
            server_process.terminate()
            server_process.wait(timeout=30)
        except:
            try:
                server_process.kill()
            except:
                pass
        return None, server_log

def kill_process_and_children(pid):
    """Terminate process and all its child processes"""
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.terminate()
            except:
                try:
                    child.kill()
                except:
                    pass
        
        # Terminate parent process
        try:
            parent.terminate()
            parent.wait(timeout=5)
        except:
            try:
                parent.kill()
            except:
                pass
        
        return True
    except:
        return False

def run_experiment(max_num_seqs, script_name, timeout=3600):  # Default 1 hour timeout
    """Run experiment script with timeout protection"""
    log(f"Running experiment script with concurrency_config={max_num_seqs}...")
    
    # Check if script exists
    if not os.path.exists(script_name):
        log(f"Script file not found: {script_name}")
        return False
    
    # Create experiment log file
    experiment_log = f"{log_dir}/experiment_max_num_seqs_{max_num_seqs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Build command
    cmd = f"python {script_name} {max_num_seqs}"
    log(f"Executing command: {cmd}")
    
    # Run experiment and capture output
    with open(experiment_log, 'w') as f:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    
    pid = process.pid
    log(f"Experiment process started, PID: {pid}")
    log(f"Experiment log file: {experiment_log}")
    log(f"Timeout set to: {timeout} seconds")
    
    # Record start time
    start_time = time.time()
    
    # Check process status every 10 seconds
    while process.poll() is None:
        # Check for timeout
        current_time = time.time()
        if current_time - start_time > timeout:
            log(f"Experiment timeout ({timeout} seconds)! Force terminating...")
            kill_process_and_children(pid)
            log(f"Terminated timeout experiment (PID: {pid})")
            return False
        
        # Check experiment log file size to determine progress
        try:
            log_size = os.path.getsize(experiment_log)
            log(f"Experiment running... Elapsed time: {int(current_time - start_time)} seconds, Log size: {log_size} bytes")
        except:
            log(f"Experiment running... Elapsed time: {int(current_time - start_time)} seconds")
        
        # Wait before checking again
        time.sleep(60)  # Report status every minute
    
    # Get return code
    returncode = process.returncode
    
    if returncode == 0:
        log("Experiment completed successfully")
        return True
    else:
        log(f"Experiment failed with return code: {returncode}")
        return False

def check_running_processes():
    """Check and clean up potentially running related processes"""
    log("Checking for potentially running related processes in the system...")
    
    # Check vllm processes
    vllm_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline'] or [])
            if 'vllm.entrypoints.api_server' in cmdline:
                vllm_running = True
                log(f"Found running VLLM server process, PID: {proc.info['pid']}")
                log(f"Command line: {cmdline}")
                
                # Ask whether to terminate
                log("Attempting to terminate this process...")
                kill_process_and_children(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if not vllm_running:
        log("No running VLLM server processes found")
    
    # Check for zombie Python processes
    zombie_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'status']):
        try:
            if proc.info['name'] == 'python' and proc.info['status'] == psutil.STATUS_ZOMBIE:
                zombie_count += 1
                log(f"Found zombie Python process, PID: {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if zombie_count > 0:
        log(f"Found {zombie_count} zombie Python process(es)")
    else:
        log("No zombie Python processes found")

def stop_server(server_process):
    """Stop VLLM server"""
    if server_process and server_process.poll() is None:
        log(f"Stopping VLLM server (PID: {server_process.pid})...")
        
        kill_process_and_children(server_process.pid)
        log("VLLM server terminated")

def main():
    """Main function"""
    log("===== Experiment Automation Script Started =====")
    
    # Set max_num_seqs values to test
    max_num_seqs_list = [10,20,30,40,50,60,70,80,90,100]
    
    # Set experiment script name
    script_name = "baseline_client.py"
    
    # First check and clean up potentially running processes
    check_running_processes()
    
    log(f"Will test the following max_num_seqs values: {max_num_seqs_list}")
    log(f"Will use experiment script: {script_name}")
    
    # Store completed configurations
    completed_configs = []
    
    # Run experiment for each max_num_seqs value
    for max_num_seqs in max_num_seqs_list:
        log(f"===== Starting test with max_num_seqs={max_num_seqs} =====")
        
        # Start server
        server_process, server_log = start_vllm_server(max_num_seqs)
        
        if server_process:
            try:
                # Run experiment with maximum runtime of 3 hours
                success = run_experiment(max_num_seqs, script_name, timeout=108000)
                if success:
                    completed_configs.append(max_num_seqs)
            finally:
                # Stop server regardless of experiment success
                stop_server(server_process)
                
                # Check and clean up possible zombie processes
                check_running_processes()
        
        log(f"===== Completed test with max_num_seqs={max_num_seqs} =====")
        
        # Wait between different configurations
        if max_num_seqs != max_num_seqs_list[-1]:
            log("Waiting 30 seconds before starting next configuration...")
            time.sleep(30)
    
    # Summarize results
    log("===== Experiment Summary =====")
    log(f"Successfully completed configurations: {completed_configs}")
    
    if len(completed_configs) != len(max_num_seqs_list):
        failed_configs = [seq for seq in max_num_seqs_list if seq not in completed_configs]
        log(f"Failed configurations: {failed_configs}")
    
    log("===== Experiment Automation Script Ended =====")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nExperiment interrupted by user")
    except Exception as e:
        log(f"Error occurred: {e}")
        import traceback
        log(traceback.format_exc())