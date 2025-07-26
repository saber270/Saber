import requests
import threading
import time
import csv
import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import sys

# Config Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the base URL and headers
url = 'http://localhost:8000/generate'
headers = {'Content-Type': 'application/json'}

Chat_QnAinput = "In a sprawling valley nestled between jagged mountain peaks, the town of Eldervale lay cloaked in the shimmering veil of twilight, its cobblestone streets glowing faintly under the amber light of antiquated lanterns. The air carried a subtle aroma of woodsmoke and wildflowers, mingling with the occasional, tantalizing scent of baked goods wafting from the hearths of homes that had stood for centuries. Eldervale was a place of old secrets and whispered tales, where every stone in the ancient walls seemed to hum with the resonance of bygone eras. At the heart of the town was the Grand Clocktower, its monumental gears visible through ornate glass panels, ticking steadily as though measuring the heartbeat of the world itself. Travelers often paused in their journeys to marvel at its intricate design, a masterpiece of engineering left behind by a civilization long forgotten, . "

codegen_input = 4*"Develop a Python-based Library Management System with three key features: managing books, members, and borrowing activities. Implement the system using object-oriented principles with Book, Member, and Library classes. The Book class should store details like title, author, ISBN, and availability. The Member class should handle member information, including name, ID, contact, and borrowed books. The Library class integrates the functionalities, managing books, members, and issuing or returning books. Use JSON files for persistent storage of data and ensure proper error handling. The system must adhere to PEP8 coding standards and be modular and maintainable"

Code_trans_input = 3*"Here's rewritten to meet the 1403-character count requirement: Create a Library Management System in Python that automates core library functions: managing books, members, and borrowing processes. The system should include book management, allowing adding, updating, deleting, and searching for books using attributes like title, author, ISBN, and availability status. Member management should support adding, updating, and removing members while tracking their borrowed books. The borrowing system should handle issuing and returning books, updating availability, and calculating penalties for late returns. Use object-oriented programming principles to design the system with three classes: Book, Member, and Library. The Book class represents individual books with properties such as title, author, ISBN, and availability. The Member class stores member details, including name, ID, contact, and borrowed books. The Library class integrates all functionalities, managing books, members, and borrowing activities. class Book: def __init__(self, title, author, isbn): self.title = title self.author = author self.isbn = isbn self.available = True class Member: def __init__(self, name, member_id, contact): self.name = name self.member_id = member_id self.contact = contact self.borrowed_books = [] class Library: def __init__(self): self.books = [] self.members = []"

Code_summary_input = "protected final void bindIndexed(ConfigurationPropertyName name, Bindable<?> target, AggregateElementBinder elementBinder, ResolvableType aggregateType, ResolvableType elementType, IndexedCollectionSupplier result) { for (ConfigurationPropertySource source : getContext().getSources()) { bindIndexed(source, name, target, elementBinder, result, aggregateType, elementType); if (result.wasSupplied() && result.get() != null) { return; } } }"

# Input data templates
data_templates = {
    'long_input_long_output': {
        'prompt': Code_trans_input,
        'max_tokens': 617,
        'temperature': 0.1,
        'stream':True
    },
    'short_input_short_output': {
        'prompt': Chat_QnAinput,
        'max_tokens': 51,
        'temperature': 0.1,
        'stream':True
    },
    'short_input_long_output': {
        'prompt': codegen_input,
        'max_tokens': 387,
        'temperature': 0.1,
        'stream':True
    },
    'long_input_short_output': {
        'prompt':Code_summary_input,
        'max_tokens': 48,
        'temperature': 0.1,
        'stream':True
    }
}

def generate_poisson_times(num_requests, duration):
    """
    Generate time points following Poisson distribution within given duration
    
    Args:
        num_requests: Total number of requests
        duration: Duration (seconds)
    
    Returns:
        times: List of time points when requests should be sent (relative to start time)
    """
    # Generate random time points uniformly distributed on [0, duration]
    uniform_times = np.random.uniform(0, duration, num_requests)
    
    # Sort to get ordered time points
    times = np.sort(uniform_times)
    
    return times

def send_request(thread_id, sleep_time, dataset_file, metrics_file, concurrency_config, rps):
    """Send request and handle streaming response"""
    try:
        # Read request type
        with open(dataset_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header
            for row in csvreader:
                if int(row[0]) == thread_id:
                    request_type = row[1]
                    break
            else:
                print(f"Thread ID {thread_id} not found in dataset.")
                return

        # Prepare request data
        sample_input = data_templates[request_type]
        
        # Initialize metrics
        metrics = {
            'start_time': time.time(),
            'first_token_time': None,
            'end_time': None,
            'error': None
        }
        
        # Send request and handle streaming response
        response = requests.post(url, json=sample_input, headers=headers, stream=True)
        
        # Process streaming response
        for chunk in response.iter_lines():
            if chunk:
                if metrics['first_token_time'] is None:
                    metrics['first_token_time'] = time.time()
                metrics['end_time'] = time.time()
        
        # Ensure request succeeded
        response.raise_for_status()
        
        # Calculate metrics
        waiting_time = metrics['first_token_time'] - metrics['start_time'] if metrics['first_token_time'] else None
        total_time = metrics['end_time'] - metrics['start_time'] if metrics['end_time'] else None
        execution_time = metrics['end_time'] - metrics['first_token_time'] if metrics['first_token_time'] and metrics['end_time'] else None
        
        # Record metrics
        with open(metrics_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                thread_id,
                request_type,
                waiting_time,
                execution_time,
                total_time,
                concurrency_config,
                rps
            ])
            
    except Exception as e:
        print(f"Thread {thread_id}: Request failed: {e}")

def run_experiment(dataset_file, metrics_file, concurrency_config, rps):
    """Run experiment, sending requests following Poisson distribution"""
    # Create metrics file
    with open(metrics_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            "Thread ID", 
            "Request Type", 
            "Waiting Time", 
            "Execution Time", 
            "Total Round Trip Time", 
            "Concurrency Config",
            "RPS"
        ])

    # Read dataset to determine number of requests
    with open(dataset_file, 'r') as csvfile:
        num_requests = sum(1 for _ in csvfile) - 1  # Subtract header row

    # Calculate experiment duration
    duration = num_requests / rps
    print(f"Starting experiment with {num_requests} requests at target RPS {rps}")
    print(f"Expected duration: {duration:.2f} seconds")
    
    # Generate Poisson distributed time points
    request_times = generate_poisson_times(num_requests, duration)
    
    # Create a shared variable to record the last request send time
    last_request_time = [0]  # Use list for easy sharing between threads
    
    # Start threads
    threads = []
    start_time = time.time()
    
    for i in range(num_requests):
        # Create a thread that records the last request send time
        thread = threading.Thread(
            target=lambda idx, sleep_time: (
                time.sleep(sleep_time),
                # Update last request time
                last_request_time.__setitem__(0, time.time() - start_time),
                send_request(idx, 0, dataset_file, metrics_file, concurrency_config, rps)
            ),
            args=(i, request_times[i])
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Calculate actual RPS using the last request send time
    actual_duration = last_request_time[0]  # Time when last request was sent
    actual_rps = num_requests / actual_duration if actual_duration > 0 else 0
    
    end_time = time.time()
    total_experiment_time = end_time - start_time
    
    print(f"Experiment completed for RPS {rps}")
    print(f"Expected duration: {duration:.2f} seconds")
    print(f"Last request sent at: {actual_duration:.2f} seconds")
    print(f"Total experiment time: {total_experiment_time:.2f} seconds")
    print(f"Actual RPS: {actual_rps:.2f} requests per second")
    
    # Add summary information to metrics file
    with open(metrics_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Empty line separator
        csvwriter.writerow([])
        # Write overall statistics
        csvwriter.writerow(["Summary", "", "", "", "", "", ""])
        csvwriter.writerow(["Expected Duration", f"{duration:.2f}s", "", "", "", "", ""])
        csvwriter.writerow(["Last Request Sent", f"{actual_duration:.2f}s", "", "", "", "", ""])
        csvwriter.writerow(["Total Experiment Time", f"{total_experiment_time:.2f}s", "", "", "", "", ""])
        csvwriter.writerow(["Target RPS", rps, "", "", "", "", ""])
        csvwriter.writerow(["Actual RPS", f"{actual_rps:.2f}", "", "", "", "", ""])

if __name__ == "__main__":
    dataset_dir = "./generated_workloads"
    output_dir = "baseline_experiment_results"

    concurrency_config = int(sys.argv[1])
    
    rps_list = [1,2,3,4,5,6,7,8,9,10,15,20]
    
    # Create subfolder named after concurrency configuration
    concurrency_dir = os.path.join(output_dir, f"max_concurrency_{concurrency_config}")
    os.makedirs(concurrency_dir, exist_ok=True)
    
    # Run experiments for each workload and RPS combination
    for workload_name in ["workload1", "workload2", "workload3"]:
        # Create workload subfolder under concurrency configuration folder
        workload_output_dir = os.path.join(concurrency_dir, workload_name)
        os.makedirs(workload_output_dir, exist_ok=True)
        
        dataset_file = os.path.join(dataset_dir, f"{workload_name}.csv")
        
        if os.path.exists(dataset_file):
            # Run experiment for each RPS value
            for rps in rps_list:
                # Create filename containing all parameters
                metrics_file = os.path.join(
                    workload_output_dir, 
                    f"{workload_name}_concurrency{concurrency_config}_rps{rps}.csv"
                )
                
                print(f"\nRunning experiment for workload: {workload_name}, concurrency: {concurrency_config}, RPS: {rps}")
                run_experiment(dataset_file, metrics_file, concurrency_config, rps)
        else:
            print(f"Dataset file {dataset_file} not found.")
    
    print("\nAll experiments completed successfully.")