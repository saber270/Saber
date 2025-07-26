import time
import json
import requests
import csv
from threading import Lock, Thread, Condition
import pandas as pd
import numpy as np
import random
import argparse
import sys
import os
import logging
from queue import PriorityQueue

# Log configs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the vLLM API url
url = 'http://localhost:8000/generate'
headers = {'Content-Type': 'application/json'}

# window size for task selection
window_size = 20

# upper bound of vllm generation speed
speed_limit = 100

# Sample Prompt
Chat_QnAinput = "In a sprawling valley nestled between jagged mountain peaks, the town of Eldervale lay cloaked in the shimmering veil of twilight, its cobblestone streets glowing faintly under the amber light of antiquated lanterns. The air carried a subtle aroma of woodsmoke and wildflowers, mingling with the occasional, tantalizing scent of baked goods wafting from the hearths of homes that had stood for centuries. Eldervale was a place of old secrets and whispered tales, where every stone in the ancient walls seemed to hum with the resonance of bygone eras. At the heart of the town was the Grand Clocktower, its monumental gears visible through ornate glass panels, ticking steadily as though measuring the heartbeat of the world itself. Travelers often paused in their journeys to marvel at its intricate design, a masterpiece of engineering left behind by a civilization long forgotten, . "

codegen_input = 4*"Develop a Python-based Library Management System with three key features: managing books, members, and borrowing activities. Implement the system using object-oriented principles with Book, Member, and Library classes. The Book class should store details like title, author, ISBN, and availability. The Member class should handle member information, including name, ID, contact, and borrowed books. The Library class integrates the functionalities, managing books, members, and issuing or returning books. Use JSON files for persistent storage of data and ensure proper error handling. The system must adhere to PEP8 coding standards and be modular and maintainable"

Code_trans_input = 3*"Here's rewritten to meet the 1403-character count requirement: Create a Library Management System in Python that automates core library functions: managing books, members, and borrowing processes. The system should include book management, allowing adding, updating, deleting, and searching for books using attributes like title, author, ISBN, and availability status. Member management should support adding, updating, and removing members while tracking their borrowed books. The borrowing system should handle issuing and returning books, updating availability, and calculating penalties for late returns. Use object-oriented programming principles to design the system with three classes: Book, Member, and Library. The Book class represents individual books with properties such as title, author, ISBN, and availability. The Member class stores member details, including name, ID, contact, and borrowed books. The Library class integrates all functionalities, managing books, members, and borrowing activities. class Book: def __init__(self, title, author, isbn): self.title = title self.author = author self.isbn = isbn self.available = True class Member: def __init__(self, name, member_id, contact): self.name = name self.member_id = member_id self.contact = contact self.borrowed_books = [] class Library: def __init__(self): self.books = [] self.members = []"

Code_summary_input = "protected final void bindIndexed(ConfigurationPropertyName name, Bindable<?> target, AggregateElementBinder elementBinder, ResolvableType aggregateType, ResolvableType elementType, IndexedCollectionSupplier result) { for (ConfigurationPropertySource source : getContext().getSources()) { bindIndexed(source, name, target, elementBinder, result, aggregateType, elementType); if (result.wasSupplied() && result.get() != null) { return; } } }"

# SLA setting for each type of tasks
sla_thresholds = {
    "long_input_short_output": 1,    # CodeSum
    "short_input_short_output": 1,   # ChatQnA
    "short_input_long_output": 8,    # CodeGen
    "long_input_long_output": 12     # CodeTrans
}

# data template
data_templates = {
    'long_input_long_output': {
        'prompt': Code_trans_input,
        'max_tokens': 617,
        'temperature': 0.1
    },
    'short_input_short_output': {
        'prompt': Chat_QnAinput,
        'max_tokens': 51,
        'temperature': 0.1
    },
    'short_input_long_output': {
        'prompt': codegen_input,
        'max_tokens': 387,
        'temperature': 0.1
    },
    'long_input_short_output': {
        'prompt': Code_summary_input,
        'max_tokens': 48,
        'temperature': 0.1
    }
}


# prepare output file, can be overwritten in main function
output_file = ""  

# data tracing
request_counters = {
    'total_requests': 0,
    'high_priority_processed': 0,
    'low_priority_processed': 0,
    'auto_downgraded': 0,
    'manually_added_low_priority': 0,
    'csv_records_written': 0
}


# Request Class
class Request:
    def __init__(self, index, start_time, model, request_tokens, total_tokens, log_type, max_new_tokens, time_sla, request_type):
        self.index = index
        self.start_time = start_time  
        self.low_priority_start_time = None  
        self.high_priority_wait_time = None  
        self.model = model
        self.request_tokens = request_tokens
        self.total_tokens = total_tokens  
        self.log_type = log_type
        self.max_new_tokens = max_new_tokens
        self.time_sla = time_sla
        self.deadline = self.start_time + self.time_sla 
        self.prompt = ""  
        self.request_type = request_type
        self.in_low_priority = False  
        self.priority = 0  
        self.is_processed = False  
        self.is_written_to_csv = False  
        self.processed_lock = Lock() 
        
        
    def __str__(self):
        return (f"Request(Index: {self.index}, Start Time: {self.start_time:.2f}s, "
                f"SLA: {self.time_sla:.2f}s, Deadline: {self.deadline:.2f}s, "
                f"Model: {self.model}, Tokens: {self.total_tokens}, "
                f"Request Type: {self.request_type}, Priority: {self.priority:.2f}, "
                f"Low Priority: {'Yes' if self.in_low_priority else 'No'}")

task_queue = [] # high priority queue
low_priority_queue = [] #low priority queue

# locks for multi-threads operations
task_queue_lock = Lock()
low_priority_queue_lock = Lock()
queue_condition = Condition(task_queue_lock)

# SABER agent
class Agent:
    def __init__(self, url, agent_id, rps):
        self.current_requests = 0
        self.lock = Lock()
        self.pending_requests = []
        self.url = url
        self.id = agent_id
        self.max_idle_time = 10
        self.idle_start_time = None
        self.check_interval = 0.1
        self.active_requests_info = []  
        self.active_requests_lock = Lock()  
        logger.info(f"Agent {agent_id} initialized, request fetch interval is set as: {self.check_interval:.2f}s")
    
    
    def estimate_generation_speed(self, additional_requests=1):
        """estimate generation speed with current load"""
        load = len(self.active_requests_info) + additional_requests
        # speed = 86.65 + -0.7581 * load # Linear
        speed = ( 102.55 / (1 + 0.003 * load - 0.001 * load * load))  # USL
        # speed = 7875888.46  / (1 + np.exp(-(-0.0130) * (load - (-876.47)))) #logarithm
        return speed 
        
    def discard(self, request):
        """move request to low priority queue"""
        try:
            # remove requests from high priority queue
            with task_queue_lock:
                if request in task_queue:
                    task_queue.remove(request)
                    logger.info(f"Removal:  Request {request.index} from high priority queue.")
                else:
                    logger.warning(f"Request {request.index} is not found in high priority queue!")
            
            #  record the count for discard requests
            request_counters['auto_downgraded'] += 1
            
            #  mark the request as low priority and move to low priority queue
            with low_priority_queue_lock:
                request.in_low_priority = True  #  mark to low priority
                request.low_priority_start_time = time.time()  # record the time to low priority
                request.high_priority_wait_time = request.low_priority_start_time - request.start_time  # compute the waiting time in high priority queue
                
                logger.info(f"Request {request.index} waits  {request.high_priority_wait_time:.2f}s in high priority queue")
                
                low_priority_queue.append(request)
                logger.info(f"Add: Request {request.index} to low priority queue. Queue size: {len(low_priority_queue)}")
                
                # show all the requests in low priority queue
                low_prio_ids = [req.index for req in low_priority_queue]
                logger.info(f"Current requests in low priority queue: {low_prio_ids}")
        except Exception as e:
            logger.error(f"Error in discard method for request {request.index}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def process_request(self, req, low_priority=False):
        """Process single request"""        
        start_execution_time = time.time()
        
        # Compute remaining time
        remaining_time = req.deadline - start_execution_time
        logger.info(f"Request {req.index} has {remaining_time:.2f}s until SLA deadline")
        
    
        # check if need to discard the request to low priority queue
        if not low_priority and not req.in_low_priority and remaining_time > 0:

            processing_speed_required = -1
            if remaining_time > 0:
            # compute the required speed
                processing_speed_required = req.max_new_tokens / remaining_time
            
            # if required speed higher than limit, discard to low priority queue
            if processing_speed_required > speed_limit or remaining_time < 0:
                logger.warning(f"Request {req.index} need processing speed reach {processing_speed_required:.2f} tokens/over {speed_limit} tokens/s limit, discard")
                self.discard(req)
                return  
        
        # evaluate speed
        if not low_priority: 
            expected_speed = req.max_new_tokens / remaining_time if remaining_time > 0 else float('inf')
            estimation_speed = self.estimate_generation_speed(len(self.active_requests_info) + 1)
        
            with self.active_requests_lock:
                # query the required speed to meet SLA, for all the active requests
                active_speeds = [s for _, s in self.active_requests_info]
                logger.info(active_speeds)
                min_active_speed = max(active_speeds) if active_speeds else 0
                
                # if adding current request won't let any request violate SLA
                if estimation_speed >= expected_speed and estimation_speed >= min_active_speed:
                    # meet condition, join execution
                    self.active_requests_info.append((req, expected_speed))
                    logger.info(f"Request {req.index} meet condition | estimated speed:{estimation_speed:.2f} >= expected speed:{expected_speed:.2f} and >= minimum active speed:{min_active_speed:.2f}")
                else:
                    # does not meet condition, back to request queue
                    logger.warning(f"Reject request {req.index} |  estimated speed:{estimation_speed:.2f} <  expected speed:{expected_speed:.2f} or < minimum active speed:{min_active_speed:.2f}")
                    self.return_to_high_priority(req)
                    return  

        is_low_priority = low_priority or req.in_low_priority
        status = "low_priority" if is_low_priority else "high_priority"
        logger.info(f"Start processing {status} Request {req.index}")
        
        tokens = req.total_tokens
        payload = {
            'prompt': req.prompt,
            'max_tokens': int(tokens),
            'temperature': 0.7,
            'stream': True
        }
        
        try:
            if is_low_priority:
                request_counters['low_priority_processed'] += 1
            else:
                request_counters['high_priority_processed'] += 1
                
            response = requests.post(self.url, headers=headers, json=payload, stream=True, timeout=30)
            
            first_token_time = None
            token_count = 0
            
            # processing response streamming
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # record TTFT
                if first_token_time is None:
                    first_token_time = time.time()
                    service_time_to_first_token = first_token_time - start_execution_time
                    logger.info(f"First token is received after {service_time_to_first_token:.4f}s")
                
                # token count
                token_count += 1
            
            # record complete time
            response_received_time = time.time()
            
            logger.info(f"Collect {token_count} tokens in total, executing for {response_received_time - start_execution_time:.4f}s")
            
            # calculate waiting time, from request enqueue to first token
            waiting_time = first_token_time - req.start_time
            
            # calculate execution time, from first token to the end
            execution_time = response_received_time - first_token_time
            
            # calcualte e2e time
            total_time = response_received_time - req.start_time
            
            # get SLA time
            sla_time = sla_thresholds.get(req.request_type, 10.0)
            
            #record in csv
            csv_file_lock = getattr(self, 'csv_file_lock', Lock())
            with csv_file_lock:
                if not getattr(req, 'is_written_to_csv', False):
                    with open(output_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            req.index,                # Thread ID
                            req.request_type,         # Request Type
                            waiting_time,             # Waiting Time
                            execution_time,           # Execution Time
                            total_time,               # Total Round Trip Time
                            sla_time,                 # SLA Time
                            "yes" if is_low_priority else "no",  # Low Priority
                            "yes" if total_time <= sla_time else "no"  # SLA Met
                        ])
                        req.is_written_to_csv = True
                        request_counters['csv_records_written'] += 1
                        logger.info(f"Successfully record Request {req.index} into CSV (Low priority: {'yes' if is_low_priority else 'no'})")
            
            logger.info(f"Processing Complete: Request {req.index} - Watiing time: {waiting_time:.4f}s, Execution time: {execution_time:.4f}s, "
                        f"Total time: {total_time:.4f}s, Low priority: {'yes' if is_low_priority else 'no'}, "
                        f"SLA: {'met' if total_time <= sla_time else 'not met'}")
        
        except Exception as e:
            logger.error(f"error while processing request {req.index} : {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            csv_file_lock = getattr(self, 'csv_file_lock', Lock())
            with csv_file_lock:
                if not getattr(req, 'is_written_to_csv', False):
                    try:
                        sla_time = sla_thresholds.get(req.request_type, 10.0)
                        total_time = time.time() - req.start_time
                        with open(output_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                req.index,
                                req.request_type,
                                0,
                                0,
                                total_time,
                                sla_time,
                                "yes" if is_low_priority else "no",
                                "no" 
                            ])
                            req.is_written_to_csv = True
                            request_counters['csv_records_written'] += 1
                            logger.info(f"Request {req.index} error record into CSV")
                    except Exception as write_err:
                        logger.error(f"Cannot record Request {req.index} error into CSV: {write_err}")
        
        finally:
            
            with self.lock:
                self.current_requests -= 1
                if req in self.pending_requests:
                    self.pending_requests.remove(req)
            
            # remove from current active requests
            with self.active_requests_lock:
                for i, (active_req, _) in enumerate(self.active_requests_info):
                    if active_req.index == req.index:
                        self.active_requests_info.pop(i)
                        logger.info(f"Remove Request {req.index} from active requests list")
                        break
                
            logger.info(f"Processing complete: Request {req.index}. Current number of requests: {self.current_requests}")
    
    def check_high_priority_queue(self):
        """check high priority queue to find request need to be discard to low priority queue"""
        current_time = time.time()
        to_downgrade = []
        
        with task_queue_lock:
            for req in task_queue:                    
        
                remaining_time = req.deadline - current_time
                if remaining_time < 0:  
                    to_downgrade.append(req)

        # discard the marked requests
        for req in to_downgrade:
            logger.info(f"Discard Request {req.index}")
            self.discard(req)
            
    def return_to_high_priority(self, request):
        """put requests back to high priority queue"""
        try:
            # reset the request label
            with request.processed_lock:
                request.is_processed = False
            
            with task_queue_lock:
                # put requests back to high priority queue
                task_queue.append(request)
                logger.info(f"Return: Request {request.index} is returned to high priority queue. Queue size: {len(task_queue)}")
                
                # log all the requests ID in high priority queue
                high_prio_ids = [req.index for req in task_queue]
                logger.info(f"Current requests in high priority queue: {high_prio_ids}")
        except Exception as e:
            logger.error(f"Request {request.index} occur error while back to high priority queue: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def dequeue_requests(self):        
        cycle_counter = 0          
        while True:
            cycle_counter += 1
            
            if cycle_counter % 10 == 0:
                with task_queue_lock:
                    high_queue_size = len(task_queue)
                    high_queue_ids = [req.index for req in task_queue[:5]] if task_queue else []
                
                with low_priority_queue_lock:
                    low_queue_size = len(low_priority_queue)
                    low_queue_ids = [req.index for req in low_priority_queue[:5]] if low_priority_queue else []
                
                with self.lock:
                    current_load = self.current_requests
                    pending_ids = [req.index for req in self.pending_requests[:5]] if self.pending_requests else []
                
                logger.info(f"=== System status (cycle {cycle_counter}) === "
                        f"High priority queue: {high_queue_size} {high_queue_ids if high_queue_ids else '[]'}, "
                        f"Low priority queue: {low_queue_size} {low_queue_ids if low_queue_ids else '[]'}, "
                        f"Current processing requests: {current_load}")
            
            # when system is idle
            with task_queue_lock:
                high_queue_empty = len(task_queue) == 0
            
            with low_priority_queue_lock:
                low_queue_empty = len(low_priority_queue) == 0
            
            # only idle when both queues are empty
            if high_queue_empty and low_queue_empty and self.current_requests == 0:
                if self.idle_start_time is None:
                    self.idle_start_time = time.time()
                    logger.info(f"Agent {self.id} is idle - all queues are empty")
                elif time.time() - self.idle_start_time > self.max_idle_time:
                    logger.info(f"Agent {self.id} had been idle for  {self.max_idle_time} seconds, terminating")
                    break
            else:
                # reset idle counter if any queue is not empty
                if self.idle_start_time is not None:
                    logger.info(f"Agent {self.id} is no longer idle")
                    self.idle_start_time = None
            
            if cycle_counter % 10 == 0:  
                self.check_high_priority_queue()
            
            #  process high priority queue first
            processed_high_priority = False
            with task_queue_lock:
                if len(task_queue) > 0:
                    # req = task_queue[0]
                    # task_queue.remove(req)
                    
                    candidates = task_queue[:min(len(task_queue),window_size)]
                    # randomly pick one from task window
                    req = random.choice(candidates)
                    task_queue.remove(req)
                    
                    #  process the request in thread
                    with self.lock:
                        self.current_requests += 1
                        self.pending_requests.append(req)
                        logger.info(f"Processing high priority request: {req.index}, current active requests: {self.current_requests}")
                    
                    def process_request_thread(request):
                        self.process_request(request)
                    
                    thread = Thread(target=process_request_thread, args=(req,))
                    thread.daemon = True
                    thread.start()
                    processed_high_priority = True
            
            # process low priority queue if high priority queue is empty
            if not processed_high_priority:
                with low_priority_queue_lock:
                    if len(low_priority_queue) > 0:
                        # pick the first one, FIFO
                        req = low_priority_queue[0]
                        low_priority_queue.remove(req)
                        
                        # process the request in thread
                        with self.lock:
                            self.current_requests += 1
                            self.pending_requests.append(req)
                            logger.info(f"Processing low priority request: {req.index}, current active requests: {self.current_requests}")
                        
                        def process_low_priority_thread(request):
                            self.process_request(request, True)  
                        
                        thread = Thread(target=process_low_priority_thread, args=(req,))
                        thread.daemon = True
                        thread.start()

            # sleep for a while to prevent CPU overload
            time.sleep(self.check_interval)
            
            

# generate arrival time in poisson with interval
def generate_poisson_arrivals(num_requests, lambda_rate):
    arrivals = []
    total_time = 0
    for _ in range(num_requests):
        interval = np.random.exponential(1/lambda_rate)
        total_time += interval
        arrivals.append(total_time)
    return arrivals

# enqueue from CSV
def enqueue_request(row, arrival_time):
    request_type = row['Request Type']
    template = data_templates.get(request_type, {})
    max_tokens = template.get('max_tokens', 100)
    prompt = template.get('prompt', '')

    time_sla = sla_thresholds.get(request_type, 10.0)
    
    req = Request(
        index=row['Thread ID'],
        start_time=arrival_time,
        model='Qwen2.5-Coder-3B',
        request_tokens=max_tokens,
        total_tokens=max_tokens,
        log_type='default',
        max_new_tokens=max_tokens,
        time_sla=time_sla,
        request_type=request_type
    )
    req.prompt = prompt
    
    request_counters['total_requests'] += 1

    with queue_condition:
        task_queue.append(req)
        logger.info(f"Enqueue: Request {req.index} - high priority queue size: {len(task_queue)}")
        queue_condition.notify_all()  # notify saber for new requests

# add low priority queue manually for debugging 
def add_low_priority_request():
    request_type = 'short_input_short_output'
    template = data_templates.get(request_type, {})
    max_tokens = template.get('max_tokens', 100)
    prompt = template.get('prompt', '')

    time_sla = sla_thresholds.get(request_type, 10.0)
    
    start_time = time.time()
    req = Request(
        index=1000, 
        start_time=start_time,
        model='Qwen2.5-Coder-3B',
        request_tokens=max_tokens,
        total_tokens=max_tokens,
        log_type='default',
        max_new_tokens=max_tokens,
        time_sla=time_sla,
        request_type=request_type
    )
    req.prompt = prompt
    req.in_low_priority = True  
    req.low_priority_start_time = start_time
    
    # update counter
    request_counters['total_requests'] += 1
    request_counters['manually_added_low_priority'] += 1
    
    # add to low priority queue
    with low_priority_queue_lock:
        low_priority_queue.append(req)
        logger.info(f"Added: Request {req.index} to low priority queue. Queue size: {len(low_priority_queue)}")
        low_prio_ids = [r.index for r in low_priority_queue]
        logger.info(f"Current requests in low priority queue: {low_prio_ids}")

# add low priroity for debugging
def add_multiple_low_priority_requests(count=5):
    for i in range(count):
        request_type = 'short_input_short_output'
        template = data_templates.get(request_type, {})
        max_tokens = template.get('max_tokens', 100)
        prompt = template.get('prompt', '')

        time_sla = sla_thresholds.get(request_type, 10.0)
        
        start_time = time.time()
        req = Request(
            index=1000 + i,  
            start_time=start_time,
            model='Qwen2.5-Coder-3B',
            request_tokens=max_tokens,
            total_tokens=max_tokens,
            log_type='default',
            max_new_tokens=max_tokens,
            time_sla=time_sla,
            request_type=request_type
        )
        req.prompt = prompt
        req.in_low_priority = True  
        req.low_priority_start_time = start_time
        
        request_counters['total_requests'] += 1
        request_counters['manually_added_low_priority'] += 1
        
        with low_priority_queue_lock:
            low_priority_queue.append(req)
            logger.info(f"Added: Request {req.index} to low priority queue. Queue size: {len(low_priority_queue)}")
    
    with low_priority_queue_lock:
        low_prio_ids = [r.index for r in low_priority_queue]        
        logger.info(f"Current requests in low priority queue: {low_prio_ids}")

def check_csv_records():
    try:
        with open(output_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  
            row_count = sum(1 for row in csv_reader)
        
        logger.info(f"CSV records: {row_count}")
        logger.info(f"counter records : {request_counters['csv_records_written']}")
        
        if row_count != request_counters['csv_records_written']:
            logger.warning(f"CSV records ({row_count}) and counter records  ({request_counters['csv_records_written']}) does not match!")
        
        return row_count
    except Exception as e:
        logger.error(f"Error while checking CSV: {e}")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run request with specific RPS')
    parser.add_argument('--rps', type=float, required=True, help='Requests per second')
    parser.add_argument('--input_file', type=str, required=True, help='Input file name (without extension)')
    args = parser.parse_args()

    rps = args.rps
    input_file_name = args.input_file
    
    os.makedirs("./agent_data", exist_ok=True)
    output_file = f"./agent_data/{input_file_name}_rps_{rps}.csv"    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Thread ID", 
            "Request Type", 
            "Waiting Time", 
            "Execution Time", 
            "Total Round Trip Time", 
            "SLA Time",
            "Low Priority",
            "SLA Met"
        ])

    # initialize saber agent
    agent1 = Agent(url=url, agent_id=1, rps=rps)
    
    # start in a separated thread
    agent1_thread = Thread(target=agent1.dequeue_requests)
    agent1_thread.daemon = True
    agent1_thread.start()

    # get workload requests information
    file_path = f'./generated_workloads/{input_file_name}.csv'
    try:
        dataset = pd.read_csv(file_path)
        dataset.columns = ['Thread ID', 'Request Type', 'Task Name', 'Deadline']
        logger.info(f"Found {len(dataset)} records")
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        sys.exit(1)

    # generate arrival time in poisson 
    start_time = time.time()
    arrival_times = generate_poisson_arrivals(len(dataset), rps)
    
    logger.info(f"Generated {len(arrival_times)} arrival times with mean interval {1/rps:.4f}s")
    
    # simulate send requests in poisson distribution to the task pool
    enqueue_threads = []
    for i, (_, row) in enumerate(dataset.iterrows()):
        absolute_arrival_time = start_time + arrival_times[i]
        
        def schedule_enqueue(r, t):
            time.sleep(max(0, t - time.time()))
            enqueue_request(r, t)
        
        thread = Thread(target=schedule_enqueue, args=(row, absolute_arrival_time))
        thread.daemon = True
        thread.start()
        enqueue_threads.append(thread)

    # wait until all the threads finished
    for thread in enqueue_threads:
        thread.join()
    
    logger.info("All requests have been enqueued. Waiting for processing to complete...")
    
    # set max wait time to terminate experiment after all requests finished.
    max_wait_time = 60  
    wait_start = time.time()
    
    while True:
        with task_queue_lock:
            high_queue_size = len(task_queue)
        
        with low_priority_queue_lock:
            low_queue_size = len(low_priority_queue)
        
        with agent1.lock:
            agent_requests = agent1.current_requests
        
        # Check if all finished
        if high_queue_size == 0 and low_queue_size == 0 and agent_requests == 0:
            logger.info("All queues are empty and free. Complete")
            break
        
        # In case high priority queue is empty but low priority queue is not, print log
        if high_queue_size == 0 and low_queue_size > 0:
            with low_priority_queue_lock:
                low_prio_ids = [req.index for req in low_priority_queue]
                logger.info(f"High priority queue is empty but low priority queue is not; Low priority requests IDs: {low_prio_ids}")
        
        # add low priority queue task to high priority queue after high priority queue is empty
        if high_queue_size == 0 and not add_low_priority_scheduled and time.time() - wait_start > 5:
            add_multiple_low_priority_requests(count=0)  
            add_low_priority_scheduled = True
        
        # check if exceed max waiting time
        if time.time() - wait_start > max_wait_time:
            logger.warning(f"Exceed max waiting time: {max_wait_time}s. "
                         f"remaining tasks: high priority: {high_queue_size}, "
                         f"low priority size: {low_queue_size}, "
                         f"Saber is processing: {agent_requests}")
            break
        
        time.sleep(1)
    
    # check csv records
    csv_record_count = check_csv_records()
    
    # Output final statistics
    logger.info("=== Final Statistics ===")
    logger.info(f"Total requests: {request_counters['total_requests']}")
    logger.info(f"High-priority requests processed: {request_counters['high_priority_processed']}")
    logger.info(f"Low-priority requests processed: {request_counters['low_priority_processed']}")
    logger.info(f"Requests automatically downgraded: {request_counters['auto_downgraded']}")
    logger.info(f"Manually added low-priority requests: {request_counters['manually_added_low_priority']}")
    logger.info(f"Records written to CSV: {request_counters['csv_records_written']}")
    logger.info(f"Actual records in CSV file: {csv_record_count}")

    logger.info("All requests have been processed or have reached the maximum wait time. Exiting program.")
