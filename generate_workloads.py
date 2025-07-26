import csv
import os
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the task distributions for each workload
workload_distributions = {
    "workload1": {
        'long_input_long_output': 40,  # Code Generation
        'long_input_short_output': 10,  # Code Translation
        'short_input_long_output': 40,  # Chat QnA
        'short_input_short_output': 10  # Code Summary
    },
    "workload2": {
        'long_input_long_output': 10,  # Code Generation
        'long_input_short_output': 40,  # Code Translation
        'short_input_long_output': 10,  # Chat QnA
        'short_input_short_output': 40  # Code Summary
    },
    "workload3": {
        'long_input_long_output': 25,  # Code Generation
        'long_input_short_output': 25,  # Code Translation
        'short_input_long_output': 25,  # Chat QnA
        'short_input_short_output': 25  # Code Summary
    }
}


def generate_workloads(output_dir):
    """Generate separate workload files for each distribution."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for workload_name, distribution in workload_distributions.items():
        workload_file = os.path.join(output_dir, f"{workload_name}.csv")

        # Generate task sequence
        task_sequence = []
        for task_type, count in distribution.items():
            task_sequence.extend([task_type] * count)

        random.shuffle(task_sequence)

        # Write to CSV
        with open(workload_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Thread ID", "Request Type"])
            for i, task_type in enumerate(task_sequence):
                csvwriter.writerow([i, task_type])

        logging.info(f"Workload {workload_name} generated and saved to {workload_file}")


if __name__ == "__main__":
    output_dir = "generated_workloads"
    generate_workloads(output_dir)
