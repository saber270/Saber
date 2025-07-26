#!/bin/bash

# Set the path to the Python script
SCRIPT_PATH="./saber.py"  # Replace with your actual script path

# Array of RPS values to test
RPS_VALUES=(1 2 3 4 5 6 7 8 9 10 15 20)
# RPS_VALUES=(20)

# Array of workload files to test (without extension)
WORKLOADS=("workload1" "workload2" "workload3")

# Create log directory
LOG_DIR="./experiment_logs"
mkdir -p $LOG_DIR

# Get current timestamp as run identifier
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Global log file
GLOBAL_LOG="$LOG_DIR/experiments_$TIMESTAMP.log"
echo "Starting batch experiments - $(date)" | tee $GLOBAL_LOG

# Total run counter
TOTAL_RUNS=$((${#RPS_VALUES[@]} * ${#WORKLOADS[@]}))
CURRENT_RUN=0

# Loop through all experiment combinations
for workload in "${WORKLOADS[@]}"; do
  for rps in "${RPS_VALUES[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    # Create individual log file
    LOG_FILE="$LOG_DIR/${workload}_${rps}_${TIMESTAMP}.log"
    
    echo "-------------------------------------------" | tee -a $GLOBAL_LOG
    echo "Running $CURRENT_RUN/$TOTAL_RUNS: workload=$workload, rps=$rps" | tee -a $GLOBAL_LOG
    echo "Start time: $(date)" | tee -a $GLOBAL_LOG
    echo "Log file: $LOG_FILE" | tee -a $GLOBAL_LOG
    
    # Run Python script
    echo "Command: python $SCRIPT_PATH --rps $rps --input_file $workload" | tee -a $GLOBAL_LOG
    
    # Execute command and redirect output to log file
    python $SCRIPT_PATH --rps $rps --input_file $workload > $LOG_FILE 2>&1
    
    # Check execution status
    if [ $? -eq 0 ]; then
      STATUS="SUCCESS"
    else
      STATUS="FAILED"
    fi
    
    echo "Status: $STATUS" | tee -a $GLOBAL_LOG
    echo "End time: $(date)" | tee -a $GLOBAL_LOG
    echo "-------------------------------------------" | tee -a $GLOBAL_LOG
    
    # Brief pause to avoid immediate next process startup
    sleep 2
  done
done

echo "All experiments completed - $(date)" | tee -a $GLOBAL_LOG

# Print experiment summary
echo "" | tee -a $GLOBAL_LOG
echo "===== EXPERIMENT SUMMARY =====" | tee -a $GLOBAL_LOG
echo "Total runs: $TOTAL_RUNS experiments" | tee -a $GLOBAL_LOG
echo "Workloads: ${WORKLOADS[*]}" | tee -a $GLOBAL_LOG
echo "RPS values: ${RPS_VALUES[*]}" | tee -a $GLOBAL_LOG
echo "Log directory: $LOG_DIR" | tee -a $GLOBAL_LOG
echo "Global log: $GLOBAL_LOG" | tee -a $GLOBAL_LOG
echo "===============================" | tee -a $GLOBAL_LOG