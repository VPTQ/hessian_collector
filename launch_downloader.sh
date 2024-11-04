#!/bin/bash

# Configuration
TOTAL_PROCESSES=64
START_IDX=0
END_IDX=200000000  # 200M
MIN_TEXT_LEN=1024

# Calculate the chunk size for each process
CHUNK_SIZE=$(( (END_IDX - START_IDX + TOTAL_PROCESSES - 1) / TOTAL_PROCESSES ))

# Launch processes
for ((i=0; i<TOTAL_PROCESSES; i++)); do
    PROCESS_START=$((START_IDX + i * CHUNK_SIZE))
    PROCESS_END=$((PROCESS_START + CHUNK_SIZE))
    
    # Ensure the last chunk doesn't exceed END_IDX
    if [ $PROCESS_END -gt $END_IDX ]; then
        PROCESS_END=$END_IDX
    fi
    
    echo "Launching process $i: range $PROCESS_START to $PROCESS_END"
    python dl_vision_datasets_mp.py \
        --start $PROCESS_START \
        --end $PROCESS_END \
        --min_text_len $MIN_TEXT_LEN \
        > "log_${PROCESS_START}_${PROCESS_END}.txt" 2>&1 &
done

echo "Launched $TOTAL_PROCESSES processes"
echo "Use 'ps aux | grep dl_vision_datasets_mp.py' to check running processes"
echo "Use 'tail -f log_*.txt' to monitor logs"