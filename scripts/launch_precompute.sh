#!/bin/bash
# Launch parallel precompute across GPUs, then merge shards.
#
# Usage:
#   bash scripts/launch_precompute.sh
#   bash scripts/launch_precompute.sh --hf tejasrao/ttdr-bridge-encodings

set -e

# Defaults — override via env vars (e.g. NUM_SHARDS=4 bash scripts/launch_precompute.sh)
NUM_SHARDS=${NUM_SHARDS:-8}
DATA_DIR=${DATA_DIR:-/home/ubuntu/data/rlds}
OUTPUT_DIR=${OUTPUT_DIR:-data/bridge_v2_encodings}
BATCH_SIZE=${BATCH_SIZE:-64}
CHUNK_SIZE=${CHUNK_SIZE:-4}
WINDOW_SIZE=${WINDOW_SIZE:-2}
HF_REPO=""

# Parse optional --hf flag
while [[ $# -gt 0 ]]; do
  case $1 in
    --hf) HF_REPO="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

echo "Launching $NUM_SHARDS precompute workers..."

for i in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/precompute_encodings.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_size $CHUNK_SIZE \
    --batch_size $BATCH_SIZE \
    --window_size $WINDOW_SIZE \
    --shard_id $i \
    --num_shards $NUM_SHARDS &
done

echo "Waiting for all workers to finish..."
wait
echo "All workers done."

echo "Merging shards..."
MERGE_CMD="python scripts/merge_encoding_shards.py --input_dir $OUTPUT_DIR --num_shards $NUM_SHARDS"
if [ -n "$HF_REPO" ]; then
  MERGE_CMD="$MERGE_CMD --hf_repo $HF_REPO"
fi
$MERGE_CMD

echo "Precompute complete. Output: $OUTPUT_DIR/encodings.h5"
