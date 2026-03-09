#!/bin/bash
source ~/venv/bin/activate
cd ~/TTDR
nohup python -m recap.eval.run_eval --config configs/adapt.yaml > eval_output.log 2>&1 &
echo "Started PID $!"
