#!/bin/bash
# Start the queue in the background. Run from the bundle folder with  bash run.sh  after setup_venv.sh and the data are in place.
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p "$DS_CACHE_DIR"
[ -x .venv/bin/python ] || { echo "no .venv yet, run  bash setup_venv.sh  first"; exit 1; }
[ -d "$DS_DATA_ROOT/Train_Set" ] || { echo "no Train_Set in $DS_DATA_ROOT, see README.txt"; exit 1; }
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s queue "bash -c 'cd \"$PWD\" && source ./env.sh && python run_queue.py queue-a100.jsonl 2>&1 | tee -a driver.out'"
  echo "started in tmux session queue, watch with  tmux attach -t queue  (detach with Ctrl-b d)"
else
  nohup bash -c 'source ./env.sh && python run_queue.py queue-a100.jsonl' >> driver.out 2>&1 &
  echo "started with nohup, watch with  tail -f driver.out queue-log.txt"
fi
