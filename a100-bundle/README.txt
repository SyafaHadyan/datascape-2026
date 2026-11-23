A100 bundle for the DATASCAPE 2026 runs (Jupyter terminal, no ssh)

1. Upload the zip with the Jupyter file browser, then in the terminal  unzip a100-bundle.zip && cd a100-bundle
2. bash setup.sh        only checks the GPU and internet (its pip install may fail, ignore that). If it says CANNOT RESOLVE huggingface.co, stop and tell Claude.
   bash setup_venv.sh   makes a private Python environment with torch 2.5.1 (the system torch 1.13 is too old and pip is read only). About 10 minutes.
3. Data. The folder archive/ must hold Train_Set/ and Test_Set/ (about 3.2 GB, the competition zip is 3.2 GB too).
   Option A, with internet and a Kaggle token:
     export KAGGLE_API_TOKEN=...   (or put kaggle.json in ~/.kaggle)
     kaggle competitions download -c datascape-2026-data-mining-competition -p archive && cd archive && unzip -q *.zip && cd ..
   Option B, upload the competition zip into archive/ with the file browser and unzip it there.
4. bash run.sh          starts the queue in tmux (or nohup), using the private environment and the a100venv kernel. 7 runs, each takes about 5 to 8 hours on a shared GPU.
5. Check progress       tail -n 5 queue-log.txt   and   nvidia-smi
6. Results appear as probs-<tag>.npz and submission-<tag>.csv in this folder. Download them with the file browser, one pair per run, and tell Claude.

Notes. Another process holds about 65 GB of the GPU, so the queue uses batch 16 (batch 8 for the 384 px run) with accumulation, the same effective batch of 32. If a run dies with CUDA out of memory the driver retries it after 5 minutes. To stop, tmux kill-session -t queue (a running notebook keeps going until it ends, kill it with pkill -f nbconvert).
Edit queue-a100.jsonl while it runs if you want to add or remove runs. The files run_queue.py and cv-notebook.ipynb can be replaced between runs.
