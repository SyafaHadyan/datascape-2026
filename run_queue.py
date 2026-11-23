#!/usr/bin/env python3
"""Run several notebook runs one after another, each with its own settings.

Usage (from the project folder, venv active, the usual exports set), python run_queue.py [queue.jsonl]

The queue file has one JSON object per line. RUN_TAG is required, every other name must be a
config setting of the notebook (see DS_OVERRIDES in the config cell). Names that start with an
underscore are not settings, "_env" is a dict of extra environment variables for that run only.
Example lines
  {"RUN_TAG": "run50-siglip2-seed2", "SEED_OFFSET": 2}
  {"RUN_TAG": "run51-siglip2-seed3", "SEED_OFFSET": 3, "_env": {"DS_FOLD_ASSIGNMENT": "2;1,3,5;4"}}
Lines starting with # are ignored. The file is read again before every run, so you can add or
remove lines while an earlier run is going.

A run counts as done when probs-<RUN_TAG>.npz exists. A run that fails for a real reason gets
failed-<RUN_TAG>.flag and is skipped, delete the flag to try it again (finished folds are kept,
the notebook resumes).
Set FINISH_BY="YYYY-MM-DD HH:MM" (local time) and EXPECTED_HOURS to stop starting runs that
could not finish in time. The driver never starts a run while a cv_worker.py or an nbconvert of yours is running, so it
can be started while an earlier run is still going and it then waits for that run to end.

Self healing. A failure whose log shows a temporary cause (no DNS or connection to the model hub,
a timeout, or CUDA out of memory because other users hold the GPU) is not flagged. The run is retried
later while the queue moves on to other runs, always after RETRY_WAIT_MIN minutes (default 5) and
at most MAX_RETRIES times (default 200, about 16 hours) before it is flagged. Set RETRY_BACKOFF=1
to double the wait after each failure up to 3 hours instead.
A run whose model weights are not in the local Hugging Face cache is not started while the hub
cannot be reached, it is checked again every 10 minutes and this does not count as a retry. At start-up,
failed-*.flag files whose log shows a temporary cause are removed so those runs get another chance.
Retry counts are kept in queue-retries.json.

Updating this script. The driver checks its own file before every run and every minute while it
waits. When the file changed (copy the new version to run_queue.py.new and mv it over, so it is
never half written) and the new version passes a self check, the driver restarts itself in place
with the same arguments and environment. A run that is training is never touched, because the
check happens between runs. A new version that fails the check is ignored with a log line. The
notebook and the queue file are also read fresh before every run, so all three can be updated
while the queue is going.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

queue_file = Path(sys.argv[1] if len(sys.argv) > 1 else "queue.jsonl")
notebook = os.environ.get("DS_NOTEBOOK", "datascape-2026-cv-notebook-l40.ipynb")
runner = os.environ.get("DS_RUNNER")  # only for testing, a command that replaces nbconvert
finish_by = os.environ.get("FINISH_BY")
expected = timedelta(hours=float(os.environ.get("EXPECTED_HOURS", "3")))
retry_wait = float(os.environ.get("RETRY_WAIT_MIN", "5")) * 60
max_retries = int(os.environ.get("MAX_RETRIES", "200"))
retry_backoff = os.environ.get("RETRY_BACKOFF", "0") == "1"
hub_recheck = min(600.0, retry_wait)
log_file = Path("queue-log.txt")
script_path = Path(__file__).resolve()
retry_file = Path("queue-retries.json")

# Text in a run log that points to a cause that can go away by itself.
TRANSIENT = (
    "Temporary failure in name resolution",
    "Name or service not known",
    "ConnectError",
    "Max retries exceeded",
    "Connection reset",
    "Connection aborted",
    "ReadTimeout",
    "timed out",
    "LocalEntryNotFoundError",
    "CUDA out of memory",
    "CUDA error: out of memory",
)


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def script_hash():
    return hashlib.md5(script_path.read_bytes()).hexdigest()


def reload_if_changed(started_hash):
    # Restart in place when this file changed on disk and the new version passes a self check.
    try:
        now_hash = script_hash()
    except OSError:
        return started_hash
    if now_hash == started_hash:
        return started_hash
    check = subprocess.run([sys.executable, str(script_path)], env={**os.environ, "DS_SELFCHECK": "1"},
                           capture_output=True, text=True)
    if check.returncode != 0 or "selfcheck ok" not in check.stdout:
        log(f"{script_path.name} changed on disk but the new version failed its self check, keeping the running version ({check.stderr.strip()[-200:] or check.stdout.strip()[-200:]})")
        return now_hash
    log(f"{script_path.name} changed on disk, restarting in place with the new version")
    os.execv(sys.executable, [sys.executable, str(script_path)] + sys.argv[1:])


def load_retries():
    try:
        return json.loads(retry_file.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_retries(retries):
    retry_file.write_text(json.dumps(retries, indent=1))


def transient_reason(tag):
    # Look at the end of the run log for a known temporary cause.
    try:
        with open(f"queue-{tag}.log", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 2_000_000))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for pattern in TRANSIENT:
        if pattern in text:
            return pattern
    return None


def hub_cache_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return home / "hub"


def needs_hub(job):
    # True when the run starts a timm model whose weights are not in the local cache yet.
    name = job.get("MODEL_NAME")
    if not name or name.startswith("hf-hub:"):
        return False
    snapshots = hub_cache_dir() / f"models--timm--{name}" / "snapshots"
    return not (snapshots.exists() and any(snapshots.iterdir()))


def hub_reachable():
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        socket.getaddrinfo("huggingface.co", 443)
        return True
    except OSError:
        return False


def heal_old_flags(retries):
    for flag in sorted(Path(".").glob("failed-*.flag")):
        tag = flag.name[len("failed-"):-len(".flag")]
        if retries.get(tag, {}).get("n", 0) >= max_retries:
            continue
        reason = transient_reason(tag)
        if reason:
            flag.unlink()
            log(f"healed {flag.name}, its log shows a temporary cause ({reason}), the run will be tried again")


def next_job(retries, now):
    # The first run that is not done, not flagged and not waiting for a retry. Also returns the
    # earliest time a waiting run becomes ready, or None when nothing is waiting.
    soonest = None
    for n, raw in enumerate(queue_file.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            job = json.loads(raw)
            tag = job["RUN_TAG"]
        except (json.JSONDecodeError, KeyError) as e:
            log(f"queue line {n} is not valid ({e}), skipped")
            continue
        if Path(f"probs-{tag}.npz").exists() or Path(f"failed-{tag}.flag").exists():
            continue
        until = retries.get(tag, {}).get("until", 0)
        if until > now:
            soonest = until if soonest is None else min(soonest, until)
            continue
        return job, soonest
    return None, soonest


def workers_running():
    # Training workers, or a notebook still in its final steps (test inference also uses the GPUs).
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "cv_worker.py|nbconvert"], capture_output=True)
    return r.returncode == 0


def main():
    if os.environ.get("DS_SELFCHECK"):
        print("selfcheck ok")
        return
    if not queue_file.exists():
        sys.exit(f"queue file {queue_file} not found")
    if not runner and not Path(notebook).exists():
        sys.exit(f"notebook {notebook} not found in {Path.cwd()}")
    log(f"queue driver started, queue {queue_file}, notebook {notebook}, finish by {finish_by or 'no limit'}")
    retries = load_retries()
    heal_old_flags(retries)
    started_hash = script_hash()
    waiting = False
    announced_wait = None
    deferred = set()
    while True:
        started_hash = reload_if_changed(started_hash)
        job, soonest = next_job(retries, time.time())
        if job is None:
            if soonest is None:
                log("nothing left in the queue, done")
                return
            if announced_wait != soonest:
                log(f"only runs waiting for a retry are left, the next one is ready at {datetime.fromtimestamp(soonest):%Y-%m-%d %H:%M:%S}")
                announced_wait = soonest
            time.sleep(max(1.0, min(soonest - time.time(), 60.0)))
            continue
        tag = job["RUN_TAG"]
        if finish_by and datetime.now() + expected > datetime.strptime(finish_by, "%Y-%m-%d %H:%M"):
            log(f"{tag} would not finish by {finish_by} (expected {expected}), not starting it, stopping")
            return
        if workers_running():
            if not waiting:
                log(f"a cv_worker.py or nbconvert of yours is still running, {tag} waits for it")
                waiting = True
            time.sleep(60)
            continue
        waiting = False
        if needs_hub(job) and not hub_reachable():
            if tag not in deferred:
                log(f"{tag} needs weights that are not cached and the model hub cannot be reached, skipping it for now and checking again in {hub_recheck / 60:.0f} min")
                deferred.add(tag)
            retries.setdefault(tag, {"n": retries.get(tag, {}).get("n", 0)})["until"] = time.time() + hub_recheck
            save_retries(retries)
            continue
        deferred.discard(tag)
        env_extra = job.pop("_env", {})
        overrides = {k: v for k, v in job.items() if not k.startswith("_")}
        env = {**os.environ, **{k: str(v) for k, v in env_extra.items()}, "DS_OVERRIDES": json.dumps(overrides)}
        cmd = runner.split() + [tag] if runner else [
            "jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=-1",
            "--output", f"{tag}-executed.ipynb", notebook]
        if not runner and os.environ.get("DS_KERNEL"):
            cmd.insert(-1, f"--ExecutePreprocessor.kernel_name={os.environ['DS_KERNEL']}")  # for example a kernel made from a venv
        log(f"START {tag} overrides {overrides} extra env {env_extra}")
        started = time.time()
        try:
            with open(f"queue-{tag}.log", "w") as out:
                code = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        except KeyboardInterrupt:
            log(f"interrupted by the user while running {tag}, stopping the queue (the run is not flagged as failed)")
            return
        hours = (time.time() - started) / 3600
        if code == 0 and Path(f"probs-{tag}.npz").exists():
            expected_new = timedelta(hours=hours)
            globals()["expected"] = expected_new
            retries.pop(tag, None)
            save_retries(retries)
            log(f"DONE {tag} in {hours:.2f} h, probs-{tag}.npz and submission-{tag}.csv are ready")
            continue
        probs_state = "present" if Path(f"probs-{tag}.npz").exists() else "missing"
        reason = transient_reason(tag)
        n = retries.get(tag, {}).get("n", 0) + 1
        if reason and n < max_retries:
            wait = min(retry_wait * 2 ** (n - 1), 3 * 3600.0) if retry_backoff else retry_wait
            retries[tag] = {"n": n, "until": time.time() + wait}
            save_retries(retries)
            log(f"FAILED {tag} (exit code {code}, probs file {probs_state}) with a temporary cause ({reason}), retry {n} of {max_retries - 1} in {wait / 60:.0f} min, moving on")
        else:
            why = f"exit code {code} after {hours:.2f} h"
            if reason:
                why += f", gave up after {n} attempts with a temporary cause ({reason})"
            Path(f"failed-{tag}.flag").write_text(why + "\n")
            retries[tag] = {"n": n, "until": 0}
            save_retries(retries)
            log(f"FAILED {tag} (exit code {code}, probs file {probs_state}), see queue-{tag}.log, moving on")


if __name__ == "__main__":
    main()
