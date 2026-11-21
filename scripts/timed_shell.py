#!/usr/bin/env python3
"""A shell for make that records when each recipe line starts and ends.

`make gate-timing` runs `make test SHELL=<this file>` with
CAMDL_GATE_TIMING_LOG naming a log file. make passes every recipe line, and
every `$(shell ...)` call, to this program as `[flags...] -c <command>`, and
passes the setting on to nested makes, including the one inside a test slot.
This program runs the command under bash with the same arguments, copies its
standard output and standard error through unchanged, and appends one
tab-separated record per event to the log:

    <unix time>  <pid>  start  <command>
    <unix time>  <pid>  out    <one line the command printed, either stream>
    <unix time>  <pid>  end    <exit status>

Records from concurrent recipe lines interleave; the pid keeps them apart.
Without CAMDL_GATE_TIMING_LOG it execs bash directly. The exit status is the
command's. `scripts/gate_timing_report.py` reads the log.
"""

import os
import subprocess
import sys
import threading
import time

LOG = os.environ.get("CAMDL_GATE_TIMING_LOG")
ARGS = sys.argv[1:]

if not LOG:
    os.execvp("bash", ["bash", *ARGS])

LOCK = threading.Lock()
PID = os.getpid()


def record(kind: str, payload: str) -> None:
    clean = payload.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    line = f"{time.time():.3f}\t{PID}\t{kind}\t{clean}\n"
    with LOCK, open(LOG, "a", encoding="utf-8") as f:
        f.write(line)


def copy(src, dst) -> None:
    for raw in iter(src.readline, b""):
        dst.write(raw)
        dst.flush()
        record("out", raw.decode("utf-8", errors="replace").rstrip("\n"))
    src.close()


record("start", ARGS[-1] if ARGS else "")
child = subprocess.Popen(
    ["bash", *ARGS], stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
copiers = [
    threading.Thread(target=copy, args=(child.stdout, sys.stdout.buffer)),
    threading.Thread(target=copy, args=(child.stderr, sys.stderr.buffer)),
]
for t in copiers:
    t.start()
try:
    status = child.wait()
except KeyboardInterrupt:
    # make, the wrapper and the command share a process group, so the command
    # received the same SIGINT; wait for it to finish exiting.
    status = child.wait()
for t in copiers:
    t.join()
status = 128 - status if status < 0 else status
record("end", str(status))
sys.exit(status)
