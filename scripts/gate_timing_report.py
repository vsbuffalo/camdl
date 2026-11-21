#!/usr/bin/env python3
"""Summarise where a `make gate-timing` run spent its wall-clock time.

Reads the log `scripts/timed_shell.py` writes and prints four tables:

1. The whole run: wall-clock time, and the time spent waiting for a test slot.
2. Recipe lines, longest first. A line that runs a nested make, or the
   test-slot wrapper around one, contains other lines; it is listed separately
   so its time is not counted twice.
3. Each cargo invocation, split into compile time (from the line's start to
   cargo's `Finished` line), time inside test harnesses (the sum of libtest's
   own `finished in` figures), and the rest: launching each test binary, its
   harness starting up, and doctest compilation.
4. Test binaries, longest first, each with its run time inside the harness
   and its launch overhead (wall time from cargo's `Running` line to the
   binary's last `test result` line, minus the harness time). On macOS the
   first launch of a freshly linked binary includes the system's security
   scan, so the overhead column is where that cost appears.

Usage: gate_timing_report.py LOG [--top N]
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RUNNING = re.compile(r"^\s*Running (\S+) \(")
DOCTESTS = re.compile(r"^\s*Doc-tests (\S+)")
RESULT = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed.*finished in ([\d.]+)s")
FINISHED = re.compile(r"^\s*Finished `")
SLOT_WAIT = "waiting for a test slot"


@dataclass
class Line:
    pid: str
    command: str
    start: float
    end: float | None = None
    status: str | None = None
    out: list[tuple[float, str]] = field(default_factory=list)

    @property
    def wall(self) -> float:
        return (self.end if self.end is not None else self.start) - self.start

    @property
    def is_container(self) -> bool:
        return bool(re.search(r"(^|[\s/])make\b|test-slot\.sh", self.command))


@dataclass
class Binary:
    name: str
    start: float
    last: float | None = None
    harness: float = 0.0
    passed: int = 0
    failed: int = 0

    @property
    def wall(self) -> float:
        return (self.last or self.start) - self.start

    @property
    def overhead(self) -> float:
        return max(self.wall - self.harness, 0.0)


def read(path: str) -> list[Line]:
    lines: dict[str, Line] = {}
    order: list[Line] = []
    for raw in open(path, encoding="utf-8", errors="replace"):
        parts = raw.rstrip("\n").split("\t", 3)
        if len(parts) < 4:
            continue
        t, pid, kind, payload = float(parts[0]), parts[1], parts[2], parts[3]
        if kind == "start":
            line = Line(pid, payload, t)
            lines[pid] = line
            order.append(line)
        elif pid in lines:
            if kind == "out":
                lines[pid].out.append((t, ANSI.sub("", payload)))
            elif kind == "end":
                lines[pid].end, lines[pid].status = t, payload
    return order


def binaries(line: Line) -> tuple[float | None, list[Binary]]:
    finished = None
    found: list[Binary] = []
    current: Binary | None = None
    for t, text in line.out:
        if finished is None and FINISHED.search(text):
            finished = t
        m = RUNNING.search(text) or DOCTESTS.search(text)
        if m:
            name = m.group(1)
            current = Binary(f"doctests {name}" if text.lstrip().startswith("Doc-tests") else name, t)
            found.append(current)
            continue
        r = RESULT.search(text)
        if r and current is not None:
            current.passed += int(r.group(1))
            current.failed += int(r.group(2))
            current.harness += float(r.group(3))
            current.last = t
    return finished, found


def short(command: str, width: int = 90) -> str:
    command = re.sub(r"\s+", " ", command).strip()
    return command if len(command) <= width else command[: width - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    lines = [ln for ln in read(args.log) if ln.end is not None or ln.out]
    if not lines:
        raise SystemExit(f"{args.log}: no records")
    t0 = min(ln.start for ln in lines)
    t1 = max((ln.end or ln.start) for ln in lines)

    wait = 0.0
    for ln in lines:
        if "test-slot.sh" in ln.command:
            said = [t for t, text in ln.out if SLOT_WAIT in text]
            inner = [o.start for o in lines if o.start > ln.start and o is not ln]
            if said and inner:
                wait += min(inner) - ln.start

    print(f"Whole run: {t1 - t0:.0f} s wall clock, {wait:.0f} s of it waiting for a test slot.\n")

    leaves = sorted((ln for ln in lines if not ln.is_container), key=lambda ln: -ln.wall)
    print(f"Recipe lines, longest {args.top} (of {len(leaves)}):")
    print(f"  {'wall':>7}  {'exit':>4}  command")
    for ln in leaves[: args.top]:
        print(f"  {ln.wall:7.1f}  {ln.status or '?':>4}  {short(ln.command)}")
    print()

    all_bins: list[Binary] = []
    rows = []
    for ln in leaves:
        finished, bins = binaries(ln)
        if finished is None and not bins:
            continue
        compile_s = (finished - ln.start) if finished is not None else 0.0
        harness = sum(b.harness for b in bins)
        other = max(ln.wall - compile_s - harness, 0.0)
        rows.append(f"  {ln.wall:7.0f}  {compile_s:7.0f}  {harness:7.0f}  {other:7.0f}  {len(bins):8d}  {short(ln.command, 60)}")
        all_bins.extend(bins)
    if rows:
        print("Cargo invocations:")
        print(f"  {'wall':>7}  {'compile':>7}  {'harness':>7}  {'other':>7}  {'binaries':>8}  command")
        print("\n".join(rows) + "\n")

    if all_bins:
        tests = [b for b in all_bins if not b.name.startswith("doctests")]
        over = sum(b.overhead for b in tests)
        print(
            f"Test binaries: {len(tests)}, {sum(b.harness for b in tests):.0f} s inside their harnesses, "
            f"{over:.0f} s launch overhead ({over / max(len(tests), 1):.2f} s per binary)."
        )
        print(f"  {'wall':>7}  {'harness':>7}  {'launch':>7}  {'tests':>5}  binary")
        for b in sorted(all_bins, key=lambda b: -b.wall)[: args.top]:
            print(f"  {b.wall:7.1f}  {b.harness:7.1f}  {b.overhead:7.2f}  {b.passed:5d}  {b.name}")
        failed = [b for b in all_bins if b.failed]
        if failed:
            print("\nBinaries with failures: " + ", ".join(b.name for b in failed))


if __name__ == "__main__":
    main()
