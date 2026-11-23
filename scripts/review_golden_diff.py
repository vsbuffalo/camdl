"""Review aid, read-only. For every `*.ir.json` that differs between two commits
of the camdl repository, parse the old and new versions as JSON and classify
the change:

  declared+version  the new file equals the old one except that `ir_version`
                    and `validated_by` went from the old IR version to the new
                    and a `declared` member was added;
  version only      only the two version strings changed (hand-maintained
                    files the compiler does not regenerate);
  other             anything else. Each is listed with the top-level keys, and
                    the keys of `model`, that differ.

Usage, from the repository root:
    python3 scripts/review_golden_diff.py <old-commit> <new-commit> [<old-ir> <new-ir>]
<new-commit> may be the word WORKTREE to read the working tree. The IR versions
default to 0.43 and 0.44, the bump that added the declared section.

Blind spot: files are compared as parsed JSON, which ignores the order of
object keys, and the order of `initial_conditions` is part of model identity.
A reordering is caught by the hash test
(`rust/crates/ir/tests/golden_model_hashes.rs`), not by this script.
"""
import json
import subprocess
import sys



def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def read(rev, path):
    if rev == "WORKTREE":
        return open(path).read()
    return git("show", f"{rev}:{path}")


def main():
    old, new = sys.argv[1], sys.argv[2]
    OLD_V, NEW_V = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("0.43", "0.44")
    rng = [old] if new == "WORKTREE" else [old, new]
    changed = [p for p in git("diff", "--name-only", *rng, "--", "*.ir.json").splitlines() if p]
    status = dict(
        line.split("\t")[::-1]
        for line in git("diff", "--name-status", *rng, "--", "*.ir.json").splitlines()
        if line
    )
    counts = {"declared+version": 0, "version only": 0, "other": 0}
    others = []
    old_bytes = new_bytes = 0
    for path in sorted(changed):
        if status.get(path) != "M":
            counts["other"] += 1
            others.append((path, [f"file status {status.get(path)}"]))
            print(f"other             {path}")
            continue
        o_text, n_text = read(old, path), read(new, path)
        old_bytes += len(o_text.encode())
        new_bytes += len(n_text.encode())
        o, n = json.loads(o_text), json.loads(n_text)
        notes = []
        if o.get("ir_version") != OLD_V or n.get("ir_version") != NEW_V:
            notes.append(f"ir_version {o.get('ir_version')!r} -> {n.get('ir_version')!r}")
        expect_vb = (o.get("validated_by") or "").replace(OLD_V, NEW_V)
        if n.get("validated_by") != expect_vb:
            notes.append(f"validated_by {o.get('validated_by')!r} -> {n.get('validated_by')!r}")
        for key in sorted(set(o) | set(n)):
            if key in ("ir_version", "validated_by", "declared"):
                continue
            if o.get(key) != n.get(key):
                if key == "model" and isinstance(o.get(key), dict) and isinstance(n.get(key), dict):
                    inner = sorted(k for k in set(o[key]) | set(n[key]) if o[key].get(k) != n[key].get(k))
                    notes.append(f"model differs in: {', '.join(inner)}")
                else:
                    notes.append(f"top-level `{key}` differs")
        if "declared" in o:
            notes.append("old file already had `declared`")
        if notes:
            kind = "other"
            others.append((path, notes))
        elif "declared" in n:
            kind = "declared+version"
        else:
            kind = "version only"
        counts[kind] += 1
        size = len(json.dumps(n["declared"])) if "declared" in n else 0
        print(f"{kind:17s} {path}  (declared: {size} bytes compact)")
    print()
    print(f"{len(changed)} files differ between {old} and {new}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  total bytes of these files: {old_bytes} -> {new_bytes}")
    print()
    if others:
        print("Files that differ in any other way:")
        for path, notes in others:
            print(f"  {path}")
            for note in notes:
                print(f"    - {note}")
    else:
        print("Files that differ in any other way: none")


if __name__ == "__main__":
    main()
