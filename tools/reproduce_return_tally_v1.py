"""Reproduce return_tally_v1 in a separate, disposable directory and compare.

Never touches the accepted results in this tree. Copies remnant_core/ and experiments/
into DEST (which must not exist), so the copy has no results file and the driver's
one-shot --run is allowed there.

  python3 tools/reproduce_return_tally_v1.py --dest DIR --seeds 7001,7030
      fast check (seconds): re-runs the listed seeds of every arm in the copy and compares
      each row byte-for-byte (as JSON) with the accepted rows.
  python3 tools/reproduce_return_tally_v1.py --dest DIR --full
      full sweep (5 arms x 30 seeds x 100 tides; about 2 minutes on CPython 3.12):
      runs the copy's driver with --run and compares the sha256 of the new result file
      with the accepted one.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCEPTED = os.path.join(ROOT, "results", "return_tally_v1.json")


def usage():
    print("usage: reproduce_return_tally_v1.py --dest DIR (--full | --seeds S1,S2,...)", file=sys.stderr)
    sys.exit(2)


def main():
    a = sys.argv[1:]
    if len(a) not in (3, 4) or a[0] != "--dest":
        usage()
    dest = os.path.abspath(a[1])
    mode = a[2]
    if mode == "--full" and len(a) == 3:
        seeds = None
    elif mode == "--seeds" and len(a) == 4:
        try:
            seeds = [int(s) for s in a[3].split(",")]
        except ValueError:
            usage()
    else:
        usage()
    if os.path.exists(dest):
        sys.exit(f"refusing: {dest} exists; choose a new disposable directory")
    if dest.startswith(ROOT + os.sep):
        sys.exit("refusing: choose a destination outside this source tree")
    os.makedirs(dest)
    ign = shutil.ignore_patterns("__pycache__")
    shutil.copytree(os.path.join(ROOT, "remnant_core"), os.path.join(dest, "remnant_core"), ignore=ign)
    shutil.copytree(os.path.join(ROOT, "experiments"), os.path.join(dest, "experiments"), ignore=ign)
    accepted_bytes = open(ACCEPTED, "rb").read()
    accepted = json.loads(accepted_bytes)
    if seeds is None:
        p = subprocess.run([sys.executable, os.path.join("experiments", "run_return_tally_v1.py"), "--run"],
                           cwd=dest, capture_output=True, text=True)
        print(p.stdout[-600:])
        if p.returncode != 0:
            print(p.stderr, file=sys.stderr)
            sys.exit(p.returncode)
        new = open(os.path.join(dest, "results", "return_tally_v1.json"), "rb").read()
        h_new, h_acc = hashlib.sha256(new).hexdigest(), hashlib.sha256(accepted_bytes).hexdigest()
        print("new     ", h_new)
        print("accepted", h_acc)
        print("IDENTICAL" if h_new == h_acc else "DIFFERENT")
        sys.exit(0 if h_new == h_acc else 1)
    code = ("import json,sys; sys.path.insert(0,'experiments'); import run_return_tally_v1 as d; "
            "seeds=json.loads(sys.argv[1]); "
            "print(json.dumps({a:[d.run_seed(s_, s) for s in seeds] for a,s_ in d.ARM_SPECS.items()}))")
    p = subprocess.run([sys.executable, "-c", code, json.dumps(seeds)], cwd=dest, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
        sys.exit(p.returncode)
    fresh = json.loads(p.stdout)
    bad = 0
    for arm, rows in fresh.items():
        acc_rows = {r["seed"]: r for r in accepted["results"][arm]}
        for row in rows:
            ok = json.loads(json.dumps(row)) == acc_rows[row["seed"]]
            bad += not ok
            print(f"{arm:15s} seed {row['seed']}: {'IDENTICAL' if ok else 'DIFFERENT'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
