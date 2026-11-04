"""Fetch the three released networks and the authors' evaluation file, verifying checksums.

Nothing from the source release is vendored here. The networks are CC BY 4.0, and the authors'
Experiment 5 prompts embed MinAtar source under GPL-3.0, so this repository links to the upstream
release rather than redistributing any of it.

    python fetch_data.py
"""
import hashlib
import os
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main"

FILES = {
    "data/networks/breakout.hand.npz": (
        f"{RAW}/exp5_write_from_scratch/networks/breakout/hand.npz",
        "6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a",
    ),
    "data/networks/seaquest.hand.npz": (
        f"{RAW}/exp5_write_from_scratch/networks/seaquest/hand.npz",
        "2625453048e97b678d73dcf199dfb44d7ef659eb9c94e15263c6c33035fb3e7e",
    ),
    "data/networks/asterix.hand.npz": (
        f"{RAW}/exp5_write_from_scratch/networks/asterix/hand.npz",
        "4c537e227824318c650d98e1e52de5a106071699213b7551e0810af71e9cdf1e",
    ),
    "data/authors_eval.json": (
        f"{RAW}/exp5_write_from_scratch/eval/minatar_eval.json",
        "bc10c6f957af95cdd52ee4be085acd249046f6117dbdb8c735fd5ddb2195c47c",
    ),
}

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    failed = []
    for rel, (url, want) in FILES.items():
        dst = os.path.join(HERE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            got = hashlib.sha256(open(dst, "rb").read()).hexdigest()
            if got == want:
                print(f"ok (cached)  {rel}")
                continue
        print(f"fetching     {rel}")
        with urllib.request.urlopen(url, timeout=60) as r:
            blob = r.read()
        got = hashlib.sha256(blob).hexdigest()
        if got != want:
            failed.append((rel, want, got))
            print(f"  CHECKSUM MISMATCH\n    expected {want}\n    got      {got}")
            continue
        with open(dst, "wb") as fh:
            fh.write(blob)
        print(f"ok           {rel}  ({len(blob):,} bytes)")

    if failed:
        print("\nUpstream content has changed since these checksums were pinned.")
        print("The results in results/ were produced against the pinned versions.")
        return 1
    print("\nAll files verified. Next: python sweep.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
