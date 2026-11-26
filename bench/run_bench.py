"""Run benchmark games ONE AT A TIME (the VM's host has few cores) and
summarise each: peak RSS, peak VM memory, per-Misty-move time, max |P|.

  python bench/run_bench.py --variant orig --python ~/venv/bin/python \
      --out ~/bench --seeds 0 1 2 3 4 5 [--env FOW_X=1 ...]

A game is a child process (bench/play_one.py). Its peak RSS is its own VmHWM,
read from /proc every 0.2 s, so a game the kernel OOM-kills still reports the
peak up to the last read. (wait4's ru_maxrss is the max over the process AND its
Stockfish children; Stockfish is ~250 MB, so it is kept only as
peak_rss_tree_mb.) VM memory
used (MemTotal - MemAvailable) is sampled every 0.2 s while the game runs; it
includes Stockfish and everything else on the VM.

Misty move time = the pick_move wall time + the belief updates fed since her
previous pick (her own last move and the opponent's reply): the time a live
opponent waits for her answer.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX_FLAGS = ("FOW_BOTTOMK_LOW_WATERMARK", "FOW_BOTTOMK_APPEND", "FOW_PSTATE_COMPACT",
             "FOW_ASSERT_OWN_SHARED")


def vm_used_mb() -> float:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / 1024


def proc_hwm_mb(pid: int) -> float | None:
    """VmHWM: the peak RSS of that one process (not its Stockfish children)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a kill
    return rows


def move_times(perply: list[dict]) -> list[dict]:
    out, pending, pending_obs = [], 0.0, 0.0
    for r in perply:
        if r.get("kind") in ("observe_own_move", "observe_opp_move"):
            pending += r.get("wall_ms", 0.0)
            pending_obs += r.get("wall_ms", 0.0)
        elif r.get("kind") == "pick_move":
            out.append({
                "ms": round(pending + r.get("wall_ms", 0.0), 1),
                "obs_ms": round(pending_obs, 1),
                "p": r.get("p_pre"),
            })
            pending = pending_obs = 0.0
    return out


def oom_killed(pid: int) -> bool:
    try:
        log = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return False
    return f"Killed process {pid} " in log or f"Killed process {pid}," in log


def run_game(args, seed: int, env: dict) -> dict:
    gdir = Path(args.out) / args.variant / f"seed-{seed:03d}"
    gdir.mkdir(parents=True, exist_ok=True)
    for name in ("moves.jsonl", "perply.jsonl", "result.json"):
        (gdir / name).unlink(missing_ok=True)
    cmd = [args.python, str(HERE / "play_one.py"), "--seed", str(seed),
           "--iters", str(args.iters), "--max-plies", str(args.max_plies),
           "--profile", args.profile, "--p-max", str(args.p_max),
           "--opp", args.opp, "--out-dir", str(gdir)]
    stderr = (gdir / "stderr.log").open("w")
    opp = None
    pass_fds: tuple = ()
    if args.opp == "remote":
        # The opponent is a sibling process on the ORIGINAL build with no fix
        # flags, so it is the same in every variant and its memory is not in
        # Misty's wait4() peak.
        opp_env = {k: v for k, v in env.items() if k not in FIX_FLAGS}
        opp_err = (gdir / "opp-stderr.log").open("w")
        opp = subprocess.Popen([args.opp_python, str(HERE / "opp_proc.py")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=opp_err, env=opp_env)
        rfd, wfd = opp.stdout.fileno(), opp.stdin.fileno()
        pass_fds = (rfd, wfd)
        cmd += ["--opp-fds", f"{rfd},{wfd}", "--opp-iters", str(args.opp_iters),
                "--opp-p-max", str(args.opp_p_max)]
    t0 = time.monotonic()
    base_used = vm_used_mb()
    proc = subprocess.Popen(cmd, env=env, stdout=stderr, stderr=stderr, pass_fds=pass_fds)
    peak_vm = base_used
    peak_rss = 0.0
    status, rusage, rc = None, None, None
    while True:
        pid, wstatus, ru = os.wait4(proc.pid, os.WNOHANG)
        if pid != 0:
            rusage = ru
            rc = os.waitstatus_to_exitcode(wstatus)
            proc.returncode = rc
            break
        peak_vm = max(peak_vm, vm_used_mb())
        hwm = proc_hwm_mb(proc.pid)
        if hwm is not None:
            peak_rss = max(peak_rss, hwm)
        if time.monotonic() - t0 > args.timeout:
            status = "timeout"
            proc.send_signal(signal.SIGKILL)
        time.sleep(0.2)
    wall = time.monotonic() - t0
    stderr.close()
    if opp is not None:
        opp.stdin.close()
        try:
            opp.wait(timeout=30)
        except subprocess.TimeoutExpired:
            opp.kill()
            opp.wait()
        opp.stdout.close()
    if status is None:
        if rc == 0:
            status = "ok"
        elif rc == -9 and oom_killed(proc.pid):
            status = "oom-killed"
        else:
            status = f"exit {rc}"
    moves = read_jsonl(gdir / "moves.jsonl")
    perply = read_jsonl(gdir / "perply.jsonl")
    mt = move_times(perply)
    obs = [r for r in perply if r.get("kind", "").startswith("observe")]
    res = json.loads((gdir / "result.json").read_text()) if (gdir / "result.json").exists() else {}
    row = {
        "variant": args.variant,
        "seed": seed,
        "status": status,
        "end_reason": res.get("end_reason"),
        "winner": res.get("winner"),
        "plies": len(moves),
        "wall_s": round(wall, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "peak_rss_tree_mb": round(rusage.ru_maxrss / 1024, 1) if rusage else None,
        "peak_vm_used_mb": round(peak_vm, 1),
        "vm_used_before_mb": round(base_used, 1),
        "misty_moves": len(mt),
        "mean_move_s": round(sum(m["ms"] for m in mt) / len(mt) / 1000, 2) if mt else None,
        "max_move_s": round(max(m["ms"] for m in mt) / 1000, 2) if mt else None,
        "max_p": max((r.get("p_post", 0) for r in obs), default=0),
        "max_m_est": max((r.get("p_pre_cap", 0) for r in obs), default=0),
        "cap_fires": sum(1 for r in obs if r.get("downsampled")),
        "moves": " ".join(m["uci"] for m in moves),
    }
    (gdir / "summary.json").write_text(json.dumps(row) + "\n")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--profile", default="v1.6")
    ap.add_argument("--p-max", type=int, default=16_000_000)
    ap.add_argument("--opp", default="remote")
    ap.add_argument("--opp-python", default=os.path.expanduser("~/venv/bin/python"),
                    help="interpreter of the ORIGINAL build, for the opponent process")
    ap.add_argument("--opp-iters", type=int, default=150)
    ap.add_argument("--opp-p-max", type=int, default=1_000_000)
    ap.add_argument("--timeout", type=float, default=2400)
    args = ap.parse_args()
    env = dict(os.environ)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = Path(args.out) / f"{args.variant}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        row = run_game(args, seed, env)
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        short = {k: v for k, v in row.items() if k != "moves"}
        print(json.dumps(short), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
