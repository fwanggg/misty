"""Play ONE deterministic benchmark game: Misty (a named profile) vs a sparring
opponent, and stream everything to disk as it happens.

Deterministic on purpose: the per-move search is bounded by an ITERATION cap,
not by wall time (time_budget_seconds=None, FOW_V2_CLOCK_TIME=0), so the same
seed plays the same game on a fast or a slow build. That is what lets the
memory fixes be checked for byte-identical play.

Files in --out-dir (all flushed per line, so an OOM SIGKILL keeps the prefix):
  moves.jsonl   one line per ply: {"ply", "color", "uci"}
  perply.jsonl  Misty's own telemetry rows (EngineV2Strategy sink)
  result.json   written only when the game finishes normally

The parent (bench/run_bench.py) reads this process's peak RSS (VmHWM) from
/proc while it runs, so it has the number even when the kernel kills it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import sys
import time
from pathlib import Path


class RemoteStrategy:
    """play_game Strategy whose engine lives in bench/opp_proc.py."""

    def __init__(self, rfd: int, wfd: int) -> None:
        self._r = os.fdopen(rfd, "rb")
        self._w = os.fdopen(wfd, "wb")

    def call(self, *cmd):
        data = pickle.dumps(cmd)
        self._w.write(struct.pack(">Q", len(data)) + data)
        self._w.flush()
        head = self._r.read(8)
        if len(head) < 8:
            raise RuntimeError("opponent process closed")
        (n,) = struct.unpack(">Q", head)
        return pickle.loads(self._r.read(n))

    def reset(self, perspective) -> None:
        self.call("reset", perspective)

    def observe_own_move(self, move, observation) -> None:
        self.call("own", move, observation)

    def observe_opp_move(self, observation) -> None:
        self.call("opp", observation)

    def pick_move(self, view):
        return self.call("pick", view)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--profile", default="v1.6")
    ap.add_argument("--p-max", type=int, default=16_000_000)
    ap.add_argument("--opp", choices=("remote", "greedy", "random", "v2small"), default="remote")
    ap.add_argument("--opp-fds", default="", help="remote: 'read_fd,write_fd' of bench/opp_proc.py")
    ap.add_argument("--opp-profile", default="v1.6")
    ap.add_argument("--opp-iters", type=int, default=150)
    ap.add_argument("--opp-i", type=int, default=8)
    ap.add_argument("--opp-p-max", type=int, default=1_000_000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # Deterministic search: iterations bind, never the clock.
    os.environ["FOW_V2_CLOCK_TIME"] = "0"
    os.environ.setdefault("FOW_BOTTOMK_EXPANSION", "1")

    import chess
    from fow_chess import engine_profile
    from fow_chess.engine_v2 import EngineV2Strategy
    from fow_chess.selfplay import play_game
    from fow_chess.strategies import LegalGreedy, RandomStrategy

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    moves_fh = (out / "moves.jsonl").open("w")

    misty_color = chess.WHITE if args.seed % 2 == 0 else chess.BLACK
    prof = engine_profile.PROFILES[args.profile]
    misty = prof.build_strategy(
        seed=args.seed + 7,
        time_budget_seconds=None,
        p_max_size=args.p_max,
        iterations=args.iters,
        capture_telemetry=True,
    )
    misty.set_telemetry_sink(str(out / "perply.jsonl"))
    if args.opp == "remote":
        rfd, wfd = (int(x) for x in args.opp_fds.split(","))
        opp = RemoteStrategy(rfd, wfd)
        opp.call("init", {"profile": args.opp_profile, "seed": args.seed + 13,
                          "iterations": args.opp_iters, "p_max": args.opp_p_max})
    elif args.opp == "greedy":
        opp = LegalGreedy(seed=args.seed)
    elif args.opp == "random":
        opp = RandomStrategy(seed=args.seed)
    else:
        opp = EngineV2Strategy(
            seed=args.seed + 13,
            iterations=args.opp_iters,
            i_sample_size=args.opp_i,
            time_budget_seconds=None,
            p_max_size=args.opp_p_max,
        )

    def analyzer(prev: chess.Board, move: chess.Move, color: chess.Color) -> None:
        moves_fh.write(json.dumps({
            "ply": prev.ply() + 1,
            "color": "white" if color == chess.WHITE else "black",
            "uci": move.uci(),
            "t": round(time.monotonic(), 3),
        }) + "\n")
        moves_fh.flush()

    white = misty if misty_color == chess.WHITE else opp
    black = opp if misty_color == chess.WHITE else misty
    t0 = time.monotonic()
    try:
        result = play_game(
            white, black,
            max_plies=args.max_plies,
            room_id=f"bench-{args.seed}",
            seed=args.seed,
            analyzer=analyzer,
        )
    finally:
        moves_fh.close()
    wall = time.monotonic() - t0
    (out / "result.json").write_text(json.dumps({
        "seed": args.seed,
        "misty_color": "white" if misty_color == chess.WHITE else "black",
        "winner": result.winner,
        "end_reason": result.end_reason,
        "plies": result.plies,
        "wall_s": round(wall, 1),
    }) + "\n")
    misty.close()
    if hasattr(opp, "close") and not isinstance(opp, RemoteStrategy):
        opp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
