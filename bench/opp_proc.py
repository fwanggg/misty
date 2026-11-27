"""The sparring opponent in its OWN process, so the measured process (Misty,
bench/play_one.py) holds only one belief list.

Run it with the ORIGINAL build's interpreter: then the opponent is the same
engine in every variant, and only Misty's process changes.

Frames on stdin/stdout: 8-byte big-endian length + a pickle.
  ("init", {profile, seed, iterations, p_max})  -> "ok"
  ("reset", color)                              -> "ok"
  ("own", move, observation)                    -> "ok"
  ("opp", observation)                          -> "ok"
  ("pick", view)                                -> move
"""

from __future__ import annotations

import os
import pickle
import struct
import sys


def main() -> int:
    # Keep the protocol fd clean: anything printed goes to stderr.
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    os.environ["FOW_V2_CLOCK_TIME"] = "0"
    os.environ.setdefault("FOW_BOTTOMK_EXPANSION", "1")
    strat = None

    def reply(obj) -> None:
        data = pickle.dumps(obj)
        proto_out.write(struct.pack(">Q", len(data)) + data)
        proto_out.flush()

    while True:
        head = proto_in.read(8)
        if len(head) < 8:
            break
        (n,) = struct.unpack(">Q", head)
        cmd = pickle.loads(proto_in.read(n))
        op = cmd[0]
        if op == "init":
            from fow_chess import engine_profile
            kw = cmd[1]
            strat = engine_profile.PROFILES[kw["profile"]].build_strategy(
                seed=kw["seed"], time_budget_seconds=None,
                p_max_size=kw["p_max"], iterations=kw["iterations"],
            )
            reply("ok")
        elif op == "reset":
            strat.reset(cmd[1])
            reply("ok")
        elif op == "own":
            strat.observe_own_move(cmd[1], cmd[2])
            reply("ok")
        elif op == "opp":
            strat.observe_opp_move(cmd[1])
            reply("ok")
        elif op == "pick":
            reply(strat.pick_move(cmd[1]))
        else:
            reply(f"unknown op {op}")
    if strat is not None:
        strat.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
