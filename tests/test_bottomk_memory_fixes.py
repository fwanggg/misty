"""The bottom-K memory fixes must keep EXACTLY the same belief as the original.

Each fix is an env flag that defaults to the original behaviour. For every
flag set in MODES this test runs the same belief updates and requires, against
the original: the same worlds IN THE SAME ORDER (get_by_index feeds the seeded
root sampling, so order is part of determinism), and the same raw count,
pre-cap estimate and downsample flag.

Fixtures:
  * one large opp-move expansion (random game seed 6: ~1.2K predecessors ->
    ~205K consistent worlds), at caps that fire the periodic compaction many
    times, at a cap above M, and at the degenerate caps 0 and 1;
  * whole random games through PEnumerator at a small cap, own and opp moves.
"""
from __future__ import annotations

import random

import chess
import pytest

pytest.importorskip("fow_rust")
import fow_rust

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import PEnumerator
from fow_chess.p_enum.enumerator import _obs_piece_bitmasks

FIX_FLAGS = ("FOW_BOTTOMK_LOW_WATERMARK", "FOW_BOTTOMK_APPEND", "FOW_BOTTOMK_APPEND_CHUNK",
             "FOW_PSTATE_COMPACT", "FOW_ASSERT_OWN_SHARED")

MODES = {
    "original": {},
    "fix1-low-watermark": {"FOW_BOTTOMK_LOW_WATERMARK": "1"},
    "fix2-append": {"FOW_BOTTOMK_LOW_WATERMARK": "1", "FOW_BOTTOMK_APPEND": "1"},
    # Tiny chunks: many intermediate compactions even on small fixtures.
    "fix2-append-chunk7": {"FOW_BOTTOMK_APPEND": "1", "FOW_BOTTOMK_APPEND_CHUNK": "7"},
}


def _set_mode(monkeypatch, mode: str) -> None:
    for name in FIX_FLAGS:
        monkeypatch.delenv(name, raising=False)
    for name, value in MODES[mode].items():
        monkeypatch.setenv(name, value)


def _capture_large_M(seed: int, steps: int):
    rng = random.Random(seed)
    board = chess.Board()
    pe = PEnumerator(chess.WHITE, use_rust_state=True)
    best = None
    for _ in range(steps):
        ms = list(board.legal_moves)
        if not ms:
            break
        mv = rng.choice(ms)
        prev = board.copy()
        board.push(mv)
        obs = observation_from_transition(prev, board, chess.WHITE)
        if prev.turn == chess.WHITE:
            pe.update_own_move(mv, obs)
        else:
            before = sorted(pe.positions)
            pe.update_opp_move(obs)
            if best is None or pe.size > best[2]:
                best = (before, obs, pe.size)
    assert best is not None and best[2] > 100_000, "fixture did not produce a large M"
    return best[0], best[1], best[2]


@pytest.fixture(scope="module")
def scenario():
    return _capture_large_M(seed=6, steps=12)


def _run(fens, obs, cap):
    w, b = _obs_piece_bitmasks(obs)
    own = -1 if obs.own_capture_square is None else int(obs.own_capture_square)
    opp = -1 if obs.opp_capture_landing_square is None else int(obs.opp_capture_landing_square)
    ps = fow_rust.PEnumState(fens)
    if cap is not None:
        ps.set_bottomk_cap(cap)
    sz = ps.update_opp_move(
        False, True, int(obs.visibility_mask),
        w[0], w[1], w[2], w[3], w[4], w[5],
        b[0], b[1], b[2], b[3], b[4], b[5], own, opp,
    )
    return (sz, ps.all_positions(), ps.last_raw_count, ps.last_pre_cap_count,
            ps.last_was_downsampled)


@pytest.mark.parametrize("mode", [m for m in MODES if m != "original"])
@pytest.mark.parametrize("cap_div", [32, 8, 2, 0])
def test_one_expansion_same_as_original(scenario, monkeypatch, mode, cap_div):
    fens, obs, m = scenario
    cap = m // cap_div if cap_div else m + 100
    _set_mode(monkeypatch, "original")
    want = _run(fens, obs, cap)
    _set_mode(monkeypatch, mode)
    got = _run(fens, obs, cap)
    assert got[0] == want[0]
    assert got[1] == want[1], "different worlds or different order"
    assert got[2:] == want[2:], "raw / pre_cap / downsampled differ"
    assert want[4] is (cap < m)


@pytest.mark.parametrize("mode", [m for m in MODES if m != "original"])
@pytest.mark.parametrize("cap", [0, 1])
def test_degenerate_caps_same_as_original(scenario, monkeypatch, mode, cap):
    fens, obs, _ = scenario
    _set_mode(monkeypatch, "original")
    want = _run(fens, obs, cap)
    _set_mode(monkeypatch, mode)
    assert _run(fens, obs, cap) == want


def _game_trace(seed: int, cap: int, plies: int, perspective: chess.Color):
    """Belief after every update of a random game, at a small bottom-K cap."""
    rng = random.Random(seed)
    board = chess.Board()
    pe = PEnumerator(perspective, use_rust_state=True, max_size=cap,
                     rng=random.Random(seed))
    trace = []
    for _ in range(plies):
        ms = list(board.legal_moves)
        if not ms or board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            break
        mv = rng.choice(ms)
        prev = board.copy()
        board.push(mv)
        obs = observation_from_transition(prev, board, perspective)
        try:
            if prev.turn == perspective:
                pe.update_own_move(mv, obs)
            else:
                pe.update_opp_move(obs)
        except RuntimeError as exc:  # the cap dropped the truth: same in every mode
            trace.append(("empty", str(exc)))
            break
        trace.append((pe.size, pe._pstate.all_positions(), pe.last_raw_count,
                      pe.last_pre_cap_count, pe.last_was_downsampled,
                      pe._pstate.get_by_index([0, pe.size // 2, pe.size - 1])))
    return trace


@pytest.mark.parametrize("mode", [m for m in MODES if m != "original"])
@pytest.mark.parametrize("seed,color", [(6, chess.WHITE), (9, chess.WHITE), (10, chess.BLACK)])
def test_whole_game_same_as_original(monkeypatch, mode, seed, color):
    monkeypatch.setenv("FOW_BOTTOMK_EXPANSION", "1")
    _set_mode(monkeypatch, "original")
    want = _game_trace(seed, cap=3000, plies=24, perspective=color)
    assert any(t[4] for t in want), "fixture never hit the cap"
    _set_mode(monkeypatch, mode)
    got = _game_trace(seed, cap=3000, plies=24, perspective=color)
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"update {i} differs"
