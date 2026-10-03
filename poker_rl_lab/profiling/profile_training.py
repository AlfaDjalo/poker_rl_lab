"""
profiling/profile_training.py — Where does RL Lab training time go?

Usage (from the directory containing the poker_rl_lab package):

    python -m poker_rl_lab.profiling.profile_training \
        --scenario-name push_fold_duo --iterations 3 --episodes 256

    # Full hold'em, with micro-benchmarks and a cProfile dump too:
    python -m poker_rl_lab.profiling.profile_training \
        --scenario-name holdem_full_hand --iterations 2 --episodes 128 \
        --num-showdown-samples 1 --micro --cprofile

    # Other options: --variant, --device cuda, --json-out report.json

What it does
------------
1. Monkey-patches a fixed set of hot-path functions with nested timers
   (inclusive AND self/exclusive time, so percentages add up to 100%).
2. Runs one un-timed warm-up iteration (TensorFlow tracing, lazy imports).
3. Runs N timed training iterations.
4. Prints: phase split (rollout vs PPO update), ranked self-time table,
   and encoder-specific stats (rows per call, cache-hit potential).
5. Optional --micro: isolated throughput of the C++ scoring engine and
   the Keras hand encoder at various batch sizes.
6. Optional --cprofile: stock cProfile for anything the patch list misses.

Reading the output
------------------
"self" = time inside that section excluding timed sections nested in it.
Sum of all self times == wall time. The "TOTAL (unattributed)" row is
plain-Python overhead not covered by any patched section.
"""

from __future__ import annotations

import argparse
import copy as _copy
import functools
import json
import random
import time
import types
from collections import defaultdict
from typing import Callable, Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Nested timer
# ---------------------------------------------------------------------------

class Profiler:
    def __init__(self, cuda_sync: bool = False) -> None:
        self.enabled = False
        self.cuda_sync = cuda_sync
        self.stats: Dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0])  # calls, incl, self
        self.counters: Dict[str, float] = defaultdict(float)
        self._stack: list = []
        self.seen_card_sets: set = set()

    def now(self) -> float:
        if self.cuda_sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def wrap(self, fn: Callable, label: Optional[str] = None,
             label_fn: Optional[Callable] = None) -> Callable:
        prof = self

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not prof.enabled:
                return fn(*args, **kwargs)
            name = label_fn(*args) if label_fn else label
            frame = [0.0]  # child time
            prof._stack.append(frame)
            t0 = prof.now()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = prof.now() - t0
                prof._stack.pop()
                s = prof.stats[name]
                s[0] += 1
                s[1] += dt
                s[2] += dt - frame[0]
                if prof._stack:
                    prof._stack[-1][0] += dt

        return wrapper

    def reset(self) -> None:
        self.stats.clear()
        self.counters.clear()
        self._stack.clear()


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

_skipped: list = []


def _patch(prof: Profiler, obj, attr: str, label: str,
           label_fn: Optional[Callable] = None) -> None:
    orig = getattr(obj, attr, None)
    if orig is None:
        _skipped.append(f"{getattr(obj, '__name__', obj)}.{attr}")
        return
    setattr(obj, attr, prof.wrap(orig, label, label_fn))


def install_patches(prof: Profiler) -> None:
    import poker_rl_lab.trainers.graph_training_loop as gtl
    from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv
    from poker_rl_lab.representation.graph_observation_adapter import GraphObservationAdapter
    from poker_rl_lab.representation.representation_model import RepresentationModel
    from poker_rl_lab.representation.composite_hand_encoder import CompositeHandEncoder
    from poker_rl_lab.representation.composite_board_encoder import CompositeBoardEncoder
    from poker_rl_lab.representation.point_encoder import PointEncoder
    from poker_rl_lab.representation.game_encoder import GameEncoder
    from poker_rl_lab.representation.position_encoder import PositionEncoder
    from poker_rl_lab.representation.history_encoder import HistoryEncoder
    from poker_rl_lab.representation.stack_encoder import StackEncoder
    from poker_rl_lab.representation import pretrained_hand_encoder as phe
    from poker_rl_lab.models.policy_value_network import PolicyValueNetwork, PokerActorCritic
    from poker_rl_lab.models.shared_trunk_two_heads_policy import SharedTrunkTwoHeadsActorCritic
    from poker_rl_lab.models.shared_single_head_policy import SharedSingleHeadActorCritic
    from poker_rl_lab.models.independent_seat_policy import IndependentSeatActorCritic
    from poker_rl_lab.trainers.rollout_buffer import RolloutBuffer
    from poker_rl_lab.trainers.showdown_equity import HoldemEquityRewards

    # ---- Training-loop level ----------------------------------------
    _patch(prof, gtl, "collect_graph_rollouts", "loop: collect_graph_rollouts (incl.)")
    _patch(prof, gtl, "_ppo_update_seat", "loop: _ppo_update_seat (incl.)")
    _patch(prof, gtl, "_resample_showdown_rewards", "loop: MC showdown resample")
    _patch(prof, HoldemEquityRewards, "rewards", "showdown: HoldemEquityRewards.rewards (equity path, incl.)")
    try:
        import poker_engine.poker_eval as _pe
        if hasattr(_pe, "holdem_matchup_counts"):
            _pe.holdem_matchup_counts = prof.wrap(_pe.holdem_matchup_counts, "showdown: C++ holdem_matchup_counts")
    except ImportError:
        pass
    gtl.copy = types.SimpleNamespace(
        deepcopy=prof.wrap(_copy.deepcopy, "loop: copy.deepcopy (MC snapshots)")
    )
    _patch(prof, RolloutBuffer, "finish_episode", "loop: RolloutBuffer.finish_episode")
    _patch(prof, RolloutBuffer, "get", "loop: RolloutBuffer.get")

    # ---- Env / Engine -----------------------------------------------
    _patch(prof, GraphScenarioEnv, "reset", "env: reset")
    _patch(prof, GraphScenarioEnv, "step", "env: step")
    _patch(prof, GraphScenarioEnv, "_build_decision_obs", "env: _build_decision_obs")
    try:
        from poker_engine.graph.graph_engine import GraphEngine
        _patch(prof, GraphEngine, "start", "engine: GraphEngine.start")
        _patch(prof, GraphEngine, "submit_decision", "engine: GraphEngine.submit_decision")
    except ImportError:
        _skipped.append("GraphEngine")

    # Showdown scoring (C++ path)
    patched_scoring = False
    for modpath, cls in (
        ("poker_engine.scoring.scoring_engine", "CppScoringEngine"),
        ("poker_engine.scoring.mock_scoring_engine", "MockScoringEngine"),
    ):
        try:
            mod = __import__(modpath, fromlist=[cls])
            _patch(prof, getattr(mod, cls), "evaluate", f"showdown: {cls}.evaluate")
            patched_scoring = True
        except Exception:
            pass
    if not patched_scoring:
        _skipped.append("scoring_engine.evaluate")
    try:
        from poker_engine.showdown.showdown_resolver import ShowdownResolver
        for m in ("resolve", "resolve_showdown"):
            if hasattr(ShowdownResolver, m):
                _patch(prof, ShowdownResolver, m, f"showdown: ShowdownResolver.{m}")
    except ImportError:
        pass

    # ---- Adapter ----------------------------------------------------
    _patch(prof, GraphObservationAdapter, "adapt", "adapter: adapt (obs->RepresentationInput)")

    # ---- Policy entry points ----------------------------------------
    for cls in (SharedTrunkTwoHeadsActorCritic, SharedSingleHeadActorCritic,
                IndependentSeatActorCritic, PokerActorCritic):
        if "act_batch" in cls.__dict__:
            _patch(prof, cls, "act_batch", "policy: act_batch (rollout fwd, incl.)")
        if "evaluate_actions" in cls.__dict__:
            _patch(prof, cls, "evaluate_actions", "policy: evaluate_actions (PPO fwd, incl.)")
    _patch(prof, PolicyValueNetwork, "forward", "policy: PolicyValueNetwork.forward (heads)")

    # ---- Representation trunk ---------------------------------------
    _patch(prof, RepresentationModel, "forward_batch", "repr: RepresentationModel.forward_batch (incl.)")
    _patch(prof, CompositeHandEncoder, "forward_batch_torch", "repr: hole CompositeHandEncoder")
    _patch(prof, CompositeBoardEncoder, "forward_batch",
           "repr: board-texture encoder")
    _patch(prof, CompositeBoardEncoder, "forward_batch_made_hand",
           "repr: made-hand encoder")
    _patch(prof, CompositeBoardEncoder, "_encode_and_pool",
           "repr: CompositeBoardEncoder._encode_and_pool (subset pad+attn)")
    _patch(prof, PointEncoder, "forward_batch", "repr: PointEncoder")
    _patch(prof, GameEncoder, "forward_batch", "repr: GameEncoder")
    _patch(prof, PositionEncoder, "forward_batch", "repr: PositionEncoder")
    _patch(prof, HistoryEncoder, "forward_batch", "repr: HistoryEncoder (GRU)")
    _patch(prof, StackEncoder, "forward_batch", "repr: StackEncoder")

    # ---- Keras encoder ----------------------------------------------
    orig_encode = phe.PretrainedHandEncoder.encode_batch

    def encode_batch_counted(self, card_id_lists):
        if prof.enabled:
            prof.counters["keras_calls"] += 1
            prof.counters["keras_rows"] += len(card_id_lists)
            for ids in card_id_lists:
                key = tuple(sorted(ids))
                if key in prof.seen_card_sets:
                    prof.counters["keras_rows_seen_before"] += 1
                else:
                    prof.seen_card_sets.add(key)
        return orig_encode(self, card_id_lists)

    phe.PretrainedHandEncoder.encode_batch = prof.wrap(
        encode_batch_counted, "keras: PretrainedHandEncoder.encode_batch (grid+TF)"
    )
    phe.card_ids_to_grid = prof.wrap(phe.card_ids_to_grid, "keras: card_ids_to_grid (numpy)")

    # ---- Backward / optimizer ---------------------------------------
    torch.Tensor.backward = prof.wrap(torch.Tensor.backward, "ppo: loss.backward")
    torch.optim.Adam.step = prof.wrap(torch.optim.Adam.step, "ppo: Adam.step")
    torch.optim.Optimizer.zero_grad = prof.wrap(torch.optim.Optimizer.zero_grad, "ppo: zero_grad")
    torch.nn.utils.clip_grad_norm_ = prof.wrap(torch.nn.utils.clip_grad_norm_, "ppo: clip_grad_norm_")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(prof: Profiler, wall: float, iterations: int, episodes: int) -> dict:
    stats = {k: list(v) for k, v in prof.stats.items()}
    attributed_self = sum(v[2] for v in stats.values())
    stats["TOTAL (unattributed python)"] = [1, wall, max(wall - attributed_self, 0.0)]

    def pct(x): return 100.0 * x / wall if wall > 0 else 0.0

    print("\n" + "=" * 100)
    print(f"PROFILE: {iterations} iterations x {episodes} episodes  |  wall = {wall:.2f}s "
          f"({wall / iterations:.2f}s/iter, {iterations * episodes / wall:.1f} hands/s)")
    print("=" * 100)

    g = lambda k: stats.get(k, [0, 0.0, 0.0])[1]
    collect = g("loop: collect_graph_rollouts (incl.)")
    update = g("loop: _ppo_update_seat (incl.)")
    print("\nPHASE SPLIT (inclusive)")
    print(f"  Rollout collection : {collect:8.2f}s  {pct(collect):5.1f}%")
    print(f"  PPO updates        : {update:8.2f}s  {pct(update):5.1f}%")
    print(f"  Other              : {wall - collect - update:8.2f}s  {pct(wall - collect - update):5.1f}%")

    print("\nRANKED BY SELF TIME  (sums to 100% of wall)")
    print(f"  {'section':<66}{'calls':>8}{'self s':>9}{'self %':>8}{'incl s':>9}{'ms/call':>9}")
    for name, (calls, incl, self_t) in sorted(stats.items(), key=lambda kv: -kv[1][2]):
        if self_t < 1e-4 * wall:
            continue
        print(f"  {name:<66}{calls:>8}{self_t:>9.2f}{pct(self_t):>7.1f}%{incl:>9.2f}"
              f"{1000 * incl / max(calls, 1):>9.2f}")

    c = prof.counters
    if c.get("keras_calls"):
        rows, calls = c["keras_rows"], c["keras_calls"]
        print("\nKERAS ENCODER STATS")
        print(f"  encode_batch calls        : {int(calls)}")
        print(f"  total rows encoded        : {int(rows)}  (avg {rows / calls:.1f} rows/call)")
        print(f"  rows already seen before  : {100 * c['keras_rows_seen_before'] / rows:.1f}%  "
              f"<- upper bound on a card-set cache hit rate")
        print(f"  unique card sets          : {len(prof.seen_card_sets)}")
        ke = stats.get("keras: PretrainedHandEncoder.encode_batch (grid+TF)", [0, 0, 0])
        if rows:
            print(f"  cost per row              : {1e6 * ke[1] / rows:.1f} us (incl. grid build)")

    if _skipped:
        print(f"\n(note: patch targets not found, skipped: {', '.join(sorted(set(_skipped)))})")

    return {
        "wall_seconds": wall,
        "iterations": iterations,
        "episodes_per_iteration": episodes,
        "hands_per_second": iterations * episodes / wall,
        "sections": {k: {"calls": v[0], "inclusive_s": v[1], "self_s": v[2]} for k, v in stats.items()},
        "counters": dict(c),
    }


# ---------------------------------------------------------------------------
# Micro-benchmarks
# ---------------------------------------------------------------------------

def micro_scoring(n: int = 20000) -> None:
    print("\n" + "-" * 100 + "\nMICRO: showdown scoring engine (2 players, full 5-card board)")
    try:
        from poker_rl_lab.envs.engine_scoring_evaluator import _default_scoring_engine
        from poker_engine.poker_eval import ScoreType, ShowdownType
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped: {exc}")
        return
    engine = _default_scoring_engine()
    rng = random.Random(0)
    hands = []
    for _ in range(n):
        cards = rng.sample(range(52), 9)
        m = lambda cs: sum(1 << c for c in cs)
        hands.append((m(cards[:2]), m(cards[2:4]), m(cards[4:])))
    t0 = time.perf_counter()
    for a, b, board in hands:
        engine.evaluate([a, b], board, ScoreType.HIGH, ShowdownType.HOLDEM)
    dt = time.perf_counter() - t0
    print(f"  {type(engine).__name__}: {n} evals in {dt:.3f}s -> {1e6 * dt / n:.1f} us/eval "
          f"({n / dt:,.0f} evals/s)")
    print("  (per-call figure includes Python<->C++ boundary + ScoreResult construction)")


def micro_keras(batch_sizes=(1, 16, 64, 256, 1024)) -> None:
    print("\n" + "-" * 100 + "\nMICRO: Keras hand encoder throughput (2-card sets)")
    try:
        from poker_rl_lab.config.hand_encoder_config import load_default_hand_encoder
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped: {exc}")
        return
    enc = load_default_hand_encoder()
    rng = random.Random(0)
    enc.encode_batch([[0, 1]])  # warm
    print(f"  {'batch':>6}{'ms/call':>10}{'us/row':>10}{'rows/s':>12}")
    for b in batch_sizes:
        data = [rng.sample(range(52), 2) for _ in range(b)]
        enc.encode_batch(data)  # trace this bucket
        reps = max(3, 2000 // b)
        t0 = time.perf_counter()
        for _ in range(reps):
            enc.encode_batch(data)
        dt = (time.perf_counter() - t0) / reps
        print(f"  {b:>6}{1000 * dt:>10.2f}{1e6 * dt / b:>10.1f}{b / dt:>12,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Profile RL Lab graph training.")
    p.add_argument("--scenario-name", required=True)
    p.add_argument("--scenario-path", default=None)
    p.add_argument("--variant", default="separate_networks",
                   choices=["shared_trunk_two_heads", "shared_single_head", "separate_networks"])
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--num-showdown-samples", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=0, help="torch.set_num_threads (0 = default)")
    p.add_argument("--micro", action="store_true", help="Run isolated micro-benchmarks too.")
    p.add_argument("--cprofile", action="store_true", help="Also run stock cProfile pass.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario
    from poker_rl_lab.trainers.graph_training_loop import PolicyVariant, build_graph_experiment

    prof = Profiler(cuda_sync=args.device.startswith("cuda"))
    install_patches(prof)

    config = load_graph_scenario(args.scenario_name, path=args.scenario_path)
    env, actor_critic, adapter, loop = build_graph_experiment(
        config, variant=PolicyVariant(args.variant), device=args.device, seed=args.seed,
        num_showdown_samples=args.num_showdown_samples,
    )
    n_params = sum(p_.numel() for p_ in actor_critic.parameters())
    print(f"scenario={config.name} variant={args.variant} device={args.device} "
          f"params={n_params:,} showdown_samples={args.num_showdown_samples}")

    train_kwargs = dict(episodes_per_iteration=args.episodes, epochs=args.epochs,
                        minibatch_size=args.minibatch_size, verbose=False)

    print("warm-up iteration (untimed)...")
    loop.train(1, **train_kwargs)

    print(f"timed run: {args.iterations} iterations...")
    prof.reset()
    prof.seen_card_sets.clear()
    prof.enabled = True
    t0 = prof.now()
    loop.train(args.iterations, **train_kwargs)
    wall = prof.now() - t0
    prof.enabled = False

    result = report(prof, wall, args.iterations, args.episodes)
    eq = getattr(loop, "equity", None)
    if eq is not None and eq.supported:
        print(f"\nEQUITY PATH: {eq.used} showdowns handled, {eq.fallbacks} fell back to a single sample "
              f"(warm-up included)")

    if args.micro:
        micro_scoring()
        micro_keras()

    if args.cprofile:
        import cProfile
        import pstats
        print("\n" + "-" * 100 + "\ncProfile (1 iteration, top 35 by tottime)")
        pr = cProfile.Profile()
        pr.enable()
        loop.train(1, **train_kwargs)
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(35)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()