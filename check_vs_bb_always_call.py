from experiments.push_fold_duo_experiment import build_push_fold_duo_experiment, load_checkpoint, PushFoldDuoExperimentConfig
from evaluation.push_fold_duo_baselines import (
    evaluate_duo_matchup, duo_seat_policy_from_actor_critic,
    bb_always_call, bb_always_fold, sb_always_push,
)

exp = build_push_fold_duo_experiment(PushFoldDuoExperimentConfig())
load_checkpoint(exp, "ckpt_duo.pt")
trained_sb = duo_seat_policy_from_actor_critic(exp.actor_critic, 0, exp.adapter)

r = evaluate_duo_matchup(exp.env, exp.scenario_gen, trained_sb, bb_always_call,
                          num_episodes=5000, big_blind=2, seed=1)
print(r.summary())