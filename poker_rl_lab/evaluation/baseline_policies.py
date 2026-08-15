"""
evaluation/baseline_policies.py — Fixed, non-learning opponents/baselines
for benchmarking a trained policy.

Per RL_lab_implementation_guide.md (§10, "Evaluation"):

    Policies should be benchmarked against:
    - Random
    - Rule Based
    - Previous Checkpoints
    - Current Best

"Random" already exists: envs.base_env.RandomHeroPolicy. "Previous
Checkpoints" / "Current Best" are just earlier/later trained
actor_critics run through evaluation.policy_evaluator.policy_from_actor_critic
— nothing new needed here for those. This module adds the "Rule Based"
baselines.

Every policy here implements the same `Callable[[StructuredObservation],
AbstractAction]` contract as envs.base_env.run_episode's hero_policy
parameter, so any of them — and any actor_critic-backed policy from
policy_evaluator.policy_from_actor_critic — plugs into run_episode() /
evaluate_policy() / compare_policies() interchangeably. None of these
policies know anything about a specific environment (push-fold vs.
river vs. a future variant): they only read obs.legal_mask, so the same
baseline works unchanged across every curriculum stage — it just picks
from whatever happens to be legal there.

Dependencies
------------
    actions.abstract_action, envs.observations. No torch, no engine
    rules, no poker_eval.
"""

from __future__ import annotations

from typing import Callable, Sequence

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.envs.observations import StructuredObservation

HeroPolicy = Callable[[StructuredObservation], AbstractAction]


def preference_policy(preferred_order: Sequence[ActionType]) -> HeroPolicy:
    """
    Build a policy that always takes the first legal action in
    `preferred_order`, falling back to the first legal action overall
    if none of the preferred actions are legal in the current state.

    Parameters
    ----------
    preferred_order : Sequence[ActionType]
        Actions tried in priority order; the first one that's legal in
        the current observation is taken.

    Returns
    -------
    HeroPolicy
    """
    def policy(obs: StructuredObservation) -> AbstractAction:
        for action_type in preferred_order:
            if obs.legal_mask[action_type]:
                return AbstractAction(action_type)

        # Fallback: shouldn't normally trigger (FOLD is legal at every
        # decision point in every environment built so far), but guards
        # against a future environment whose preferred actions are
        # never legal, rather than raising for a solvable state.
        for i, legal in enumerate(obs.legal_mask):
            if legal:
                return AbstractAction(ActionType(i))

        raise RuntimeError(
            "No legal actions available — this policy should never be "
            "called on a terminal observation."
        )

    return policy


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------

#: Always folds when fold is legal (it always is). The maximally
#: passive/losing baseline — an agent that can't beat this consistently
#: has a real problem; it's the floor, not a target.
always_fold_policy: HeroPolicy = preference_policy([ActionType.FOLD])

#: Never voluntarily folds: calls any bet (including going all-in for
#: less when that's the only way to continue), checks when free. The
#: classic "calling station" baseline — useful for checking that an
#: agent's value bets and bluffs actually show a profit against a
#: villain who never gives up a hand.
calling_station_policy: HeroPolicy = preference_policy(
    [ActionType.CALL, ActionType.CHECK, ActionType.ALL_IN]
)