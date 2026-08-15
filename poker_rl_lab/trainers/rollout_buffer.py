"""
trainers/rollout_buffer.py — Episode rollout storage + GAE computation for PPO.

Sits between run_episode() (envs/base_env.py) and the PPO update step.
A training loop calls start_episode()/add()/finish_episode() once per
hero decision inside one hand, across many hands, then get() to pull out
minibatch-ready tensors.

Why GAE per-episode rather than one long fixed-length rollout
------------------------------------------------------------------
Standard PPO implementations (e.g. Atari) collect a fixed number of
*steps* across possibly-truncated episodes and bootstrap with V(s_last)
at the cut point. Poker hands here are always played to a true terminal
state (fold or showdown) — PushFoldEnv/RiverEnv never truncate — so
there is never a real bootstrap value to estimate; last_value is always
0.0 in practice (exposed as a parameter only for future-proofing, e.g.
if a max-street cutoff is ever introduced). Advantages are computed
episode-by-episode instead of over one flat multi-episode buffer so a
GAE trace never accidentally bleeds backward across an episode boundary
(done=True already prevents this via next_non_terminal, but per-episode
computation makes that invariant structural rather than relying on the
done flag alone).

Reward shape this is built for
----------------------------------
Per envs/base_env.py's BasePokerEnv contract and push_fold_env.py /
river_env.py's docstrings: intermediate steps return reward=0.0, and
the entire hand's chip outcome lands on the final (done=True) step. So
within one hand, GAE is doing real work only when a hand has more than
one hero decision (river betting with a raise-reraise back-and-forth);
push-fold hands (always exactly one hero decision) reduce to
advantage = reward - value, return = reward.

Dependencies
------------
    PyTorch (just for the tensors returned by get()).
    representation.representation_model.RepresentationInput — stored
    per-transition so a later PPO update can recompute log-probs/values
    under the *current* policy via PokerActorCritic.evaluate_actions().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from poker_rl_lab.representation.representation_model import RepresentationInput


@dataclass
class Transition:
    """One hero decision point recorded during rollout collection."""
    obs: RepresentationInput
    legal_mask: Tuple[bool, ...]
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    """
    Accumulates Transitions across many episodes, then computes
    per-episode GAE advantages and discounted returns.

    Usage
    -----
        buf = RolloutBuffer(gamma=1.0, gae_lambda=0.95)

        for _ in range(num_episodes):
            buf.start_episode()
            obs = env.reset(scenario_generator.generate())
            while not obs.is_terminal:
                rep_input = adapter.adapt(obs)
                result = actor_critic.act(rep_input, obs.legal_mask)
                next_obs, reward, done, info = env.step(
                    AbstractAction(ActionType(result.action_type))
                )
                buf.add(
                    rep_input, obs.legal_mask, result.action_type,
                    result.log_prob.item(), result.value.item(),
                    reward, done,
                )
                obs = next_obs
            buf.finish_episode()

        batch = buf.get()
    """

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95) -> None:
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._transitions: List[Transition] = []
        self._episode_start: int = 0

        self.advantages: List[float] = []
        self.returns: List[float] = []

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """Mark the beginning of a new episode. Call before the first add()."""
        self._episode_start = len(self._transitions)

    def add(
        self,
        obs: RepresentationInput,
        legal_mask: Sequence[bool],
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """Record one hero decision. Must be called between start_episode()
        and the matching finish_episode()."""
        self._transitions.append(
            Transition(
                obs=obs,
                legal_mask=tuple(legal_mask),
                action=int(action),
                log_prob=float(log_prob),
                value=float(value),
                reward=float(reward),
                done=bool(done),
            )
        )

    def finish_episode(self, last_value: float = 0.0) -> None:
        """
        Compute GAE advantages + returns for the episode just added
        (everything since the last start_episode()) and append them to
        self.advantages / self.returns, aligned index-for-index with
        self._transitions[self._episode_start:].

        Parameters
        ----------
        last_value : float
            Bootstrap value for the (hypothetical) state after the final
            transition. See module docstring — always 0.0 for a
            complete hand; kept as a parameter for future truncated
            rollouts.

        Raises
        ------
        RuntimeError
            If called with no transitions recorded since start_episode()
            (an empty episode is almost certainly a caller bug).
        """
        ep = self._transitions[self._episode_start:]
        n = len(ep)
        if n == 0:
            raise RuntimeError(
                "finish_episode() called with zero transitions recorded "
                "since start_episode()."
            )
        if not ep[-1].done:
            raise RuntimeError(
                "finish_episode() called but the last transition has "
                "done=False. Every hand must end with a terminal step."
            )

        advantages = [0.0] * n
        gae = 0.0

        for t in reversed(range(n)):
            next_value = ep[t + 1].value if t + 1 < n else last_value
            next_non_terminal = 0.0 if ep[t].done else 1.0
            delta = (
                ep[t].reward
                + self.gamma * next_value * next_non_terminal
                - ep[t].value
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = [advantages[t] + ep[t].value for t in range(n)]

        self.advantages.extend(advantages)
        self.returns.extend(returns)

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._transitions)

    def clear(self) -> None:
        """Empty the buffer. Call after each PPO update (on-policy)."""
        self._transitions.clear()
        self.advantages.clear()
        self.returns.clear()
        self._episode_start = 0

    def get(self, normalize_advantages: bool = True) -> Dict[str, object]:
        """
        Return the full buffer as minibatch-ready tensors.

        Parameters
        ----------
        normalize_advantages : bool
            Standard PPO trick: normalise advantages to zero mean / unit
            std across the whole buffer before use. Default True.
            Skipped (left as raw GAE values) when the buffer has 0 or 1
            transitions, since a std over a single value is degenerate.

        Returns
        -------
        dict with keys:
            obs           : List[RepresentationInput]   (length N)
            legal_masks   : List[Tuple[bool, ...]]       (length N)
            actions       : torch.LongTensor             (N,)
            old_log_probs : torch.FloatTensor            (N,)
            values        : torch.FloatTensor            (N,)
            advantages    : torch.FloatTensor            (N,)
            returns       : torch.FloatTensor            (N,)

        Raises
        ------
        RuntimeError
            If any recorded transition's episode never had
            finish_episode() called (advantages/returns length mismatch).
        """
        if len(self.advantages) != len(self._transitions):
            raise RuntimeError(
                "Advantages not computed for all transitions — did you "
                "forget to call finish_episode() after the last episode?"
            )

        advantages = torch.tensor(self.advantages, dtype=torch.float32)
        if normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs": [t.obs for t in self._transitions],
            "legal_masks": [t.legal_mask for t in self._transitions],
            "actions": torch.tensor(
                [t.action for t in self._transitions], dtype=torch.long
            ),
            "old_log_probs": torch.tensor(
                [t.log_prob for t in self._transitions], dtype=torch.float32
            ),
            "values": torch.tensor(
                [t.value for t in self._transitions], dtype=torch.float32
            ),
            "advantages": advantages,
            "returns": torch.tensor(self.returns, dtype=torch.float32),
        }