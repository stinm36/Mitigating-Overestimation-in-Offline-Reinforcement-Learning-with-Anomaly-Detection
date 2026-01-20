# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import gym
import d4rl


# -----------------------------
# Utils
# -----------------------------
def _get_bool_field(d, key, n):
    """Safely get boolean field from D4RL dict; if missing, return all-false."""
    if key in d and d[key] is not None:
        arr = d[key].astype(bool)
        if len(arr) != n:
            raise ValueError(f"Field '{key}' has length {len(arr)} != {n}")
        return arr
    return np.zeros(n, dtype=bool)


def load_d4rl_qlearning_dict(env_name: str):
    """
    Returns d4rl.qlearning_dataset(env) dict.
    Must include:
      observations, actions, rewards, next_observations, terminals
    May include:
      timeouts
    """
    env = gym.make(env_name)
    d = d4rl.qlearning_dataset(env)

    required = ["observations", "actions", "rewards", "next_observations", "terminals"]
    for k in required:
        if k not in d:
            raise KeyError(f"d4rl.qlearning_dataset missing key '{k}' for env={env_name}")
    return d


def stratified_subsample_indices(d: dict, fraction: float, seed: int, preserve_ratio: bool = True):
    """
    Transition-level non-replacement subsampling.

    Strata (for preserve_ratio=True):
      - mid: not terminal and not timeout
      - terminal: terminals == True
      - timeout: timeouts == True

    If preserve_ratio=False: uniform sample from all transitions.

    Returns:
      idx: shuffled np.int64 indices
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")

    n = d["observations"].shape[0]
    rng = np.random.default_rng(seed)

    if not preserve_ratio:
        k = int(round(n * fraction))
        idx = rng.choice(np.arange(n), size=k, replace=False)
        rng.shuffle(idx)
        return idx.astype(np.int64)

    terminals = _get_bool_field(d, "terminals", n)
    timeouts = _get_bool_field(d, "timeouts", n)

    mid_mask = (~terminals) & (~timeouts)
    term_mask = terminals
    tout_mask = timeouts

    def sample(mask):
        all_idx = np.where(mask)[0]
        k = int(round(len(all_idx) * fraction))
        if k <= 0:
            return np.array([], dtype=np.int64)
        return rng.choice(all_idx, size=k, replace=False).astype(np.int64)

    idx = np.concatenate([sample(mid_mask), sample(term_mask), sample(tout_mask)]).astype(np.int64)
    rng.shuffle(idx)
    return idx


# -----------------------------
# Dataset for AD modules (obs+act)
#   - You said you can keep your current D4RLDataset.
#   - This version is drop-in compatible but supports subsampling.
# -----------------------------
class D4RLDataset(Dataset):
    """
    Returns concat([obs, act]) like your current D4RLDataset,
    but supports transition-level subsampling.

    Use:
      dataset = D4RLDataset(env_name, fraction=0.1, seed=0)
      or reuse indices:
      dataset = D4RLDataset(env_name, indices=subsample_idx)
    """
    def __init__(
        self,
        env_name: str,
        fraction: float = 1.0,
        seed: int = 0,
        preserve_ratio: bool = True,
        indices=None,
    ):
        d = load_d4rl_qlearning_dict(env_name)
        n = d["observations"].shape[0]

        if indices is None:
            if fraction < 1.0:
                idx = stratified_subsample_indices(d, fraction, seed, preserve_ratio=preserve_ratio)
            else:
                idx = np.arange(n, dtype=np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64)

        obs = torch.tensor(d["observations"][idx], dtype=torch.float32)
        act = torch.tensor(d["actions"][idx], dtype=torch.float32)
        self.data = torch.cat([obs, act], dim=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return self.data[i]


# -----------------------------
# Optional: RLkit replay buffer prefill helper
#   - Call from your main script with minimal code changes.
# -----------------------------
def prefill_rlkit_replay_buffer_from_d4rl(
    replay_buffer,
    env_name: str,
    fraction: float = 1.0,
    seed: int = 0,
    preserve_ratio: bool = True,
    done_includes_timeouts: bool = False,
    verbose: bool = True,
):
    """
    Fill rlkit EnvReplayBuffer with a (subsampled) D4RL dataset via add_sample().

    Returns:
      idx: indices used to subsample transitions (so AD dataset can reuse it)

    Notes:
      - For CQL/SAC-style backups, it's usually safer to set done_includes_timeouts=False
        so timeouts do not cut bootstrap.
    """
    d = load_d4rl_qlearning_dict(env_name)
    n = d["observations"].shape[0]

    if fraction < 1.0:
        idx = stratified_subsample_indices(d, fraction, seed, preserve_ratio=preserve_ratio)
    else:
        idx = np.arange(n, dtype=np.int64)

    terminals = _get_bool_field(d, "terminals", n)[idx]
    timeouts = _get_bool_field(d, "timeouts", n)[idx]

    if done_includes_timeouts:
        dones = terminals | timeouts
    else:
        dones = terminals

    obs = d["observations"][idx]
    acts = d["actions"][idx]
    rews = d["rewards"][idx]
    next_obs = d["next_observations"][idx]

    for i in range(len(idx)):
        replay_buffer.add_sample(
            observation=obs[i],
            action=acts[i],
            reward=rews[i],
            terminal=float(dones[i]),
            next_observation=next_obs[i],
            env_info={"timeout": float(timeouts[i])},
        )

    if verbose:
        print(f"[prefill_rlkit_replay_buffer_from_d4rl] filled {len(idx)} transitions "
              f"(fraction={fraction}, seed={seed}, preserve_ratio={preserve_ratio}, done_includes_timeouts={done_includes_timeouts})")
        print(f"  terminal_ratio={terminals.mean():.4f}, timeout_ratio={timeouts.mean():.4f}")

    return idx