from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import random
from tqdm import trange

from .diffusion import DiffusionSchedule
from .priors import Prior


@dataclass(frozen=True)
class ScoreTrainingConfig:
    """Configuration for denoising score matching on a 2D toy prior."""

    batch_size: int = 2048
    learning_rate: float = 2e-4
    num_steps: int = 20000
    seed: int = 0
    record_loss: bool = True


def save_checkpoint(path: str | Path, params: dict) -> None:
    """Save Flax parameters (a PyTree/FrozenDict) to a msgpack file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))


def load_checkpoint(path: str | Path, params_template: dict) -> dict:
    """Load Flax parameters given a template with the same structure."""
    data = Path(path).read_bytes()
    return serialization.from_bytes(params_template, data)


def train_score_model(
    *,
    prior: Prior,
    model: object,
    schedule: DiffusionSchedule,
    config: ScoreTrainingConfig,
    init_key: Optional[jax.Array] = None,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """Train a score network s_theta(t, x) for the given prior.

    This follows the loss used in your notebooks:

        x0 ~ prior
        t ~ U(0,1)
        eps ~ N(0, I)
        x_{1-t} = alpha(1-t) x0 + sigma(1-t) eps
        loss = E || eps + sigma(1-t) * s_theta(1-t, x_{1-t}) ||^2

    Notes:
        - For a repo-quality demo, consider reducing hidden_dim and num_steps.
        - The returned `state.params` contains the full variable dict
          (e.g., {'params': ...}) matching how the notebooks used TrainState.
    """

    key = init_key if init_key is not None else random.PRNGKey(config.seed)
    key, init_k = random.split(key)

    # Dummy init inputs (shapes must match future calls)
    dummy_t = jnp.ones((config.batch_size, 1), dtype=jnp.float32)
    dummy_x = jnp.ones((config.batch_size, prior.dim), dtype=jnp.float32)

    variables = model.init(init_k, dummy_t, dummy_x)
    optimizer = optax.adam(learning_rate=config.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables, tx=optimizer)

    def loss_fn(params: dict, key: jax.Array) -> jax.Array:
        k1, k2, k3 = random.split(key, 3)
        x0 = prior.sample(k1, config.batch_size)
        t = random.uniform(k2, shape=(config.batch_size, 1))
        eps = random.normal(k3, shape=x0.shape)

        rev_t = 1.0 - t
        alpha = schedule.alpha(rev_t)
        sigma = schedule.sigma(rev_t)
        x_1_minus_t = alpha * x0 + sigma * eps

        pred = state.apply_fn(params, rev_t, x_1_minus_t)
        per_sample = jnp.sum((eps + sigma * pred) ** 2, axis=-1)
        return jnp.mean(per_sample)

    @jax.jit
    def train_step(state: train_state.TrainState, key: jax.Array) -> Tuple[train_state.TrainState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(state.params, key)
        state = state.apply_gradients(grads=grads)
        return state, loss

    losses = []
    for i in trange(config.num_steps):
        key, step_k = random.split(key)
        state, loss = train_step(state, step_k)
        if config.record_loss:
            losses.append(loss)

    return state, jnp.array(losses)
