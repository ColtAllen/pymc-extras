#   Copyright 2024 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""BlackJAX MCLMC (Microcanonical Langevin Monte Carlo) sampling for PyMC models.

Provides two sampling functions:
- ``sample_blackjax_mclmc``: Unadjusted MCLMC (recommended for most use cases).
- ``sample_blackjax_adjusted_mclmc``: Adjusted MCLMC with Metropolis-Hastings correction.

References
----------
.. [1] Robnik, J., De Luca, G. B., Silverstein, E., & Seljak, U. (2023).
       Microcanonical Hamiltonian Monte Carlo. JMLR.
.. [2] Robnik, J. & Seljak, U. (2024). Microcanonical Langevin Monte Carlo. ArXiv.
.. [3] https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
"""

import logging
import time

import arviz as az
import blackjax
import blackjax.mcmc.integrators
import jax
import jax.numpy as jnp
import numpy as np

from blackjax.util import run_inference_algorithm
from pymc import modelcontext, to_inference_data
from pymc.backends import NDArray
from pymc.backends.base import MultiTrace
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.util import RandomState, _get_seeds_per_chain

from pymc_extras.inference.pathfinder.pathfinder import get_jaxified_logp_of_ravel_inputs

log = logging.getLogger(__name__)


def _get_initial_position(model, random_seed):
    """Get the initial position as a flat JAX array from the model's initial point.

    Uses DictToArrayBijection to ravel the model's initial point dict into a single
    flat array, matching the input expected by get_jaxified_logp_of_ravel_inputs.
    """
    ip = model.initial_point(random_seed=random_seed)
    raveled = DictToArrayBijection.map(ip)
    return jnp.array(raveled.data)


def _samples_to_inferencedata(model, flat_samples):
    """Convert flat (draws, dim) JAX array back to named variables and build InferenceData.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model, used to determine variable names and shapes.
    flat_samples : jax.Array
        Array of shape (draws, total_dim) containing the raveled samples.

    Returns
    -------
    az.InferenceData
    """
    ip = model.initial_point()
    point_map_info = DictToArrayBijection.map(ip).point_map_info

    draws = flat_samples.shape[0]

    with model:
        strace = NDArray(name=model.name)
        strace.setup(draws, 0)

    for i in range(draws):
        raveled = RaveledVars(np.asarray(flat_samples[i]), point_map_info)
        point = DictToArrayBijection.rmap(raveled, ip)
        strace.record(point={k: np.asarray(v) for k, v in point.items()})

    multitrace = MultiTrace((strace,))
    return to_inference_data(multitrace, log_likelihood=False)


def sample_blackjax_mclmc(
    draws: int = 1000,
    tune: int = 1000,
    random_seed: RandomState = None,
    L: float | None = None,
    step_size: float | None = None,
    desired_energy_var: float = 5e-4,
    diagonal_preconditioning: bool = True,
    frac_tune1: float = 0.1,
    frac_tune2: float = 0.1,
    frac_tune3: float = 0.1,
    model=None,
) -> az.InferenceData:
    """Sample from a PyMC model using BlackJAX's unadjusted MCLMC algorithm.

    Microcanonical Langevin Monte Carlo (MCLMC) is a gradient-based sampler that
    numerically integrates Langevin-like dynamics on an extended phase space. It is
    particularly efficient for high-dimensional targets. Note that this is an *unadjusted*
    method: the resulting samples have a small bias controlled by the step size. Use
    ``sample_blackjax_adjusted_mclmc`` if asymptotic exactness is required.

    Parameters
    ----------
    draws : int
        Number of samples to draw after tuning. Each draw corresponds to one
        integration step.
    tune : int
        Number of tuning steps used to find optimal ``L`` and ``step_size``.
        Ignored when both ``L`` and ``step_size`` are provided.
    random_seed : RandomState, optional
        Seed for the random number generator.
    L : float, optional
        Momentum decoherence length. If both ``L`` and ``step_size`` are provided,
        tuning is skipped entirely.
    step_size : float, optional
        Integration step size. If both ``L`` and ``step_size`` are provided,
        tuning is skipped entirely.
    desired_energy_var : float
        Target energy variance per dimension used during tuning.
    diagonal_preconditioning : bool
        Whether to estimate and use a diagonal mass matrix during tuning.
    frac_tune1 : float
        Fraction of ``tune`` steps for step-size adaptation.
    frac_tune2 : float
        Fraction of ``tune`` steps for L estimation.
    frac_tune3 : float
        Fraction of ``tune`` steps for L refinement via autocorrelation.
    model : pymc.Model, optional
        PyMC model. If ``None``, uses the current model context.

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData with a single chain of posterior samples. Sampler
        metadata (``L``, ``step_size``, timing, etc.) is stored in
        ``posterior.attrs``.

    Raises
    ------
    ValueError
        If exactly one of ``L`` / ``step_size`` is provided, or if the model has
        fewer than 2 parameters (MCLMC requires dimensionality >= 2).

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.inference.mclmc import sample_blackjax_mclmc

        with pm.Model() as model:
            x = pm.Normal("x", shape=10)
            obs = pm.Normal("obs", mu=x.sum(), sigma=1, observed=0.0)
            idata = sample_blackjax_mclmc(draws=5000, tune=2000)
    """
    model = modelcontext(model)
    random_seed = np.random.default_rng(seed=random_seed)

    if (L is None) != (step_size is None):
        raise ValueError(
            "Either both `L` and `step_size` must be provided to skip tuning, "
            "or neither (to run automatic tuning)."
        )

    key = jax.random.PRNGKey(_get_seeds_per_chain(random_seed, 1)[0])
    init_key, tune_key, run_key = jax.random.split(key, 3)

    logdensity_fn = get_jaxified_logp_of_ravel_inputs(model)
    initial_position = _get_initial_position(model, random_seed=random_seed.integers(2**30))

    dim = initial_position.shape[0]
    if dim < 2:
        raise ValueError(
            f"MCLMC requires at least 2 parameters, but model has {dim}. "
            "Consider using a different sampler."
        )

    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=init_key,
    )

    skip_tuning = L is not None and step_size is not None

    if skip_tuning:
        state_after_tuning = initial_state
        inverse_mass_matrix = 1.0
        log.info("Skipping MCLMC tuning: using provided L=%.4f, step_size=%.4f", L, step_size)
    else:

        def kernel(inverse_mass_matrix):
            return blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=logdensity_fn,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=inverse_mass_matrix,
            )

        (
            state_after_tuning,
            sampler_params,
            _,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=tune,
            state=initial_state,
            rng_key=tune_key,
            frac_tune1=frac_tune1,
            frac_tune2=frac_tune2,
            frac_tune3=frac_tune3,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=diagonal_preconditioning,
        )

        L = float(sampler_params.L)
        step_size = float(sampler_params.step_size)
        inverse_mass_matrix = sampler_params.inverse_mass_matrix
        log.info("MCLMC tuning complete: L=%.4f, step_size=%.4f", L, step_size)

    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=L,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    start = time.time()
    _, samples = run_inference_algorithm(
        rng_key=run_key,
        initial_state=state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=draws,
        transform=lambda state, info: state.position,
        progress_bar=True,
    )
    elapsed = time.time() - start

    inference_data = _samples_to_inferencedata(model, samples)

    inference_data.posterior.attrs.update(
        {
            "sampler": "BlackJAX MCLMC (unadjusted)",
            "L": float(L),
            "step_size": float(step_size),
            "draws": draws,
            "tune": tune,
            "desired_energy_var": desired_energy_var,
            "diagonal_preconditioning": diagonal_preconditioning,
            "running_time_seconds": elapsed,
        }
    )

    return inference_data


def sample_blackjax_adjusted_mclmc(
    draws: int = 1000,
    tune: int = 1000,
    random_seed: RandomState = None,
    L: float | None = None,
    step_size: float | None = None,
    target_acceptance_rate: float = 0.9,
    diagonal_preconditioning: bool = True,
    random_trajectory_length: bool = True,
    frac_tune1: float = 0.1,
    frac_tune2: float = 0.1,
    frac_tune3: float = 0.1,
    model=None,
) -> az.InferenceData:
    """Sample from a PyMC model using BlackJAX's adjusted MCLMC algorithm.

    This variant adds a Metropolis-Hastings correction step to MCLMC, making it
    asymptotically exact (unbiased). It is recommended when strict unbiasedness is
    required. For most practical purposes the unadjusted version
    (``sample_blackjax_mclmc``) is preferred due to lower computational cost.

    Parameters
    ----------
    draws : int
        Number of samples to draw after tuning.
    tune : int
        Number of tuning steps used to find optimal ``L`` and ``step_size``.
        Ignored when both ``L`` and ``step_size`` are provided.
    random_seed : RandomState, optional
        Seed for the random number generator.
    L : float, optional
        Momentum decoherence length. If both ``L`` and ``step_size`` are provided,
        tuning is skipped entirely.
    step_size : float, optional
        Integration step size. If both ``L`` and ``step_size`` are provided,
        tuning is skipped entirely.
    target_acceptance_rate : float
        Target Metropolis-Hastings acceptance rate for step-size tuning.
    diagonal_preconditioning : bool
        Whether to estimate and use a diagonal mass matrix during tuning.
    random_trajectory_length : bool
        Whether to randomise the number of integration steps per proposal.
        Recommended for better mixing.
    frac_tune1 : float
        Fraction of ``tune`` steps for step-size adaptation.
    frac_tune2 : float
        Fraction of ``tune`` steps for L estimation.
    frac_tune3 : float
        Fraction of ``tune`` steps for L refinement via autocorrelation.
    model : pymc.Model, optional
        PyMC model. If ``None``, uses the current model context.

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData with a single chain of posterior samples. Sampler
        metadata is stored in ``posterior.attrs``.

    Raises
    ------
    ValueError
        If exactly one of ``L`` / ``step_size`` is provided, or if the model has
        fewer than 2 parameters.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.inference.mclmc import sample_blackjax_adjusted_mclmc

        with pm.Model() as model:
            x = pm.Normal("x", shape=10)
            obs = pm.Normal("obs", mu=x.sum(), sigma=1, observed=0.0)
            idata = sample_blackjax_adjusted_mclmc(draws=5000, tune=2000)
    """
    from blackjax.mcmc.adjusted_mclmc_dynamic import rescale

    model = modelcontext(model)
    random_seed = np.random.default_rng(seed=random_seed)

    if (L is None) != (step_size is None):
        raise ValueError(
            "Either both `L` and `step_size` must be provided to skip tuning, "
            "or neither (to run automatic tuning)."
        )

    key = jax.random.PRNGKey(_get_seeds_per_chain(random_seed, 1)[0])
    init_key, tune_key, run_key = jax.random.split(key, 3)

    logdensity_fn = get_jaxified_logp_of_ravel_inputs(model)
    initial_position = _get_initial_position(model, random_seed=random_seed.integers(2**30))

    dim = initial_position.shape[0]
    if dim < 2:
        raise ValueError(
            f"MCLMC requires at least 2 parameters, but model has {dim}. "
            "Consider using a different sampler."
        )

    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )

    if random_trajectory_length:

        def integration_steps_fn(avg_num_integration_steps):
            return lambda k: jnp.ceil(jax.random.uniform(k) * rescale(avg_num_integration_steps))
    else:

        def integration_steps_fn(avg_num_integration_steps):
            return lambda _: jnp.ceil(avg_num_integration_steps)

    skip_tuning = L is not None and step_size is not None

    if skip_tuning:
        state_after_tuning = initial_state
        inverse_mass_matrix = jnp.ones((dim,))
        log.info(
            "Skipping adjusted MCLMC tuning: using provided L=%.4f, step_size=%.4f",
            L,
            step_size,
        )
    else:

        def kernel(rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix):
            return blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
                integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
                inverse_mass_matrix=inverse_mass_matrix,
            )(rng_key=rng_key, state=state, step_size=step_size, logdensity_fn=logdensity_fn)

        (
            state_after_tuning,
            sampler_params,
            _,
        ) = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=tune,
            state=initial_state,
            rng_key=tune_key,
            target=target_acceptance_rate,
            frac_tune1=frac_tune1,
            frac_tune2=frac_tune2,
            frac_tune3=frac_tune3,
            diagonal_preconditioning=diagonal_preconditioning,
        )

        L = float(sampler_params.L)
        step_size = float(sampler_params.step_size)
        inverse_mass_matrix = sampler_params.inverse_mass_matrix
        log.info("Adjusted MCLMC tuning complete: L=%.4f, step_size=%.4f", L, step_size)

    sampling_alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(jax.random.uniform(key) * rescale(L / step_size))
        if random_trajectory_length
        else lambda _: jnp.ceil(L / step_size),
        inverse_mass_matrix=inverse_mass_matrix,
    )

    start = time.time()
    _, samples = run_inference_algorithm(
        rng_key=run_key,
        initial_state=state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=draws,
        transform=lambda state, info: state.position,
        progress_bar=True,
    )
    elapsed = time.time() - start

    inference_data = _samples_to_inferencedata(model, samples)

    inference_data.posterior.attrs.update(
        {
            "sampler": "BlackJAX MCLMC (adjusted)",
            "L": float(L),
            "step_size": float(step_size),
            "draws": draws,
            "tune": tune,
            "target_acceptance_rate": target_acceptance_rate,
            "diagonal_preconditioning": diagonal_preconditioning,
            "random_trajectory_length": random_trajectory_length,
            "running_time_seconds": elapsed,
        }
    )

    return inference_data
