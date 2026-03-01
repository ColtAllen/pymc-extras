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

import pymc as pm
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("blackjax")

from pymc_extras.inference.mclmc.sampling import (
    sample_blackjax_adjusted_mclmc,
    sample_blackjax_mclmc,
)


def simple_gaussian_model():
    """A simple multivariate Gaussian for basic correctness checks."""
    with pm.Model() as model:
        x = pm.Normal("x", mu=0, sigma=1, shape=5)
        pm.Normal("obs", mu=x.sum(), sigma=0.1, observed=0.0)
    return model


def multivariable_model():
    """Model with multiple variables of different shapes."""
    with pm.Model() as model:
        x = pm.Normal("x", mu=1.0, sigma=0.5, shape=3)
        y = pm.Normal("y", mu=-1.0, sigma=0.5)
        pm.Normal("obs", mu=x.sum() + y, sigma=0.1, observed=0.0)
    return model


class TestSampleBlackjaxMCLMC:
    def test_simple_gaussian(self):
        model = simple_gaussian_model()
        draws = 500
        with model:
            idata = sample_blackjax_mclmc(draws=draws, tune=500, random_seed=42)

        assert "posterior" in idata.groups()
        assert idata.posterior["x"].shape == (1, draws, 5)

    def test_multivariable(self):
        model = multivariable_model()
        draws = 500
        with model:
            idata = sample_blackjax_mclmc(draws=draws, tune=500, random_seed=42)

        assert idata.posterior["x"].shape == (1, draws, 3)
        assert idata.posterior["y"].shape == (1, draws)

    def test_metadata_attrs(self):
        model = simple_gaussian_model()
        with model:
            idata = sample_blackjax_mclmc(draws=200, tune=200, random_seed=42)

        attrs = idata.posterior.attrs
        assert attrs["sampler"] == "BlackJAX MCLMC (unadjusted)"
        assert "L" in attrs
        assert "step_size" in attrs
        assert attrs["draws"] == 200
        assert attrs["tune"] == 200
        assert "running_time_seconds" in attrs
        assert attrs["running_time_seconds"] > 0

    def test_pretrained_params_skip_tuning(self):
        model = simple_gaussian_model()
        draws = 200
        with model:
            idata = sample_blackjax_mclmc(
                draws=draws,
                tune=0,
                L=3.0,
                step_size=0.5,
                random_seed=42,
            )

        assert idata.posterior["x"].shape == (1, draws, 5)
        assert idata.posterior.attrs["L"] == 3.0
        assert idata.posterior.attrs["step_size"] == 0.5

    def test_partial_params_raises(self):
        model = simple_gaussian_model()
        with model:
            with pytest.raises(ValueError, match="Either both"):
                sample_blackjax_mclmc(L=3.0, random_seed=42)

            with pytest.raises(ValueError, match="Either both"):
                sample_blackjax_mclmc(step_size=0.5, random_seed=42)

    def test_low_dim_raises(self):
        with pm.Model() as model:
            pm.Normal("x", mu=0, sigma=1)
            pm.Normal("obs", mu=0, sigma=1, observed=0.0)

        with model:
            with pytest.raises(ValueError, match="at least 2 parameters"):
                sample_blackjax_mclmc(random_seed=42)


class TestSampleBlackjaxAdjustedMCLMC:
    def test_simple_gaussian(self):
        model = simple_gaussian_model()
        draws = 500
        with model:
            idata = sample_blackjax_adjusted_mclmc(draws=draws, tune=500, random_seed=42)

        assert "posterior" in idata.groups()
        assert idata.posterior["x"].shape == (1, draws, 5)

    def test_metadata_attrs(self):
        model = simple_gaussian_model()
        with model:
            idata = sample_blackjax_adjusted_mclmc(draws=200, tune=200, random_seed=42)

        attrs = idata.posterior.attrs
        assert attrs["sampler"] == "BlackJAX MCLMC (adjusted)"
        assert "L" in attrs
        assert "step_size" in attrs
        assert attrs["target_acceptance_rate"] == 0.9
        assert "running_time_seconds" in attrs

    def test_pretrained_params_skip_tuning(self):
        model = simple_gaussian_model()
        draws = 200
        with model:
            idata = sample_blackjax_adjusted_mclmc(
                draws=draws,
                tune=0,
                L=3.0,
                step_size=0.5,
                random_seed=42,
            )

        assert idata.posterior["x"].shape == (1, draws, 5)
        assert idata.posterior.attrs["L"] == 3.0
        assert idata.posterior.attrs["step_size"] == 0.5

    def test_partial_params_raises(self):
        model = simple_gaussian_model()
        with model:
            with pytest.raises(ValueError, match="Either both"):
                sample_blackjax_adjusted_mclmc(L=3.0, random_seed=42)
