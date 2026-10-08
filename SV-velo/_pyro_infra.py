"""Local Pyro infrastructure to avoid hard dependency on cell2fate internals."""

from __future__ import annotations

from functools import partial
from typing import Optional

import pyro
from pyro.infer.autoguide import AutoNormal, init_to_feasible, init_to_mean
from scvi.module.base import PyroBaseModuleClass
from scvi.train import PyroTrainingPlan

CTF_MAX_EPOCHS = 4000
CTF_START_LR = 0.01
CTF_FINAL_LR = 0.001
CTF_LRD = (CTF_FINAL_LR / CTF_START_LR) ** (1 / CTF_MAX_EPOCHS)



def init_to_value(site=None, values=None):
    if values is None:
        values = {}
    if site is None:
        return partial(init_to_value, values=values)
    if site["name"] in values:
        value = values[site["name"]]
        reference = init_to_mean(site, fallback=init_to_feasible)
        return value.to(device=reference.device, dtype=reference.dtype)
    return init_to_mean(site, fallback=init_to_feasible)


class LocalPyroBaseModule(PyroBaseModuleClass):
    """Wrap a Pyro model class into a scvi-compatible Pyro module."""

    def __init__(self, model, init_vals: Optional[dict] = None, **kwargs):
        super().__init__()
        self._model = model(**kwargs)
        if init_vals is None:
            init_vals = {}
        self._guide = AutoNormal(
            self.model,
            init_loc_fn=init_to_value(values=init_vals),
            create_plates=self.model.create_plates,
        )
        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @property
    def list_obs_plate_vars(self):
        return self.model.list_obs_plate_vars()


class PyroTrainingPlanClippedAdamDecayingRate(PyroTrainingPlan):
    """Cell2fate-style clipped Adam with decaying learning rate."""

    def __init__(
        self,
        pyro_module: PyroBaseModuleClass,
        loss_fn=None,
        optim=None,
        optim_kwargs: Optional[dict] = None,
    ):
        if optim is None:
            optim = pyro.optim.ClippedAdam({"lr": CTF_START_LR, "lrd": CTF_LRD, "clip_norm": 10.0} | (optim_kwargs or {}))
            optim_kwargs = None
        super().__init__(
            pyro_module=pyro_module,
            loss_fn=loss_fn,
            optim=optim,
            optim_kwargs=optim_kwargs,
        )
        self.svi = pyro.infer.SVI(
            model=pyro_module.model,
            guide=pyro_module.guide,
            optim=self.optim,
            loss=self.loss_fn,
        )
