"""Dataclasses for parameters used in simulations for all schemes"""
from dataclasses import dataclass


@dataclass
class ModelParameters:
    start_price: float
    start_variance: float
    alpha: float
    rho: float
    eta: float

    def get_parameter_str(self):
        return f"S0: {self.start_price}, v0: {self.start_variance}, alpha: {self.alpha}, rho: {self.rho}, eta: {self.eta}"


@dataclass
class SchemeParameters:
    n_paths: int
    n_time_steps: int
    terminal_time: float



standard_model_parameters = ModelParameters(
    start_price=100,
    start_variance=round(0.235**2, 3),
    alpha=-0.43,
    rho=-0.9,
    eta=1.9
)

standard_scheme_parameters = SchemeParameters(
    n_paths=10000,
    n_time_steps=500,
    terminal_time=1.0
)

