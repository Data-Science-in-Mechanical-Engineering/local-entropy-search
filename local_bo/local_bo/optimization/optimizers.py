from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List

from abc import ABC, abstractmethod

import numpy as np
import torch
import gpytorch
import botorch
from botorch.models import SingleTaskGP

from src.model import DerivativeExactGPSEModel
#from src.environment_api import EnvironmentObjective
from src.acquisition_function import GradientInformation
from src.model import ExactGPSEModel, DerivativeExactGPSEModel


class AbstractOptimizer(ABC):
    """Abstract optimizer class.

    Sets a default optimizer interface.

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        param_args_ignore: Which parameters should not be optimized.
        optimizer_config: Configuration file for the optimizer.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor]],
        param_args_ignore: List[int] = None,
        **optimizer_config: Dict,
    ):
        """Inits the abstract optimizer."""
        # Optionally add batchsize to parameters.
        if len(params_init.shape) == 1:
            params_init = params_init.reshape(1, -1)
        self.params = params_init.clone()
        self.param_args_ignore = param_args_ignore
        self.objective = objective

    def __call__(self):
        """Call method of optimizers."""
        self.step()

    @abstractmethod
    def step(self) -> None:
        """One parameter update step."""
        pass

class CMAES(AbstractOptimizer):
    """CMA-ES: Evolutionary Strategy with Covariance Matrix Adaptation for
    nonlinear function optimization.

    Inspired by the matlab code of https://arxiv.org/abs/1604.00772.
    Hence this function does not implement negative weights, that is, w_i = 0 for i > mu.

    Attributes:
        params_init: Objective parameters initial value.
        objective: Objective function.
        sigma: Coordinate wise standard deviation (step-size).
        maximization: True if objective function is maximized, False if minimized.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Union[Callable[[torch.Tensor], torch.Tensor]],
        sigma: float = 0.5,
        maximization: bool = True,
        verbose: bool = True,
        bound_to_unit_cube: bool = False,
        
    ):
        """Inits CMA-ES optimizer."""
        super(CMAES, self).__init__(params_init, objective)

        
        
        self.dim = self.params.shape[-1]
        
        self.xmean = self.params.clone().view(-1)
        self.params_history_list = [torch.unsqueeze(self.xmean.clone(),0)]
        self.maximization = maximization
        self.sigma = sigma

        # Strategy parameter setting: Selection.
        self.lambda_ = 4 + int(
            np.floor(3 * np.log(self.dim))
        )  # Population size, offspring number.
        self.mu = self.lambda_ // 2  # Number of parents/points for recombination.
        weights = np.log(self.mu + 0.5) - np.log(range(1, self.mu + 1))
        self.weights = torch.tensor(
            weights / sum(weights), dtype=torch.float32
        )  # Normalize recombination weights array.
        self.mueff = sum(self.weights) ** 2 / sum(
            self.weights ** 2
        )  # Variance-effective size of mu.

        # Strategy parameter setting: Adaption.
        self.cc = (4 + self.mueff / self.dim) / (
            self.dim + 4 + 2 * self.mueff / self.dim
        )  # Time constant for cumulation for C.
        self.cs = (self.mueff + 2) / (
            self.dim + self.mueff + 5
        )  # Time constant for cumulation for sigma-/step size control.
        self.c1 = 2 / (
            (self.dim + 1.3) ** 2 + self.mueff
        )  # Learning rate for rank-one update of C.
        self.cmu = (
            2
            * (self.mueff - 2 + 1 / self.mueff)
            / ((self.dim + 2) ** 2 + 2 * self.mueff / 2)
        )  # Learning rate for rank-mu update.
        self.damps = (
            1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        )  # Damping for sigma.

        # Initialize dynamic (internal) strategy parameters and constant.
        self.ps = torch.zeros(self.dim)  # Evolution path for sigma.
        self.pc = torch.zeros(self.dim)  # Evolution path for C.
        self.B = torch.eye(self.dim)
        self.D = torch.eye(
            self.dim
        )  # Eigendecomposition of C (pos. def.): B defines the coordinate system, diagonal matrix D the scaling.
        self.C = self.B @ self.D ** 2 @ self.D.transpose(0, 1)  # Covariance matrix.
        self.eigeneval = 0  # B and D updated at counteval == 0
        self.chiN = self.dim ** 0.5 * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2)
        )  # Expectation of ||N(0,I)|| == norm(randn(N,1))

        # Generation Loop.
        self.arz = torch.empty((self.dim, self.lambda_))
        self.arx = torch.empty((self.dim, self.lambda_))
        self.arfitness = torch.empty((self.lambda_))
        self.counteval = 0
        self.hs = 0
        self.objective_history_list = [self.objective(self.params)[0]]
        self.verbose = verbose
        self.bound_to_unit_cube = bound_to_unit_cube

    def step(self):

        # 1. Sampling and evaluating.
        for k in range(self.lambda_):
            # Reparameterization trick for samples.
            self.arz[:, k] = torch.randn(
                (self.dim)
            )  # Standard normally distributed vector.
            #if self.bound_to_unit_cube:
            #    self.arx[:, k] = torch.max(torch.min((
            #         self.xmean + self.sigma * self.B @ self.D @ self.arz[:, k]
            #    ),torch.ones_like(self.arx[:, k])),torch.zeros_like(self.arx[:, k]))  # Add mutation.
            #else:
            self.arx[:, k] = self.xmean + self.sigma * self.B @ self.D @ self.arz[:, k]
                
            self.arfitness[k] = self.objective(self.arx[:, k].unsqueeze(0))
            self.counteval += 1

        # 2. Sort solutions.
        args = torch.argsort(self.arfitness, descending=self.maximization)

        # 3. Update mean.
        self.xmean = self.arx[:, args[: self.mu]] @ self.weights  # Recombination.
        zmean = (
            self.arz[:, args[: self.mu]] @ self.weights
        )  # == D.inverse() @ B.transpose(0,1) * (xmean-xold)/sigma

        # 4. Update evolution paths.
        self.ps = (1 - self.cs) * self.ps + (
            np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        ) * (self.B @ zmean)

        if np.linalg.norm(self.ps) / (
            np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lambda_))
        ) < (1.4 + 2 / (self.dim + 1)):
            self.hs = 1

        self.pc = (1 - self.cc) * self.pc + self.hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * self.B @ self.D @ zmean

        # 5. Update covariance matrix.
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1
            * (
                self.pc.view(-1, 1) @ self.pc.view(-1, 1).transpose(0, 1)
                + (1 - self.hs) * self.cc * (2 - self.cc) * self.C
            )
            + self.cmu
            * (self.B @ self.D @ self.arz[:, args[: self.mu]])
            @ torch.diag(self.weights)
            @ (self.B @ self.D @ self.arz[:, args[: self.mu]]).transpose(0, 1)
        )

        # 6. Update step-size sigma.
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

        # 7. Update B and D from C.
        if (
            self.counteval - self.eigeneval
            > self.lambda_ / (self.c1 + self.cmu) / self.dim / 10
        ):
            self.eigeneval = self.counteval
            self.C = torch.triu(self.C) + torch.triu(self.C, diagonal=1).transpose(
                0, 1
            )  # Enforce symmetry.
            try:
                D, self.B = torch.symeig(
                    self.C, eigenvectors=True
                )  # Eigendecomposition, B == normalized eigenvectors.
            except:
                D, self.B = torch.linalg.eigh(self.C, UPLO='U')

            self.D = torch.diag(
                torch.sqrt(D.clamp_min(1e-20))
            )  # D contains standard deviations now.

        # Escape flat fitness, or better terminate?
        if self.arfitness[0] == self.arfitness[int(np.ceil(0.7 * self.lambda_)) - 1]:
            self.sigma *= np.exp(0.2 + self.cs / self.damps)

        self.params = self.arx[:, args[0]].view(
            1, -1
        )  # Return the best point of the last generation. Notice that xmean is expected to be even better.

        self.params_history_list.append(torch.unsqueeze(torch.clip(self.xmean,0,1).clone(),0))
        #objective at mean
 
        self.objective_history_list.append(self.objective(self.params_history_list[-1])[0].clone())

        if self.verbose:
            print(f"Parameter: {self.params.numpy()}.")
            print(f"Function value: {self.arfitness[args[0]]}.")
            print(f"Sigma: {self.sigma}.")


