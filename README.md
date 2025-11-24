# Local Entropy Search - Supplementary Material

This repository contains the code for the ICLR submission titled "Local Entropy Search over Descent Sequences for Bayesian Optimization".

Note that this is prototype-stage research python code. For the camera ready version we are working on a BOtorch version of LES.

## Overview

![Local Entropy Search Overview](LESoverview.png)

Illustration of Local Entropy Search (LES) on a 2D example: a) After three initial evaluations, the distribution
over reachable local optima is wide. b, c) As LES selects new points, evaluations concentrate near
the descent sequence, and the distribution of the local optimum narrows. d) Convergence behavior of
LES. After 14 evaluations, the convergence criterion (see Appx. E.1) stops the optimization..

If you find our code or paper useful, please consider citing

```
@inproceedings{stenger2024early,
  title={Local Entropy Search over Descent Sequences for Bayesian Optimization},
  author={David Stenger, Armin Lindicke, Alexander von Rohr, Sebastian Trimpe},
  booktitle={Submitted to the Fourteenth International Conference on Learning Representations (under Review)},
  year={2025}
}
```



## Dependencies:

- The conda_env.yaml file contains most dependencies. Note that we did not use the newest BoTorch version.
- The GPflowSampling sampling code is linked via a git submodule (see .gitmodules)
- The original GIBO repository is located in benchmarking/gibo
- All experiments were conducted using python 3.12.7 

## Main Code Files:

- The local entropy search code is located in local_bo/local_bo/optimization/local_bo.py.
- The file benchmarking/benchmarking.py is the main file used for benchmarking. It interfaces the local entropy search code, all baselines and the objective functions.
- The folder notebooks contains the main notebooks for post processing.

## Reproducing the Results:

To run the benchmarks run the benchmarking python file with the following command line arguments:

python benchmarking/benchmarking.py \<index\> \<obj_name\> \<target_folder\> \<algorithms\> \<within_model\> \<hypeprior\>   

- \<index\>: integer - for the conversion from index to seed and problem dimension please refer to lines 320-324 of benchmarking py
- \<obj_name\>: name of the objective function
- \<target_folder\>: location where results are saved to
- \<algorithms\>: list of algorithms to evaluate
- \<within_model\>: True if ground truth GP hyperparameters should be used - only applicable if \<obj_name\> is gp_sample 
- \<hypeprior\>: 1: high complexity, 2: medium complexity, 3: low complexity. 4: extremely low complexity -2: Box hyperprior on length scales (used for the other benchmarks) 


We do not share the slurm skripts here for anonymity reasons. Below you can find some examples of how to run the benchmarks:

For within model comparison on 20 seeds with medium complexity the following should be run:
python benchmarking/benchmarking.py {1-100} gpsample results folder "('turbo','sobol','les_250_8','logei','hci_gibo','mes')"  True 2

For out-of-model comparison on 20 seeds with high complexity the following should be run:
python benchmarking/benchmarking.py {1-100} gpsample results folder "('turbo','sobol','les_250_8','logei','loghvarei','hci_gibo','mes')"  False 1

And for the synthetic and policy search benchmarks with 10 samples the following should be ran:
python benchmarking/benchmarking.py {1-10} {ackley,dixonprice,lunar,rover_trajectory,mopta08,square} results folder "('turbo','sobol','les_250_8','logei','hci_gibo','mes')"  False -2 

Parenthesis {} mean that the command should be executed for all of the elements. We parallelized the evaluation with slurm scripts.




















