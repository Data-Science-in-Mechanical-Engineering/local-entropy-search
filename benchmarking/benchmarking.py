import sys
import os
import gc
import tensorflow as tf
import csv
import ast
import torch
import numpy as np
import math
import logging
import time
import pandas as pd
import gpflow as gpf
from gpflow_sampling.models import PathwiseGPR 
from gpflow.kernels import SquaredExponential
from gpflow import set_trainable as gp_set_trainable

## Global constants    
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 
NUMBER_OF_INITIAL_SAMPLES = 2
DIMENSIONS = [5, 10, 20, 30, 50] # [5, 30] # Dimensions of problems checked
OPTIMIZE_NOISE = False

try:
    HYPERPRIOR = ast.literal_eval(sys.argv[6])
    print("Hyperprior is:")
    print(HYPERPRIOR)
except: 
    raise NotImplementedError()

if HYPERPRIOR == 0:
    USE_HVARFNER_HYPERPRIOR =  False
else:
    USE_HVARFNER_HYPERPRIOR =  True
    if HYPERPRIOR == 1: # complexity: high
        HVARFNER_LOC_FACTOR = -2.5*np.sqrt(2) 
        HVARFNER_SCALE = np.sqrt(3)/5 
    elif HYPERPRIOR == 2: # complexity: normal
        HVARFNER_LOC_FACTOR = -2.0*np.sqrt(2) 
        HVARFNER_SCALE = np.sqrt(3)/4    
    elif HYPERPRIOR == 3: # complexity: low
        HVARFNER_LOC_FACTOR = -1.0*np.sqrt(2) 
        HVARFNER_SCALE = np.sqrt(3)/2     
    elif HYPERPRIOR == 4: # complexity: extremely low
        HVARFNER_LOC_FACTOR = np.sqrt(2) 
        HVARFNER_SCALE = np.sqrt(3) 

if HYPERPRIOR == -1:
    USE_HVARFNER_HYPERPRIOR =  False

if HYPERPRIOR == -2:
    USE_HVARFNER_HYPERPRIOR =  False

NOISY_EVALS = True

try:
    WITHIN_MODEL_COMPARISON = ast.literal_eval(sys.argv[5]) #$HPCWORK/results/ 1 gpsample /hpcwork/vd818225/results_20250428_1703 ('les_250_8','localTS')
except Exception as e:
    WITHIN_MODEL_COMPARISON  = False
print("Within model comparison:")
print(WITHIN_MODEL_COMPARISON)

GPSAMPLES_LS_BOUNDS = [0.2, 2] # only used if USE_HVARFNER_HYPERPRIOR =  False squeue
VARIANCE = 1

ANALYTIC_TARGET = True
TARGET_FUNCTION = sys.argv[2] #'square', 'ackley', 'dixonprice', 'gpsample' has to be one!


if WITHIN_MODEL_COMPARISON and not TARGET_FUNCTION == 'gpsample':
    raise NotImplementedError("within model compariosn only works on gp sample paths") 

if (not TARGET_FUNCTION == 'gpsample') and (not HYPERPRIOR == -2):
    OPTIMIZE_NOISE = True
    print("always optimize the noise hyperparameter for non-gpsample target functions") 
       

#if TARGET_FUNCTION == 'lunar':
#    NOISE_STD = 1e-2
#else:
if HYPERPRIOR == -2:
    NOISE_STD = 1.42e-3
else:
    NOISE_STD = 2e-3 # TOODO change back to 2e-3

NOISE_STD_UB = 0.1
NOISE_STD_LB = 1.41e-3 #--> 2e-6 this is the lower bound the underlying gpflow model can handle



if TARGET_FUNCTION == 'rover_trajectory':
    DIMENSIONS = [60]
    print('Rover trajectory was picked - changing dimensions to 200')

if TARGET_FUNCTION == 'mopta08':
    DIMENSIONS = [124]
    print('Mopta08 was picked - changing dimensions to 124')

if TARGET_FUNCTION == 'lunar':
    DIMENSIONS = [12]
    print('lunar was picked - changing dimensions to 12')

if TARGET_FUNCTION == 'dixonprice' or TARGET_FUNCTION == 'square' or TARGET_FUNCTION == 'ackley' :
    DIMENSIONS = [5, 30]
    print('Analytical test functions were picked - changing dimensions to [5 30]')

# lesgradcond_20_8: conditioning on the gradient traces with 20 samples and 8 gradient points
# lesGD_250_8: use gradient descent instead of adam as the low level optimization algorithm
# lesCMAES_20_8: use gradient descent instead of adam as the low level optimization algorithm
# localTS: minimize one sample path and sample at that one local minimum
# les_250_8: 250_samples and conditioning 8 function value points
# les_250_4: 250_samples and conditioning on 4 function value points
# les_250_16: 250_samples and conditioning on 16 function value points
# les_20_8: 20_samples and conditioning 8 function value points

# std_gibo: normal gibo
# hci_gibo: gibo with alpha = 50%

try:
    METHODS = ast.literal_eval(sys.argv[4]) #$HPCWORK/results/ 1 gpsample /hpcwork/vd818225/results_20250428_1703 ('les_250_8','localTS')
    print(METHODS)
except Exception as e:
    resultsPrefix =  'benchmarking'
    METHODS = ('les_250_8','localTS','lesgradcond_20_8','les_20_8')#'localTS','les_250_8','mes','turbo','sobol','gibo')#,'gibo') # available methods: 'les_250_8', 'es_with_derivatives', 'es_without_derivatives', 'gibo', 'mes', 'turbo', 'sobol'


if 'lesCMAES_20_8' in METHODS:
    DIMENSIONS = [5, 10]


try:
    resultsPrefix = sys.argv[3] #$HPCWORK/results/
except Exception as e:
    resultsPrefix =  'benchmarking'

benchmark_results_dir = resultsPrefix + '/' 

FILES = {'results_dir':benchmark_results_dir+TARGET_FUNCTION+'/sampled_data',
            'general_results_file': 'sampled_data_history',
            'optimizer_history_dir': benchmark_results_dir+TARGET_FUNCTION+'/optimizer_history',
            'optimizer_history_file': 'list_of_bests',
            'length_scale_history_dir':benchmark_results_dir+TARGET_FUNCTION+'/length_scale',
            'length_scale_history_file': 'length_scale',
            'local_optima_dist_dir': benchmark_results_dir+TARGET_FUNCTION+'/local_optima_distribution',
            'local_optima_dist_file': 'local_opt_distr'}
GIBO_CONFIG = 'benchmarking/gibo_benchmarking_config.yaml'


# setup function for ground truth evaluations

def add_ground_truth_objective(FILES,SEED,algo_name,DIM,ground_truth_fun):
    
    
    optimizer_dir = FILES['optimizer_history_dir'] # list of bests
    optimizer_file = FILES['optimizer_history_file']
    optimizer_file = f'{optimizer_file}_{SEED:05d}_{DIM}_{algo_name}.csv'
    optimizer_filepath = os.path.join(optimizer_dir, optimizer_file)

    results_dir = FILES['results_dir'] # sampled data
    results_file = FILES['general_results_file']
    results_file = f'{results_file}_{SEED:05d}_{DIM}_{algo_name}.csv'
    results_filepath = os.path.join(results_dir, results_file)
    
    """
    1) Check if the file exists, else throw an error.
    2) Check if a column named 'f' exists; if yes, throw an error.
    3) Check if a column named 'y' exists; if yes, then also check that
       columns x1, x2, x3, ..., x{DIM} exist.
    4) If all the above checks are satisfied, create a new column 'f'
       by calling 'ground_truth_function' for each row on [x1, x2, x3, ..., x{DIM}].
    """ 

    for file_path in [optimizer_filepath,results_filepath]:
    # 1) Confirm file existence
        if not os.path.isfile(file_path):
            print("File not found:")
            print(file_path)
            continue
            #if file_path == 
            #raise FileNotFoundError(f"File not found: {file_path}")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # 2) Check if 'f' column exists
        if 'f' in df.columns:
            raise ValueError(
                "Error: Column 'f' already exists in the file. Cannot proceed."
            )

        # 3) Check if 'y' column exists and if so, check for x1.. xDIM columns
        if 'y' in df.columns:
            # Build the list of column names that we expect
            x_columns = [f"x{i}" for i in range(1, DIM + 1)]

            # Check if all required x-columns are present
            missing_cols = [col for col in x_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Error: The file is missing required columns: {missing_cols}"
                )

            # 4) Create new column 'f' using ground_truth_function
            #    For each row in the dataframe, gather x1.. xDIM and pass them
            #    as a numpy array to ground_truth_function.
            #print("")
            #print( df[x_columns].values)
            df['f'] = 0.0
            for idx, row_data in df[x_columns].iterrows():
                if not isinstance(row_data.values[0], str) or not row_data.values[0].startswith('Total'):
                    df.at[idx, 'f'] = ground_truth_fun(np.expand_dims(np.array(row_data.values), axis=0))


            #df['f'] = df[x_columns].apply(lambda row: ground_truth_fun(np.expand_dims(np.array(row.values), axis=0)))

            # Save the updated CSV (if you want to overwrite the original file)
            df.to_csv(file_path, index=False)
            print("New column 'f' has been created successfully.")
        else:
            print("Column 'y' does not exist. No action taken.")



    
    


## Conditional imports
## Should probably clean up this behemoth
from local_bo.optimization import Local_bo
import gpytorch
import botorch
from gpytorch.constraints import Positive, Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.priors import UniformPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from sklearn.preprocessing import StandardScaler
from gpytorch.constraints import Positive
from gibo.src.loop import loop as gibo_loop
from gibo.src import config
import yaml
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf
import math
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.generation import MaxPosteriorSampling


if any(s in METHODS for s in ['turbo','mes','logei','es_with_derivatives', 'es_without_derivatives', 'les_250_8','les_beta05_250_8', 'les_250_4', 'les_250_16','les_250_32','lesgradcond_20_8','lesGD_250_8','lesCMAES_20_8','localTS','les_20_8']):
    def setup_gpytorch_model_oom(trainx: torch.Tensor, trainy: torch.Tensor, dim, noise,enforce_extlow_prior = False) -> botorch.models.SingleTaskGP:
        s_d = np.sqrt(dim)
        if USE_HVARFNER_HYPERPRIOR:
            if enforce_extlow_prior:
                loc = np.sqrt(2) + np.log(np.sqrt(PROBLEM_DIMENSION)) 
                scale = np.sqrt(3)
            else:
                loc = HVARFNER_LOC_FACTOR + np.log(np.sqrt(PROBLEM_DIMENSION))  # Mean of the underlying normal distribution
                scale = HVARFNER_SCALE                    # Standard deviation of the underlying normal distribution
            # Create the LogNormalPrior
            lengthscale_prior = gpytorch.priors.LogNormalPrior(loc, scale)
            #lengthscale_prior = LogNormalPrior(0.0 + np.log(dim) / 2, 1.0)
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=dim, lengthscale_constraint=gpytorch.constraints.Interval(length_scale_bounds[0], length_scale_bounds[1]), lengthscale_prior=lengthscale_prior)
        else:
            lengthscale_prior = gpytorch.priors.UniformPrior(length_scale_bounds[0], length_scale_bounds[1])
            #lengthscale_prior = LogNormalPrior(0.0 + np.log(dim) / 2, 1.0)
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=dim, lengthscale_constraint=gpytorch.constraints.Interval(length_scale_bounds[0], length_scale_bounds[1]), lengthscale_prior=lengthscale_prior)
        
        try:
            base_kernel.lengthscale = torch.full((1, dim), length_scale_init)
        except Exception as e:
            base_kernel.lengthscale = length_scale_init
        custom_kernel = gpytorch.kernels.ScaleKernel(base_kernel=base_kernel)
        

        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(NOISE_STD_LB**2,NOISE_STD_UB**2))
        likelihood.noise = noise
        if not OPTIMIZE_NOISE:
            likelihood.noise_covar.raw_noise.requires_grad = False # disable noise hyperparameter tuning
        model = botorch.models.SingleTaskGP(train_X=trainx, train_Y=trainy, covar_module=custom_kernel, likelihood=likelihood, outcome_transform=None)
        model.covar_module.outputscale = VARIANCE
        return model


    def gpytorch_hyperparam_opt(trainx, trainy, dim, noise_std): 
        a = time.time()
        x_train = torch.tensor(trainx)
        y_train = torch.tensor(trainy)

        model = setup_gpytorch_model_oom(x_train, y_train, dim, noise_std**2)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        with gpytorch.settings.max_cholesky_size(float("inf")):
            botorch.fit.fit_gpytorch_mll(mll, max_attempts=20,optimizer_kwargs={'options':{'disp':True}})
        hyperparameters = {'lengthscales':model.covar_module.base_kernel.lengthscale.detach().numpy(),
                           'outputscale':model.covar_module.outputscale.detach().numpy(),
                           'noisevar':model.likelihood.noise.detach().numpy()}
        
        print(f'Lengthscales: {model.covar_module.base_kernel.lengthscale}')
        print(f'Variance: {model.covar_module.outputscale}')
        print(f'Noise: {model.likelihood.noise}')

        b = time.time()
        elapsed_time = b-a
        print(f"Time for hyperparameter tuning: {elapsed_time}")
        sys.stdout.flush()
        return hyperparameters


##########################################
## Analytic target functions            ##
##########################################

def denormalize(x, lower, upper): 
    lower = np.array(lower)
    upper = np.array(upper)
    return x * (upper - lower) + lower

def branin_function_01(x): 
    """
    2D test function. Expects normalized inputs, usually evaluated on [-5, 10] x [0 , 15] with recommended parameters a,b,c,r,s,t
    Global minimum: f(x) = 0.397887 at {(-pi,12.275), (pi, 2.275), (9.42478,2.475)}
    Source: https://www.sfu.ca/~ssurjano/branin.html
    """
    lower_bound = [-5, 0]
    upper_bound = [10, 15]
    x_denormalized = denormalize(x, lower_bound, upper_bound)
    a = 1
    b = 5.1/(4*np.square(np.pi))
    c = 5/np.pi
    r = 6
    s = 10
    pi = np.pi
    t = 1/(8*pi)

    x1 = x_denormalized[:,0]
    x2 = x_denormalized[:,1]
    res = a*np.square(x2-b*np.square(x1)+c*x1-r) + s*(1-t)*np.cos(x1) + s
    return res, x_denormalized

def ackley_function_01(x):
    """
    Test function on arbitrary dimensions. Expects normalized inputs, usually evaluated on [-32.768, 32.768]^d with recommended parameters a,b,c
    Global minimum: f(x) = 0 at (0, ..., 0)
    Source: https://www.sfu.ca/~ssurjano/ackley.html
    """
    d = x.shape[1] if x.ndim == 2 else x.shape[0] if x.ndim == 1 else 0
    lower_bounds = np.full([d], -32.768)
    upper_bounds = np.full([d], 32.768)
    a = 20
    b = 0.2
    c = 2*np.pi
    x_denormalized = denormalize(x, lower_bounds, upper_bounds)
    
    s1 = -a * np.exp(-b * np.sqrt(np.sum(np.square(x_denormalized), axis=1) / d))
    s2 = -np.exp(np.sum(np.cos(c*x_denormalized), axis=1) / d)

    return s1 + s2 + a +np.exp(1)

def square_function_01(x): 
    """
    Square function centered at (0.5, 0.5, ..., 0.5)
    """
    return np.sum(np.square(x-0.5), axis=1)

def dixon_price_function_01(x):
    """ 
    d-dimensional function with optimum in a valley
    https://www.sfu.ca/~ssurjano/dixonpr.html
    """
    d = x.shape[1] 
    lower_bounds = np.full([d], -10)
    upper_bounds = np.full([d], 10) 
    x_denormalized = denormalize(x, lower_bounds, upper_bounds)
    res = np.square(x_denormalized[:,0] - 1)

    for i in range(1,d): 
        res += (i+1) * np.square(2*np.square(x_denormalized[:,i]) - x_denormalized[:,i-1])

    return res

def rover_function(x):
    import sys
    sys.path.append('/anonymized/git/local-bo/benchmarking.external_test_functions')
    from external_test_functions.rover import Rover
    print('Rover was called with')
    print(x)
    rover = Rover()
    y = np.zeros([len(x)])
    for i in range(len(x)):
        x_denormalized = denormalize(x[i], rover.lb, rover.ub)
        y[i] = rover(x_denormalized)
    print('and returned')
    print(y)

    return y

def rover_trajectory_function(x):
    import sys
    sys.path.append('/anonymized/git/local-bo/benchmarking.external_test_functions')
    from external_test_functions.rover_trajectory import RoverObjective
    print('Rover trajectory was called with')
    print(x)
    rover = RoverObjective(dim=60)
    y = np.zeros([len(x)])
    for i in range(len(x)):
        x_denormalized = denormalize(x[i], rover.lb, rover.ub)
        y[i] = -rover.evaluate_true(x_denormalized)
    print('and returned')
    print(y)

    return y
    #print('asdf')

def lunar_function(x):
    import sys
    sys.path.append('/anonymized/git/local-bo/benchmarking.external_test_functions')
    import external_test_functions.lunar_lander as lunar_lander
    print('lunar lander was called with')
    print(x)
    y = np.zeros([len(x)])
    for i in range(len(x)):
        y[i]  =lunar_lander.objective(x[i])
    print('and returned')
    print(y)

    return y
    #print('asdf')

def mopta08_function(x):
    import sys
    sys.path.append('/anonymized/git/local-bo/benchmarking.external_test_functions')
    import external_test_functions.mopta08 as mopta08
    print('mopta08 was called with')
    print(x)
    y = np.zeros([len(x)])
    for i in range(len(x)):
        y[i]  =mopta08.objective(x[i])
    print('and returned')
    print(y)

    return y
    #print('asdf')



def dixon_price_function_01(x):
    """ 
    d-dimensional function with optimum in a valley
    https://www.sfu.ca/~ssurjano/dixonpr.html
    """
    d = x.shape[1] 
    lower_bounds = np.full([d], -10)
    upper_bounds = np.full([d], 10) 
    x_denormalized = denormalize(x, lower_bounds, upper_bounds)
    res = np.square(x_denormalized[:,0] - 1)

    for i in range(1,d): 
        res += (i+1) * np.square(2*np.square(x_denormalized[:,i]) - x_denormalized[:,i-1])

    return res


def anderssen_length_scale(d):
    return (d/6)**0.5*(1/3*(1+2*(1-3/(5*d))**0.5))**0.5

def get_length_scale(d: int, reference_dim: int = 2, reference_lengthscale: float = 0.2):
    """
    Function returns a scaled length scale according to Anderssen. Lengthscale is normalized to equal reference_lengthscale in reference_dim dimensions

    Args: 
        d (int): Dimension for which the lengthscale should be returned
        reference_dim (int): Dimension for which the lengthscale is normalized to reference_lengthscale
        reference_lengthscale (float): Lengthscale returned for reference_dim
    """

    return reference_lengthscale * anderssen_length_scale(d) / anderssen_length_scale(reference_dim)



ANALYTIC_FUNCTION_DICT = {'branin': branin_function_01,
                          'ackley': ackley_function_01,
                          'square': square_function_01,
                          'dixonprice':dixon_price_function_01,
                          'rover':rover_function,
                          'rover_trajectory':rover_trajectory_function,
                          'lunar':lunar_function,
                          'mopta08':mopta08_function}

##########################################
## Main code                            ##
##########################################
if __name__ == "__main__":
    index = int(sys.argv[1])
    SEED = (index-1) // len(DIMENSIONS)	+ 1
    ## Initialization
    PROBLEM_DIMENSION = DIMENSIONS[index % len(DIMENSIONS) - 1]
    NUMBER_OF_ITERATIONS = min(20*PROBLEM_DIMENSION, 400)  
    

    
    #NUMBER_OF_ITERATIONS = 5 # TOODO   
    
    def setup_target_standard(dimension: int, sample_num: int = 1, base_num: int = 1024) -> PathwiseGPR: 
        """
        Setup of GPR with SE_ARD kernel, in d dimensions. 
        Samples a specified number of paths based on k base functions (default: 1024)
        """
        if USE_HVARFNER_HYPERPRIOR:
            loc = HVARFNER_LOC_FACTOR + np.log(np.sqrt(PROBLEM_DIMENSION))  # Mean of the underlying normal distribution
            scale = HVARFNER_SCALE                    # Standard deviation of the underlying normal distribution

            # Create the LogNormalPrior
            prior = gpytorch.priors.LogNormalPrior(loc, scale)
            # Sample from the prior
            length_scales = prior.sample(sample_shape=torch.Size((PROBLEM_DIMENSION,))).numpy()       
            print(length_scales)  
        else:
            #if WITHIN_MODEL_COMPARISON:
            #    length_scales = get_length_scale(PROBLEM_DIMENSION)
            #else:    
            length_scales = np.random.uniform(low=GPSAMPLES_LS_BOUNDS[0], high=GPSAMPLES_LS_BOUNDS[1], size=dimension)

        print(length_scales)

        kernel = SquaredExponential(lengthscales=length_scales, variance=VARIANCE)

        noise2 = NOISE_STD**2 # measurement noise variance
        x_dummy = 1e6 * tf.ones(shape=(1,dimension), dtype=tf.float64)
        y_dummy = tf.zeros((1,1), dtype=tf.float64)

        model = PathwiseGPR(data=(x_dummy,y_dummy), kernel=kernel, noise_variance=noise2)
        gp_set_trainable(model.kernel, False)
        gp_set_trainable(model.likelihood, False)
        gpf.utilities.print_summary(model)

        paths = model.generate_paths(num_samples=sample_num, num_bases=base_num)
        _ = model.set_paths(paths)  # use a persistent set of sample pathstf.co

        #objFunModel.predict_f_samples(X),shape=(X.shape[0])

        return model,length_scales

    

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    if any(s in METHODS for s in ['gibo','les','localTS', 'mes','logei', 'turbo', 'sobol']):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)  # PyTorch (CPU)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups

    lb = tf.zeros((1,PROBLEM_DIMENSION),dtype=tf.double)
    ub = tf.ones((1,PROBLEM_DIMENSION),dtype=tf.double)

    ## Setup target function for the gibo sample
    if TARGET_FUNCTION == "gpsample":  
        objFunModel, length_scales_ground_truth = setup_target_standard(PROBLEM_DIMENSION)
        if WITHIN_MODEL_COMPARISON:
            length_scale_init = length_scales_ground_truth
            length_scale_bounds = [0.0001, 10**10]
        else:  
            if USE_HVARFNER_HYPERPRIOR:
                loc = HVARFNER_LOC_FACTOR + np.log(np.sqrt(PROBLEM_DIMENSION))  # Mean of the underlying normal distribution
                scale = HVARFNER_SCALE                    # Standard deviation of the underlying normal distribution
                #mean_pdf_val = # initialize with mean of hyperprior
                length_scale_init = np.exp(loc+ scale/2)
                length_scale_bounds = [0.0001, 10**10]
            else:
                raise NotImplementedError("not implemented")
                length_scale_init = max(min(GPSAMPLES_LS_BOUNDS[1], np.sqrt(PROBLEM_DIMENSION)/10), GPSAMPLES_LS_BOUNDS[0])
                length_scale_bounds = GPSAMPLES_LS_BOUNDS

        def gpObj(X): 
            print(X)
            rawResults = objFunModel.predict_f_samples(tf.convert_to_tensor(X, dtype=tf.float64)) 
            reshapedResult = tf.reshape(rawResults,shape=(X.shape[0])).numpy()
            return reshapedResult

        ANALYTIC_FUNCTION_DICT["gpsample"] = gpObj # lambda X: tf.reshape(objFunModel.predict_f_samples(X),shape=(X.shape[0]))
        
    else:
        NOISY_EVALS = False
        print("Do not use artificial noise on the target function in the synthetic and black-box functions")
        if USE_HVARFNER_HYPERPRIOR:
            loc = HVARFNER_LOC_FACTOR + np.log(np.sqrt(PROBLEM_DIMENSION))  # Mean of the underlying normal distribution
            scale = HVARFNER_SCALE                    # Standard deviation of the underlying normal distribution
            #mean_pdf_val = # initialize with mean of hyperprior
            length_scale_init = np.exp(loc+ scale/2)
            length_scale_bounds = [0.0001, 10**10]
        else:
            length_scales_ground_truth =[]
            length_scale_init = np.sqrt(PROBLEM_DIMENSION)*0.2
            length_scale_bounds = [0.05, 1*np.sqrt(PROBLEM_DIMENSION)]
            if HYPERPRIOR == -1:
                length_scale_init = 0.999
                length_scale_bounds = [0.05, 1]


    
    if not os.path.exists(FILES['results_dir']):
        os.makedirs(FILES['results_dir'], exist_ok=True)

    if not os.path.exists(FILES['optimizer_history_dir']):
        os.makedirs(FILES['optimizer_history_dir'], exist_ok=True)

    if not os.path.exists(FILES['length_scale_history_dir']):
        os.makedirs(FILES['length_scale_history_dir'], exist_ok=True)

    if not os.path.exists(FILES['local_optima_dist_dir']):
        os.makedirs(FILES['local_optima_dist_dir'], exist_ok=True)


    ################################# prepare objective functions

    obj_ground_truth = ANALYTIC_FUNCTION_DICT[TARGET_FUNCTION] # evaluation without noise
    if NOISY_EVALS:
        obj_noisy = lambda X: ANALYTIC_FUNCTION_DICT[TARGET_FUNCTION](X) + np.random.normal(loc=0.0, scale=NOISE_STD, size=len(X)) # evaluation without noise
    else:
        obj_noisy = obj_ground_truth 
    

    if any(s in METHODS for s in ['es_with_derivatives', 'es_without_derivatives', 'les_250_8','les_beta05_250_8', 'les_250_4', 'les_250_16','les_250_32','lesgradcond_20_8','lesGD_250_8','lesCMAES_20_8','localTS','les_20_8']):
        #raise NotImplementedError()
        # Define call pattern for local entropy methods
        if ANALYTIC_TARGET: 
            objFun = lambda X: obj_noisy(X.numpy())
        else: 
            raise NotImplementedError()
       

        specs_default = {'no_initial_samples': NUMBER_OF_INITIAL_SAMPLES,
                    'no_total_samples':NUMBER_OF_ITERATIONS, 
                    'lb':lb,
                    'ub':ub, 
                    'length_scale': length_scale_init,
                    'seed':SEED,
                    'length_scale_bounds_scaled': length_scale_bounds,
                    'acquisition_function_opt_method':'best_entropy_pt', 
                    'optimize_hyperparameters':True,
                    'scale_problem': True,
                    'use_random_start_pts_for_opt':False,
                    'init_noise_std': NOISE_STD,
                    'log_to_file': False, 
                    'log_level': 50, #50, #logging.DEBUG --> debug logging,   #50--> Disable logging
                    'plot_at_each_iter':False,
                    'output_files':FILES,
                    'external_hyper_param_opt':gpytorch_hyperparam_opt} # now uses hyperparameter optimization of botorch
        
        if WITHIN_MODEL_COMPARISON:
            specs_default['optimize_hyperparameters'] = False
            specs_default['scale_problem'] =  False
            specs_default['external_hyper_param_opt'] = None


        #################################
        # Implementation of all methods #
        #################################




        if 'localTS' in METHODS:
            ## Optimization without derivatives
            specs = specs_default
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = False
            specs['number_of_entropy_pts']      = 1
            specs['low_level_adam'] = True
            specs['derivative_cond'] = False
            specs['output_files']['algo_name']   = 'localTS'
            try:
                optimizer_without_derivatives = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
                #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_without_derivatives = []

            try:
                optimizer_without_derivatives.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_without_derivatives
            gc.collect()

        if 'lesGD_250_8' in METHODS:
            specs = specs_default
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = True      
            specs['low_level_adam'] = False # use vanilla gradient descent instead of ADAM to optimize the individual sample paths
            specs['number_of_entropy_pts'] = 250
            specs['derivative_cond'] = False
            specs['output_files']['algo_name']   = 'lesGD_250_8'      
            try:
                optimizer_with_derivatives = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
                #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_with_derivatives = []

            try:
                optimizer_with_derivatives.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization with derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_derivatives
            gc.collect()

        if 'lesCMAES_20_8' in METHODS:
            specs = specs_default
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = True
            specs['low_level_adam'] = False
            specs['low_level_CMAES'] = True,
            specs['output_files']['algo_name']   = 'lesCMAES_20_8'
            specs['derivative_cond'] = False
            specs['number_of_entropy_pts'] = 20
            try:
                optimizer_with_traces = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
                #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_with_traces = []

            try:
                optimizer_with_traces.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_traces
            gc.collect()


        if 'lesgradcond_20_8' in METHODS:
            ## Optimization with derivatives

            
            specs = specs_default
            specs['no_total_samples'] = min(20*PROBLEM_DIMENSION, 300) # only 300 samples because we had a memory overflow in 50d  
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = True 
            specs['low_level_adam'] = True
            specs['number_of_entropy_pts'] = 20
            specs['derivative_cond'] = True
            specs['output_files']['algo_name']   = 'lesgradcond_20_8'  
            try:
                optimizer_with_derivatives = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
                #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_with_derivatives = []

            try:
                optimizer_with_derivatives.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization with derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_derivatives
            gc.collect()
        
        if 'es_with_derivatives' in METHODS:
            #raise NotImplementedError()

            specs = specs_default
            specs['use_derivative_information'] = True
            specs['use_gd_traces'] = False
            specs['low_level_adam'] = True
            specs['number_of_entropy_pts'] = 250
            specs['derivative_cond'] = False
            specs['output_files']['algo_name']   = 'les_fp_wgrad'
            try:
                optimizer_with_traces = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
                if e is FileExistsError:
                    print(e)
                    print('Optimization was already completed')
                    optimizer_with_traces = []
            try:
                optimizer_with_traces.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_traces
            gc.collect()

        if 'es_without_derivatives' in METHODS:
            raise NotImplementedError()
            

            ## Optimization without derivatives
            specs['use_derivative_information'] = False
            specs['output_files']['algo_name']   = 'les_fp_nograd'
            optimizer_without_derivatives = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            try:
                optimizer_without_derivatives.cluster_study()
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_without_derivatives
            gc.collect()

        if 'les_250_8' in METHODS: 
            specs = specs_default
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = True
            specs['low_level_adam'] = True
            specs['number_of_entropy_pts'] = 250
            specs['derivative_cond'] = False
            specs['output_files']['algo_name']   = 'les_250_8'
            try:
                optimizer_with_traces = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
                #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_with_traces = []
            try:
                optimizer_with_traces.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_traces
            gc.collect()

        if 'les_beta05_250_8' in METHODS: 
            specs = specs_default
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = True
            specs['low_level_adam'] = True
            specs['low_level_adam_beta05'] = True
            specs['number_of_entropy_pts'] = 250
            specs['derivative_cond'] = False
            specs['output_files']['algo_name']   = 'les_beta05_250_8'
            try:
                optimizer_with_traces = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
                #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_with_traces = []
            try:
                optimizer_with_traces.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_traces
            gc.collect()

        

        if 'les_250_4' in METHODS: 
            specs = specs_default
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = True
            specs['low_level_adam'] = True
            specs['number_of_entropy_pts'] = 250
            specs['derivative_cond'] = False
            specs['GD_trace_divisions'] = 4
            specs['output_files']['algo_name']   = 'les_250_4'
            try:
                optimizer_with_traces = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
            #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_with_traces = []
            try:
                optimizer_with_traces.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_traces
            gc.collect() 

        if 'les_250_16' in METHODS: 
            specs = specs_default
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = True
            specs['low_level_adam'] = True
            specs['number_of_entropy_pts'] = 250
            specs['derivative_cond'] = False
            specs['GD_trace_divisions'] = 16
            specs['output_files']['algo_name']   = 'les_250_16'
            try:
                optimizer_with_traces = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
            #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_with_traces = []
            try:
                optimizer_with_traces.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_traces
            gc.collect()  


        if 'les_20_8' in METHODS: 
            specs = specs_default
            specs['use_derivative_information'] = False
            specs['use_gd_traces'] = True
            specs['low_level_adam'] = True
            specs['output_files']['algo_name']   = 'les_20_8'
            specs['derivative_cond'] = False
            specs['number_of_entropy_pts'] = 20
            try:
                optimizer_with_traces = Local_bo(objFun, PROBLEM_DIMENSION, **specs)
            except Exception as e:
                #if e is FileExistsError:
                print(e)
                print('Optimization was already completed')
                optimizer_with_traces = []

            try:
                optimizer_with_traces.cluster_study()
                add_ground_truth_objective(FILES,SEED,specs['output_files']['algo_name'],PROBLEM_DIMENSION,obj_ground_truth)
            except Exception as e:
                print(f'Failed optimization without derivatives with seed {SEED} in dimension {PROBLEM_DIMENSION}')
                print(e)
            del optimizer_with_traces
            gc.collect()

    if 'std_gibo' in METHODS: ### careful there are a lot of redundancies with hci-gibo below
        # Redefine objective function to suit torch implementation (maximization)
        if ANALYTIC_TARGET: 
            objFun_gibo = lambda X: -torch.tensor(obj_noisy(X.numpy())).to(torch.float32)
        else: 
            raise NotImplementedError()

        with open(GIBO_CONFIG, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        if USE_HVARFNER_HYPERPRIOR:
            loc = HVARFNER_LOC_FACTOR + np.log(np.sqrt(PROBLEM_DIMENSION))  # Mean of the underlying normal distribution
            scale = HVARFNER_SCALE                    # Standard deviation of the underlying normal distribution

            # Create the LogNormalPrior
            cfg['optimizer_config']['model_config']['lengthscale_constraint'] = {'constraint': 'interval', 
                                                                                'kwargs': {'lower_bound': length_scale_bounds[0], 
                                                                                        'upper_bound':  length_scale_bounds[1]}}
            
            cfg['optimizer_config']['model_config']['lengthscale_hyperprior'] = {'prior': 'lognormal', 
                                                                                'kwargs': {'loc': loc , 
                                                                                        'scale':  scale}}


        else:

            # Manually manipulate config dict to get dynamic upper bounds
            cfg['optimizer_config']['model_config']['lengthscale_constraint'] = {'constraint': 'interval', 
                                                                                'kwargs': {'lower_bound': length_scale_bounds[0], 
                                                                                        'upper_bound':  length_scale_bounds[1]}}
            
            cfg['optimizer_config']['model_config']['lengthscale_hyperprior'] = {'prior': 'uniform', 
                                                                                'kwargs': {'a': length_scale_bounds[0], 
                                                                                        'b':  length_scale_bounds[1]}}

        # Translate config dictionary.
        cfg = config.insert(cfg, config.insertion_config)       

        # Set hyperparameters for out-of-model comparison 
        hypers = {
            "covar_module.base_kernel.lengthscale":torch.full((1, PROBLEM_DIMENSION), length_scale_init),
            "likelihood.noise": torch.tensor(NOISE_STD**2),
        }
        if WITHIN_MODEL_COMPARISON:
            cfg['optimizer_config']["hyperparameter_config"]['optimize_hyperparameters'] = False
        if OPTIMIZE_NOISE:
           print("Original no noise optimization config:")
           print(cfg['optimizer_config']["hyperparameter_config"]["no_noise_optimization"])
           cfg['optimizer_config']["hyperparameter_config"]["no_noise_optimization"] = False


        cfg_dim = config.evaluate(
                        cfg,
                        dim_search_space=PROBLEM_DIMENSION,
                        factor_lengthscale=None,
                        factor_N_max=5,
                        hypers=hypers,
                    )
        
        # Set noise constraint after parsing of config, because Gibo has bug where there has to be a kwarg for any constraint but gpytorch.constraints.Positive doesnt have any kwargs
        #cfg_dim['optimizer_config']['model_config']['noise_constraint'] = Positive()
        cfg_dim['optimizer_config']['model_config']['noise_constraint'] = Interval(NOISE_STD_LB**2,NOISE_STD_UB**2)

        # Initial point is selected to coincide with other methods
        tf.random.set_seed(SEED)
        initial_samples = torch.from_numpy(tf.random.uniform(shape=(NUMBER_OF_INITIAL_SAMPLES,PROBLEM_DIMENSION),
                                                     minval=0.0,
                                                     maxval=1.0,
                                                     dtype=tf.double).numpy())

        y_vals = -objFun_gibo(initial_samples)
        _, min_index = torch.min(y_vals, dim=0)
        initial_sample = initial_samples[min_index, :]
        if WITHIN_MODEL_COMPARISON:
            cfg_dim["optimizer_config"]["standardize_obj"] = False
        else:
            cfg_dim["optimizer_config"]["standardize_obj"] = True
        #cfg_dim["optimizer_config"]["standardize_obj"] = True
        try: #TOODO insert try cath
            optimizer_hist_file = FILES['optimizer_history_dir'] + f'/{FILES["optimizer_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_std_gibo.csv'
            if os.path.exists(optimizer_hist_file):
                raise FileExistsError(f"Path '{optimizer_hist_file}' already exists.")    
            params, calls_in_iteration = gibo_loop(
                params_init= initial_sample.to(torch.float32),
                max_iterations=None,
                max_objective_calls=NUMBER_OF_ITERATIONS,
                objective=objFun_gibo,
                Optimizer=cfg_dim["method"],
                optimizer_config=cfg_dim["optimizer_config"],
                verbose=False,
            )
            params = torch.squeeze(torch.stack(params))
            calls_in_iteration = torch.unsqueeze(torch.tensor(calls_in_iteration), axis=-1)
            calls_in_iteration = torch.cat([calls_in_iteration, calls_in_iteration[-1:]], dim=0)
            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            header.append('y')
            header.append('n')
            
            y_values = torch.unsqueeze(-objFun_gibo(params), 1)

            with open(optimizer_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(torch.concat([params, y_values, calls_in_iteration], axis=1).numpy())
                writer.writerow([f'Total calls to objective function: {calls_in_iteration[-1]}'])
                
                # results_file = FILES['results_dir'] + f'/{FILES['general_results_file']}_{SEED:05d}_{PROBLEM_DIMENSION}_gibo.csv'
                # y_values = torch.unsqueeze(-objFun_gibo(objFun_gibo.saved_tensor, monitor=False), 1)
                # with open(results_file, 'w', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow(header)
                #     writer.writerows(torch.concat([objFun_gibo.saved_tensor, y_values], axis=1).numpy())
        except Exception as e:
            print(f'Failed optimization std_gibo with seed {SEED} in dimension {PROBLEM_DIMENSION}')
            print(e)
        
        add_ground_truth_objective(FILES,SEED,'std_gibo',PROBLEM_DIMENSION,obj_ground_truth)

    if 'hci_gibo' in METHODS: 
        gibo_start = time.time()
        # Redefine objective function to suit torch implementation (maximization)
        def log_calls_to_csv(csv_path,func):
            def wrapper(params):
                # Call the original function
                result = func(params)
                
                # Determine if we need to write the header (when file doesn't exist)
                write_header = not os.path.exists(csv_path)
                print("wrapper called")
                print(write_header)
                # Write to CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header if file is new
                    if write_header:
                        header = [f"x{i+1}" for i in range(len(params[0]))] + ["y"] + ["timestamp"]
                        print(header)
                        writer.writerow(header)
                    
                    ## Convert params to a list and append result
                    if len(params) == 1:
                        print("called with one param")
                        row = np.concatenate((params[0], result, np.expand_dims(np.array(time.time()),0)))
                        print(row)
                        print(row.shape)
                        writer.writerow(row)
                        
                    else:
                        print("called with multiple param")
                        for i in range(len(params)):
                            row = np.concatenate((params[i], np.expand_dims(result[i],0),np.expand_dims(np.array(time.time()),0)))
                            print(row)
                            print(row.shape)
                            writer.writerow(row)

                #if not write_header:
                #    raise NotImplementedError('debug')
                
                return result
            return wrapper




        if ANALYTIC_TARGET:
            sampling_hist_file = FILES['results_dir'] + f'/{FILES["general_results_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_hci_gibo_09.csv'
            obj_with_wrapper_noisy_gibo = log_calls_to_csv(sampling_hist_file,obj_noisy)  
            objFun_gibo = lambda X: -torch.tensor(obj_with_wrapper_noisy_gibo(X.numpy())).to(torch.float32)
            objFun_gibo_wo_csv_wrapper = lambda X: -torch.tensor(obj_noisy(X.numpy())).to(torch.float32)
        else: 
            raise NotImplementedError()

        with open(GIBO_CONFIG, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        # Manually manipulate config dict to get dynamic upper bounds
        if USE_HVARFNER_HYPERPRIOR:
            loc = HVARFNER_LOC_FACTOR + np.log(np.sqrt(PROBLEM_DIMENSION))  # Mean of the underlying normal distribution
            scale = HVARFNER_SCALE                    # Standard deviation of the underlying normal distribution

            # Create the LogNormalPrior
            cfg['optimizer_config']['model_config']['lengthscale_constraint'] = {'constraint': 'interval', 
                                                                                'kwargs': {'lower_bound': length_scale_bounds[0], 
                                                                                        'upper_bound':  length_scale_bounds[1]}}
            
            cfg['optimizer_config']['model_config']['lengthscale_hyperprior'] = {'prior': 'lognormal', 
                                                                                'kwargs': {'loc': loc , 
                                                                                        'scale':  scale}}

        else:

            # Manually manipulate config dict to get dynamic upper bounds
            cfg['optimizer_config']['model_config']['lengthscale_constraint'] = {'constraint': 'interval', 
                                                                                'kwargs': {'lower_bound': length_scale_bounds[0], 
                                                                                        'upper_bound':  length_scale_bounds[1]}}
            
            cfg['optimizer_config']['model_config']['lengthscale_hyperprior'] = {'prior': 'uniform', 
                                                                                'kwargs': {'a': length_scale_bounds[0], 
                                                                                        'b':  length_scale_bounds[1]}}
        # Translate config dictionary.
        cfg = config.insert(cfg, config.insertion_config)       

        # Set hyperparameters for out-of-model comparison
        try:
            hypers = {
            "covar_module.base_kernel.lengthscale":torch.full((1, PROBLEM_DIMENSION), length_scale_init),
            "likelihood.noise": torch.tensor(NOISE_STD**2),
        }        
        except Exception as e:
            hypers = {
            "covar_module.base_kernel.lengthscale": torch.from_numpy(length_scale_init),
            "likelihood.noise": torch.tensor(NOISE_STD**2),
            }
        if WITHIN_MODEL_COMPARISON:
            cfg['optimizer_config']["hyperparameter_config"]['optimize_hyperparameters'] = False
        if OPTIMIZE_NOISE:
           print("Original no noise optimization config:")
           print(cfg['optimizer_config']["hyperparameter_config"]["no_noise_optimization"])
           cfg['optimizer_config']["hyperparameter_config"]["no_noise_optimization"] = False


        cfg_dim = config.evaluate(
                        cfg,
                        dim_search_space=PROBLEM_DIMENSION,
                        factor_lengthscale=None,
                        factor_N_max=5,
                        hypers=hypers,
                    )
        
        # Set noise constraint after parsing of config, because Gibo has bug where there has to be a kwarg for any constraint but gpytorch.constraints.Positive doesnt have any kwargs
        #cfg_dim['optimizer_config']['model_config']['noise_constraint'] = Positive()
        cfg_dim['optimizer_config']['model_config']['noise_constraint'] = Interval(NOISE_STD_LB**2,NOISE_STD_UB**2)

        # Initial point is selected to coincide with other methods
        tf.random.set_seed(SEED)
        initial_samples = torch.from_numpy(tf.random.uniform(shape=(NUMBER_OF_INITIAL_SAMPLES,PROBLEM_DIMENSION),
                                                     minval=0.0,
                                                     maxval=1.0,
                                                     dtype=tf.double).numpy())

        y_vals = -objFun_gibo(initial_samples)
        _, min_index = torch.min(y_vals, dim=0)
        initial_sample = initial_samples[min_index, :]
        if WITHIN_MODEL_COMPARISON:
            cfg_dim["optimizer_config"]["standardize_obj"] = False
        else:
            cfg_dim["optimizer_config"]["standardize_obj"] = True
        try: #TOODO insert try catch
            optimizer_hist_file = FILES['optimizer_history_dir'] + f'/{FILES["optimizer_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_hci_gibo_09.csv'
            if os.path.exists(optimizer_hist_file):
                raise FileExistsError(f"Path '{optimizer_hist_file}' already exists.")  
            cfg_dim["optimizer_config"]["alpha"] = 0.9
            params, calls_in_iteration = gibo_loop(
                params_init= initial_sample.to(torch.float32),
                max_iterations=None,
                max_objective_calls=NUMBER_OF_ITERATIONS,
                objective=objFun_gibo,
                Optimizer=cfg_dim["method"],
                optimizer_config=cfg_dim["optimizer_config"],
                verbose=True,
            )
            params = torch.squeeze(torch.stack(params))
            calls_in_iteration = torch.unsqueeze(torch.tensor(calls_in_iteration), axis=-1)
            calls_in_iteration = torch.cat([calls_in_iteration, calls_in_iteration[-1:]], dim=0)
            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            header.append('y')
            header.append('n')

            y_values = torch.unsqueeze(-objFun_gibo_wo_csv_wrapper(params), 1)

            with open(optimizer_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(torch.concat([params, y_values, calls_in_iteration], axis=1).numpy())
                writer.writerow([f'Total calls to objective function: {calls_in_iteration[-1]}'+ f'Total Time{time.time()-gibo_start}'])

            #print("before hci_gibo_gt_add")
            add_ground_truth_objective(FILES,SEED,'hci_gibo_09',PROBLEM_DIMENSION,obj_ground_truth)
                
        except Exception as e:
            print(f'Failed optimization with gibo with seed {SEED} in dimension {PROBLEM_DIMENSION}')
            print(e)
    
    if 'mes' in METHODS: 
        optimizer_hist_file = FILES['optimizer_history_dir'] + f'/{FILES["optimizer_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_mes.csv'
        if os.path.exists(optimizer_hist_file):
            print(f"Path '{optimizer_hist_file}' already exists.") 
        else:

            if ANALYTIC_TARGET: 
                objFun_mes = lambda X: -torch.tensor(obj_noisy(X.numpy()))
            else: 
                raise NotImplementedError()
            
                
            # weird torch/tensorflow conversion to maintain same starting points across methods
            tf.random.set_seed(SEED)
            x_train = torch.from_numpy(tf.random.uniform(shape=(NUMBER_OF_INITIAL_SAMPLES,PROBLEM_DIMENSION),
                                                        minval=0.0,
                                                        maxval=1.0,
                                                        dtype=tf.float64).numpy())
            
            time_stamps = [time.time()]*NUMBER_OF_INITIAL_SAMPLES

            y_train = torch.reshape(objFun_mes(x_train), (-1,1))
            scaler_y = StandardScaler()

            #length_scales = torch.from_numpy(tf.random.uniform(shape=(0,PROBLEM_DIMENSION),
            #                                             minval=0.0,
            #                                             maxval=1.0,
            #                                             dtype=tf.float64).numpy()) 
            if WITHIN_MODEL_COMPARISON or len(length_scales_ground_truth) == 0:   
                length_scales = []
            else:
                length_scales = torch.unsqueeze(torch.from_numpy(length_scales_ground_truth),0)

            # MES BO loop
            for _ in range(NUMBER_OF_ITERATIONS-NUMBER_OF_INITIAL_SAMPLES):
                a = time.time()
                print(f'Iteration {_}')
                candidate_set = torch.rand(5000, PROBLEM_DIMENSION, dtype=torch.float64)
                if WITHIN_MODEL_COMPARISON:
                    y_train_stdzd = torch.tensor(y_train)
                else:
                    y_train_stdzd = torch.tensor(scaler_y.fit_transform(y_train))
                noise = NOISE_STD # / scaler_y.scale_
                model = setup_gpytorch_model_oom(x_train, y_train_stdzd, PROBLEM_DIMENSION, noise**2)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                with gpytorch.settings.max_cholesky_size(float("inf")):
                    if not WITHIN_MODEL_COMPARISON:
                        a = time.time()
                        fit_gpytorch_mll(mll, max_attempts=20,optimizer_kwargs={'options':{'disp':True}})
                        b = time.time()
                        elapsed_time = b-a
                        print(f"Time for hyperparameter tuning: {elapsed_time}")
                    model.eval()
                    model.posterior(x_train)
                    print(f'Lengthscales: {model.covar_module.base_kernel.lengthscale}')
                    print(f'Variance: {model.covar_module.outputscale}')
                    print(f'Noise: {model.likelihood.noise}')
                    qMES = qMaxValueEntropy(model, candidate_set)
                    candidates, ac = optimize_acqf(
                        acq_function=qMES,
                        bounds=torch.stack([torch.zeros(PROBLEM_DIMENSION), torch.ones(PROBLEM_DIMENSION)]),
                        q=1,
                        num_restarts=10,
                        raw_samples=512,
                        timeout_sec=120, # for some reason in the dixonprice case acqusition function optimization took extremely long and was inconsistent. Therefore we added this timeout
                    )

                evaluation = torch.unsqueeze(objFun_mes(candidates),-1)
                time_stamps.append(time.time())
                if length_scales == []:
                    length_scales = model.covar_module.base_kernel.lengthscale
                else:
                    length_scales = torch.concat([length_scales, model.covar_module.base_kernel.lengthscale], axis=0)
                x_train = torch.concat([x_train, candidates], axis=0)
                y_train = torch.concat([y_train, evaluation], axis=0)
                b = time.time()
                elapsed_time = b-a
                print(f"Time for mes iteration: {elapsed_time}")
                sys.stdout.flush()

            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            if TARGET_FUNCTION == 'gpsample':
                header[-1] = header[-1] + " (first row is ground truth)"

            length_scale_hist = FILES['length_scale_history_dir'] + f'/{FILES["length_scale_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_mes.csv'

            with open(length_scale_hist, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(length_scales.detach().numpy())
            
            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            header.append('y')
            sampling_hist_file = FILES['results_dir'] + f'/{FILES["general_results_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_mes.csv'

            with open(sampling_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(torch.concat([x_train, -y_train], axis=1).numpy())

            # generate history of best samples
            data = torch.concat([x_train, -y_train], axis=1)
            best_hist = torch.zeros_like(data)
            current_best = torch.full((PROBLEM_DIMENSION,), torch.inf)

            for i,row in enumerate(data): 
                if row[-1] < current_best[-1]:
                    current_best = row
                best_hist[i,:] = current_best
            
            header.append('timestamp')

            with open(optimizer_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(np.concatenate([best_hist.numpy(), np.expand_dims(np.array(time_stamps),axis=1)],axis=1))
            add_ground_truth_objective(FILES,SEED,'mes',PROBLEM_DIMENSION,obj_ground_truth)

    
     
    def logeiloop(method_name):     
        optimizer_hist_file = FILES['optimizer_history_dir'] + f'/{FILES["optimizer_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_'+ method_name+'.csv'
        if os.path.exists(optimizer_hist_file):
            print(f"Path '{optimizer_hist_file}' already exists.") 
        else:

            if ANALYTIC_TARGET: 
                objFun_logei = lambda X: -torch.tensor(obj_noisy(X.numpy()))
            else: 
                raise NotImplementedError()
            
                
            # weird torch/tensorflow conversion to maintain same starting points across methods
            tf.random.set_seed(SEED)
            x_train = torch.from_numpy(tf.random.uniform(shape=(NUMBER_OF_INITIAL_SAMPLES,PROBLEM_DIMENSION),
                                                        minval=0.0,
                                                        maxval=1.0,
                                                        dtype=tf.float64).numpy())

            y_train = torch.reshape(objFun_logei(x_train), (-1,1))
            time_stamps = [time.time()]*NUMBER_OF_INITIAL_SAMPLES
            scaler_y = StandardScaler()

            #length_scales = torch.from_numpy(tf.random.uniform(shape=(0,PROBLEM_DIMENSION),
            #                                             minval=0.0,
            #                                             maxval=1.0,
            #                                             dtype=tf.float64).numpy()) 
            if WITHIN_MODEL_COMPARISON or len(length_scales_ground_truth) == 0:   
                length_scales = []
            else:
                length_scales = torch.unsqueeze(torch.from_numpy(length_scales_ground_truth),0)

            # logei BO loop
            for _ in range(NUMBER_OF_ITERATIONS-NUMBER_OF_INITIAL_SAMPLES):
                a = time.time()
                print(f'Iteration {_}')
                if WITHIN_MODEL_COMPARISON:
                    y_train_stdzd = torch.tensor(y_train)
                else:
                    y_train_stdzd = torch.tensor(scaler_y.fit_transform(y_train))
                noise = NOISE_STD # / scaler_y.scale_
                if method_name == 'loghvarei':
                    model = setup_gpytorch_model_oom(x_train, y_train_stdzd, PROBLEM_DIMENSION, noise**2, enforce_extlow_prior = True)
                else:
                    model = setup_gpytorch_model_oom(x_train, y_train_stdzd, PROBLEM_DIMENSION, noise**2)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                with gpytorch.settings.max_cholesky_size(float("inf")):
                    if not WITHIN_MODEL_COMPARISON:
                        a = time.time()
                        fit_gpytorch_mll(mll, max_attempts=20,optimizer_kwargs={'options':{'disp':True}})
                        b = time.time()
                        elapsed_time = b-a
                        print(f"Time for hyperparameter tuning: {elapsed_time}")
                    model.eval()
                    model.posterior(x_train)
                    print(f'Lengthscales: {model.covar_module.base_kernel.lengthscale}')
                    print(f'Variance: {model.covar_module.outputscale}')
                    print(f'Noise: {model.likelihood.noise}')
                    best_value = y_train_stdzd.max()
                    LogEI = botorch.acquisition.analytic.LogExpectedImprovement(model=model, best_f=best_value)
                    candidates, ac = optimize_acqf(
                        acq_function=LogEI,
                        bounds=torch.stack([torch.zeros(PROBLEM_DIMENSION), torch.ones(PROBLEM_DIMENSION)]),
                        q=1,
                        num_restarts=10,
                        raw_samples=512,
                        timeout_sec=120, # for some reason in the dixonprice case acqusition function optimization took extremely long and was inconsistent. Therefore we added this timeout (comment originally from MES)
                    )

                evaluation = torch.unsqueeze(objFun_logei(candidates),-1)
                time_stamps.append(time.time())
                if length_scales == []:
                    length_scales = model.covar_module.base_kernel.lengthscale
                else:
                    length_scales = torch.concat([length_scales, model.covar_module.base_kernel.lengthscale], axis=0)
                x_train = torch.concat([x_train, candidates], axis=0)
                y_train = torch.concat([y_train, evaluation], axis=0)
                b = time.time()
                elapsed_time = b-a
                print("Time for"+ method_name+f" iteration: {elapsed_time}")
                sys.stdout.flush()

            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            if TARGET_FUNCTION == 'gpsample':
                header[-1] = header[-1] + " (first row is ground truth)"

            length_scale_hist = FILES['length_scale_history_dir'] + f'/{FILES["length_scale_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_'+method_name+'.csv'

            with open(length_scale_hist, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(length_scales.detach().numpy())
            
            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            header.append('y')
            sampling_hist_file = FILES['results_dir'] + f'/{FILES["general_results_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_'+method_name+'.csv'

            with open(sampling_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(torch.concat([x_train, -y_train], axis=1).numpy())

            # generate history of best samples
            data = torch.concat([x_train, -y_train], axis=1)
            best_hist = torch.zeros_like(data)
            current_best = torch.full((PROBLEM_DIMENSION,), torch.inf)

            for i,row in enumerate(data): 
                if row[-1] < current_best[-1]:
                    current_best = row
                best_hist[i,:] = current_best
            
            
            header.append('timestamp')

            with open(optimizer_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(np.concatenate([best_hist.numpy(), np.expand_dims(np.array(time_stamps),axis=1)],axis=1))
            add_ground_truth_objective(FILES,SEED,method_name,PROBLEM_DIMENSION,obj_ground_truth)

    if 'logei' in METHODS:
        logeiloop('logei')
    
    if 'loghvarei' in METHODS:
        logeiloop('loghvarei')

    if 'turbo' in METHODS: 
        optimizer_hist_file = FILES['optimizer_history_dir'] + f'/{FILES["optimizer_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_turbo.csv'
        if os.path.exists(optimizer_hist_file):
            print(f"Path '{optimizer_hist_file}' already exists.") 
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.double
            batch_size = 1
            max_cholesky_size = float("inf")  # Always use Cholesky

            if ANALYTIC_TARGET: 
                objFun_turbo = lambda X: -torch.tensor(obj_noisy(X.numpy()))
            else: 
                raise NotImplementedError()
            
            def get_initial_points(dim):
                tf.random.set_seed(SEED)
                X_init = torch.from_numpy(tf.random.uniform(shape=(NUMBER_OF_INITIAL_SAMPLES,dim),
                                                                minval=0.0,
                                                                maxval=1.0,
                                                                dtype=tf.float64).numpy())
                return X_init
            
            @dataclass
            class TurboState:
                dim: int
                batch_size: int
                length: float = 0.8
                length_min: float = 0.5**7
                length_max: float = 1.6
                failure_counter: int = 0
                failure_tolerance: int = float("nan")  # Note: Post-initialized
                success_counter: int = 0
                success_tolerance: int = 10  # Note: The original paper uses 3
                best_value: float = -float("inf")
                restart_triggered: bool = False

                def __post_init__(self):
                    self.failure_tolerance = math.ceil(
                        max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
                    )

            def update_state(state, Y_next):
                if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
                    state.success_counter += 1
                    state.failure_counter = 0
                else:
                    state.success_counter = 0
                    state.failure_counter += 1

                if state.success_counter == state.success_tolerance:  # Expand trust region
                    state.length = min(2.0 * state.length, state.length_max)
                    state.success_counter = 0
                elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
                    state.length /= 2.0
                    state.failure_counter = 0

                state.best_value = max(state.best_value, max(Y_next).item())
                if state.length < state.length_min:
                    state.restart_triggered = True
                return state

            def generate_batch(
                state,
                model,  # GP model
                X,  # Evaluated points on the domain [0, 1]^d
                Y,  # Function values
                batch_size,
                n_candidates=None,  # Number of candidates for Thompson sampling
                num_restarts=10,
                raw_samples=512,
                acqf="ts",  # "ei" or "ts"
            ):
                assert acqf in ("ts", "ei")
                assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
                if n_candidates is None:
                    n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

                # Scale the TR to be proportional to the lengthscales
                x_center = X[Y.argmax(), :].clone()
                weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
                weights = weights / weights.mean()
                weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
                tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

                if acqf == "ts":
                    dim = X.shape[-1]
                    sobol = SobolEngine(dim, scramble=True)
                    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
                    pert = tr_lb + (tr_ub - tr_lb) * pert

                    # Create a perturbation mask
                    prob_perturb = min(20.0 / dim, 1.0)
                    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
                    ind = torch.where(mask.sum(dim=1) == 0)[0]
                    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

                    # Create candidate points from the perturbations and the mask
                    X_cand = x_center.expand(n_candidates, dim).clone()
                    X_cand[mask] = pert[mask]

                    # Sample on the candidate points
                    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
                    with torch.no_grad():  # We don't need gradients when using TS
                        X_next = thompson_sampling(X_cand, num_samples=batch_size)

                elif acqf == "ei":
                    assert NotImplementedError()

                return X_next
            
            ## Turbo optimization loop start ## 
            X_turbo = get_initial_points(PROBLEM_DIMENSION)
            Y_turbo = objFun_turbo(X_turbo).unsqueeze(-1)
            time_stamps = [time.time()]*NUMBER_OF_INITIAL_SAMPLES

            state = TurboState(PROBLEM_DIMENSION, batch_size=batch_size, best_value=max(Y_turbo).item())

            NUM_RESTARTS = 10 
            RAW_SAMPLES = 512 
            N_CANDIDATES = min(5000, max(2000, 200 * PROBLEM_DIMENSION))

            torch.manual_seed(SEED)
            if WITHIN_MODEL_COMPARISON or len(length_scales_ground_truth) == 0:   
                length_scales = []
            else:
                length_scales = torch.unsqueeze(torch.from_numpy(length_scales_ground_truth),0)

            # TURBO Main optimization loop
            for _ in range(NUMBER_OF_ITERATIONS - NUMBER_OF_INITIAL_SAMPLES):  # Run until TuRBO converges
                if WITHIN_MODEL_COMPARISON:
                    train_Y = Y_turbo 
                else:
                    train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
                model = setup_gpytorch_model_oom(X_turbo, train_Y, PROBLEM_DIMENSION, NOISE_STD**2)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Do the fitting and acquisition function optimization inside the Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    if not WITHIN_MODEL_COMPARISON:
                        fit_gpytorch_mll(mll, max_attempts=20)
                    # Create a batch
                    X_next = generate_batch(
                        state=state,
                        model=model,
                        X=X_turbo,
                        Y=train_Y,
                        batch_size=batch_size,
                        n_candidates=N_CANDIDATES,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                        acqf="ts",
                    )

                Y_next = objFun_turbo(X_next).unsqueeze(-1)
                time_stamps.append(time.time())
                if length_scales == []:
                    length_scales = model.covar_module.base_kernel.lengthscale
                else:
                    length_scales = torch.concat([length_scales, model.covar_module.base_kernel.lengthscale], axis=0)
            
            # length_scales = torch.concat([length_scales, model.covar_module.base_kernel.lengthscale], axis=0)
                # Update state
                state = update_state(state=state, Y_next=Y_next)

                # Append data
                X_turbo = torch.cat((X_turbo, X_next), dim=0)
                Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            if TARGET_FUNCTION == 'gpsample':
                header[-1] = header[-1] + " (first row is ground truth)"
            
            length_scale_hist = FILES['length_scale_history_dir'] + f'/{FILES["length_scale_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_turbo.csv'

            with open(length_scale_hist, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(length_scales.detach().numpy())




            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            header.append('y')
            sampling_hist_file = FILES['results_dir'] + f'/{FILES["general_results_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_turbo.csv'

            with open(sampling_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(torch.concat([X_turbo, -Y_turbo], axis=1).numpy())

            # generate history of best samples
            data = torch.concat([X_turbo, -Y_turbo], axis=1)
            best_hist = torch.zeros_like(data)
            current_best = torch.full((PROBLEM_DIMENSION,), torch.inf)

            for i,row in enumerate(data): 
                if row[-1] < current_best[-1]:
                    current_best = row
                best_hist[i,:] = current_best
            
            header.append('timestamp')

            with open(optimizer_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(np.concatenate([best_hist.numpy(), np.expand_dims(np.array(time_stamps),axis=1)],axis=1))

            add_ground_truth_objective(FILES,SEED,'turbo',PROBLEM_DIMENSION,obj_ground_truth)

    if 'sobol' in METHODS:
        optimizer_hist_file = FILES['optimizer_history_dir'] + f'/{FILES["optimizer_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_sobol.csv'
        if os.path.exists(optimizer_hist_file):
            print(f"Path '{optimizer_hist_file}' already exists.") 
        else:
            if ANALYTIC_TARGET: 
                objFun_sobol = lambda X: torch.tensor(obj_noisy(X.numpy()))
            else: 
                raise NotImplementedError()
            
            sobol_engine = torch.quasirandom.SobolEngine(dimension=PROBLEM_DIMENSION, scramble=True, seed=SEED)
            x = sobol_engine.draw(NUMBER_OF_ITERATIONS)
            y = torch.reshape(objFun_sobol(x), (-1,1))
            
            header = [f'x{i+1}' for i in range(PROBLEM_DIMENSION)] 
            header.append('y')
            sampling_hist_file = FILES['results_dir'] + f'/{FILES["general_results_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_sobol.csv'

            with open(sampling_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(torch.concat([x, y], axis=1).numpy())

            data = torch.concat([x, y], axis=1)
            best_hist = torch.zeros_like(data)
            current_best = torch.full((PROBLEM_DIMENSION,), torch.inf)

            for i,row in enumerate(data): 
                if row[-1] < current_best[-1]:
                    current_best = row
                best_hist[i,:] = current_best
            
            optimizer_hist_file = FILES['optimizer_history_dir'] + f'/{FILES["optimizer_history_file"]}_{SEED:05d}_{PROBLEM_DIMENSION}_sobol.csv'

            with open(optimizer_hist_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(best_hist.numpy())

            add_ground_truth_objective(FILES,SEED,'sobol',PROBLEM_DIMENSION,obj_ground_truth)