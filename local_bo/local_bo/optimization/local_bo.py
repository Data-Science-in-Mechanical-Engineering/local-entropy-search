import tensorflow as tf
import torch
import tensorflow_probability as tfp
import copy
import numpy as np
from gpflow.kernels import SquaredExponential
import gpflow
from gpflow import set_trainable as gp_set_trainable
from gpflow.config import default_float as floatx 
from gpflow_sampling.models import PathwiseGPR 
from gpflow.models import GPR
import matplotlib.pyplot as plt
from local_bo.second_order_model.seard_second_order_model import SEARDSecondOrderModel
from local_bo.second_order_model import fit_derivative_gp # this is the tiny_gp model
from sklearn.preprocessing import StandardScaler
from local_bo.optimization import optimizers
import time as time
import logging
import os
import csv
import jax, jax.numpy as jnp


class Local_bo:
    def __init__(self, 
                 objective_function, 
                 problem_dim: int,  
                 no_initial_samples: int = 5,
                 number_of_entropy_pts: int = 250, 
                 initial_samples = None,  
                 no_total_samples = 20, 
                 lb: tf.Tensor = None, 
                 ub: tf.Tensor = None, 
                 seed: int = 0,
                 acquisition_function_opt_method = 'best_entropy_pt',
                 optimize_hyperparameters: bool = False, 
                 length_scale = 0.2,
                 scale_problem: bool = False,
                 length_scale_bounds_scaled = [0.001, 2.0],
                 noisy_observation = False,
                 observation_noise_std_bound_scaled = [0.001, 1],
                 init_noise_std =1e-3,  
                 use_random_start_pts_for_opt: bool = False,
                 use_derivative_information: bool = True, 
                 use_gd_traces: bool = False,
                 derivative_cond: bool = False,
                 low_level_adam: bool = True,
                 low_level_adam_beta05: bool = False,
                 low_level_CMAES: bool = False,
                 log_level = logging.INFO,
                 log_to_file = False, 
                 log_file = './Logs/local_bo.log',
                 convergence_epsilon = 0.02,
                 plot_at_each_iter: bool = False, 
                 output_files = None,
                 external_hyper_param_opt = None,
                 GD_trace_divisions:int = 8):
        """
        Constructor of Local_bo class. 

        Args: 
            objective_function: Interface to objective function. Is expected to be able to return single sample queries provided as numpy array containing data point
            no_initial_samples (int): Number of initial objective function samples to be chosen at random at beginning of BO
            lb (float tensor): 1 x d tensor defining the lower bounds for any point considered in the problem
            ub (float tensor): 1 x d tensor defining the upper bounds for any point considered in the problem
            seed (int): Tensorflow randomization seed
            plot_at_each_iter (bool): Switch to determine whether results are plotted at every iteration
            optimize_hyperparameters (bool): Switch to determine whether the model's hyperparameters are to be optimized at each iteration
        """
        
        self.objective_function = objective_function # Callable object returning objective function values for given locations
        self.no_initial_samples = no_initial_samples # Number of initial objective function samples to be chosen at random at beginning of BO
        self.total_samples = no_total_samples
        self._STD_TRAINING = init_noise_std                    # sigma_n (uncertainty in observation) - standard deviation
        self._std_training_scaled = self._STD_TRAINING
        self.length_scale_init_scaled = length_scale
        self.noisy_observation = noisy_observation
        self.observation_noise_std_bound_scaled = observation_noise_std_bound_scaled



        self.initial_samples = initial_samples  # Tensor of initial samples/observations for optimization loop
        self.lb = lb
        self.ub = ub
        self.obj_gp_model = None                # GPFlow model of current knowledge about objective function 
        self.cons_gp_model = None

        self.gp_hyperparameter_opt = optimize_hyperparameters
        self.scale_problem = scale_problem 
        self.use_random_start_pts_for_opt = use_random_start_pts_for_opt
        self.length_scale_bounds_scaled = length_scale_bounds_scaled
        self.external_hyper_param_opt = external_hyper_param_opt
        
        self.problem_dim = problem_dim              # Problem dimensionality
        self.seed = seed                            # RNG seed
        self.number_of_entropy_pts = number_of_entropy_pts
        self.number_of_bases = 1024                 # Number of bases for Thompson sampling paths
        self.gradient_descent_steps = 500           # Number of gradient descent steps during Thompson path optimization
        self.use_grad = use_derivative_information  # Switch whether first and second order derivative information are introduced in acquisition function setup. If false, only zeroth order information is used
        self.acquisition_function_opt_method = acquisition_function_opt_method

        self.convergence_epsilon = convergence_epsilon
        self.X = tf.zeros([0,problem_dim],tf.double)        # n x d tensor containing all sampled points (locations)
        self.y = tf.zeros([0,1],tf.double)                  # n x 1 tensor containing all objective function values corresponding to self.X locations
        self.X_scaled = tf.identity(self.X)
        self.y_scaled = tf.identity(self.y)
        self.l = tf.zeros([0],tf.bool) # one if feasible
        self.estimated_feasible = tf.zeros([0,1],tf.bool)

        self.scaler_y = StandardScaler()
        self.objective_scaling = {"mean": tf.zeros(1,tf.double), "std": tf.ones(1,tf.double)}
       
        # Results of local minimization of sample paths
        self.local_opt_dist_y = tf.zeros([0,0])
        self.local_opt_dist_X = tf.zeros([self.number_of_entropy_pts,self.problem_dim])
        self.local_opt_dist_y_scaled = 0   # y-values of local optimization of thompson paths (scaled)
        self.local_opt_dist_X_scaled = tf.zeros([self.number_of_entropy_pts,self.problem_dim]) # X-values of local optimization of thompson paths (scaled)
        if use_gd_traces:
            self.model_based_regret_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            self.local_opt_distr_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True) # Only used in benchmarking to store entire distribution history
        


        # Logging
        self.logger = logging.getLogger(__name__)
        if plot_at_each_iter and self.logger.level > logging.DEBUG:
            self.logger.warning('Plotting at every iteration requires logging level "debug"')
            self.plot_at_each_iter = False
        else:
            self.plot_at_each_iter=plot_at_each_iter
        self.logger.setLevel(log_level)
        if log_to_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        else:
            self.handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)

        self.output_files = output_files
        if self.output_files is not None:
            results_dir = self.output_files['results_dir']
            results_file = self.output_files['general_results_file']
            results_file = f'{results_file}_{self.seed:05d}_{self.problem_dim}_{self.output_files['algo_name']}.csv'
            self.results_filepath = os.path.join(results_dir, results_file)
            
            optimizer_dir = self.output_files['optimizer_history_dir']
            optimizer_file = self.output_files['optimizer_history_file']
            optimizer_file = f'{optimizer_file}_{self.seed:05d}_{self.problem_dim}_{self.output_files['algo_name']}.csv'
            self.optimizer_filepath = os.path.join(optimizer_dir, optimizer_file)
            if os.path.exists(self.optimizer_filepath):
                raise FileExistsError(f"Path '{self.optimizer_filepath}' already exists.")    

            distr_dir = self.output_files['local_optima_dist_dir']
            distr_file = self.output_files['local_optima_dist_file']
            distr_file = f'{distr_file}_{self.seed:05d}_{self.problem_dim}_{self.output_files['algo_name']}.csv'
            self.distribution_filepath = os.path.join(distr_dir, distr_file)

            length_scale_dir = self.output_files['length_scale_history_dir']
            length_scale_file = self.output_files['length_scale_history_file']
            length_scale_file = f'{length_scale_file}_{self.seed:05d}_{self.problem_dim}_{self.output_files['algo_name']}.csv'
            self.length_scale_filepath = os.path.join(length_scale_dir, length_scale_file)

        

        self.stats_best_observed = []                   # List of iteration-wise best observed objective function values
        self.stats_min_location  = []                   # List of iteration-wise locations of minimal observed objective function values
        self.stats_dist_sample_to_min_location  = []
        self.stats_all_evals = []                       # List of iteration-wise number of points in self.X (total number of function evaluations in iteration)
        self.stats_std_of_opt_dist = []
        self.stats_mean_of_opt_dist = []
        self.stats_dist_optsamples_to_min_location = [] 
        self.stats_GP_length_scales_obj = np.zeros([0,self.problem_dim])  
        self.time_stamps = []   

        self.use_gd_traces = use_gd_traces 
        self.derivative_cond = derivative_cond
        self.low_level_adam  = low_level_adam
        self.low_level_adam_beta05 = low_level_adam_beta05
        self.low_level_CMAES = low_level_CMAES
        
        self.GD_trace_divisions = GD_trace_divisions

        
    def close_logger(self):
        self.logger.info('Shutting down logger')
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def randomly_select_initial_samples(self):
        """
        Function initializes self.initial_samples with n x d tensor of uniformly random points within defined lower and upper bounds
        """
        tf.random.set_seed(self.seed)
        if self.initial_samples is None:
            self.initial_samples = tf.random.uniform(shape=(self.no_initial_samples,self.problem_dim),
                                                     minval=self.lb,
                                                     maxval=self.ub,
                                                     dtype=tf.double)
            self.logger.debug('The initial samples were chosen as: ' + str(self.initial_samples.numpy()))
        else:
            self.logger.debug('Initial samples already set - nothing happens.')

    def evaluate_initial_samples(self):
        self.logger.debug('Evaluating %i initial samples:', self.initial_samples.shape[0])
        for row_index in range(self.initial_samples.shape[0]):
            self.evaluate_single_sample(self.initial_samples[row_index:row_index+1,:])
            
    def evaluate_single_sample(self,X_new):
        self.X =  tf.concat([self.X,X_new],axis=0)
        self.logger.debug('Input: %s', self.X[-1,:].numpy())

        returned_list = self.objective_function(X_new)
        obj_fun_val = tf.cast(tf.reshape(returned_list[0],(1,1)),dtype=tf.double)
        self.y = tf.concat([self.y, obj_fun_val],axis=0) 
        self.logger.debug('Output Objective: %s', self.y[-1].numpy())


        self.l = tf.concat([self.l, tf.reshape(tf.cast(True,tf.bool),(1))],axis=0) 
        
        self.update_scaling()

    def update_scaling(self):
        if self.scale_problem:
            self.objective_scaling["mean"] = tf.reduce_mean(tf.boolean_mask(self.y,self.l),axis=0) 
            
            if sum(tf.cast(self.l,tf.double)) > 1:
                self.objective_scaling["std"] = tf.math.sqrt(tf.math.reduce_variance(tf.boolean_mask(self.y,self.l),axis=0))
                self.logger.debug('Objective_scaling: %s',self.objective_scaling["std"].numpy())
            else:
                self.objective_scaling["std"] = tf.ones(1,tf.double)
                self.objective_scaling["mean"] = tf.boolean_mask(self.y,self.l) + tf.ones(1,tf.double) # to ensure that we are better than our estimated optimum in the first iteration

        self.X_scaled = self.scale_domain(self.X)
        self.y_scaled = self.scale(self.y,self.objective_scaling)
        self._std_training_scaled = self._STD_TRAINING # / self.objective_scaling['std']


    def scale(self,vars,scaling_dict):
        """
        Function scales input data by provided dict containing desired mean and standard deviation

        Args: 
            vars (Tensor): Data to be scaled
            scaling_dict (dict): Dict with keys {"mean", "std"} providing mean and standard deviation for scaling
        """
        return (vars - scaling_dict["mean"])/scaling_dict["std"]
    
    def unscale(self,scaled_vars,scaling_dict):
        return (scaled_vars*scaling_dict["std"] + scaling_dict["mean"])

    def scale_domain(self,vars):
        """
        Function scales provided data to upper and lower bounds of problem. No checking if data fits lower and upper bounds done.

        Args: 
            vars (Tensor): Data to be scaled
            
        Returns: 
            vars (Tensor): Data normalized on interval [self.lb, self.ub]
        """
        if self.scale_problem:
            return (vars - self.lb)/(self.ub - self.lb)
        else:
            return vars
    
    def unscale_domain(self,scaled_vars):
        if self.scale_problem:
            return scaled_vars*(self.ub - self.lb) + self.lb
        else:
            return scaled_vars
       

    def fit_gp_model(self):
        """
        Function sets up current best model of objective function.  
        """
        lengthscales_parameter = gpflow.Parameter(tf.ones(self.problem_dim)*self.length_scale_init_scaled, 
                                                  trainable=True, 
                                                  prior=tfp.distributions.Uniform(low=tf.constant (self.length_scale_bounds_scaled[0],dtype=tf.double),
                                                                                  high=tf.constant(self.length_scale_bounds_scaled[1],dtype=tf.double),
                                                                                  validate_args=False,
                                                                                  allow_nan_stats=True,
                                                                                  name='Uniform'))
        noise2 = tf.square(self._STD_TRAINING) # measurement noise variance
        kernel = SquaredExponential(lengthscales=lengthscales_parameter, variance=1)
        
        if self.noisy_observation:
            likelihood = gpflow.likelihoods.Gaussian(variance=(0.5*(self.observation_noise_std_bound_scaled[1] + self.observation_noise_std_bound_scaled[0] )**2))
            likelihood.variance.prior = tfp.distributions.Uniform(low=tf.constant (self.observation_noise_std_bound_scaled[0]**2,dtype=tf.double),
                                                                    high=tf.constant(self.observation_noise_std_bound_scaled[1]**2,dtype=tf.double),
                                                                    validate_args=False,
                                                                    allow_nan_stats=True,
                                                                    name='Uniform')
        else:
            likelihood = gpflow.likelihoods.Gaussian(variance=noise2)
    

        def fit_model(X: tf.Tensor, y: tf.Tensor, kernel,noise2) -> PathwiseGPR:
            """
            Function sets up a PathwiseGPR model based on provided data with self.number_of_entropy_pts paths based on self.number_of_bases basis functions
            """
            mdl = PathwiseGPR((X,y), 
                              kernel=copy.deepcopy(kernel), 
                              likelihood=copy.deepcopy(likelihood))
            
            if self.gp_hyperparameter_opt:
                self.logger.debug('Started hyperparameter tuning')
                if self.noisy_observation:
                    gpflow.set_trainable(mdl.likelihood.variance, True) # Training uncertainty trainable if noise is present
                else:
                    gpflow.set_trainable(mdl.likelihood.variance, False) # Training uncertainty not trainable
                
                
                opt = gpflow.optimizers.Scipy() # See for usage of Adams optimizer: https://gpflow.github.io/GPflow/2.9.1/notebooks/getting_started/parameters_and_their_optimisation.html
               

                if self.external_hyper_param_opt is not None:
                    external_hyp = self.external_hyper_param_opt(X.numpy(),y.numpy(),self.problem_dim,np.sqrt(noise2))
                    print(external_hyp)
                    mdl.kernel.variance.assign(tf.convert_to_tensor(external_hyp["outputscale"]))
                    mdl.kernel.lengthscales.assign(tf.convert_to_tensor(external_hyp["lengthscales"][0,:]))
                    #gpflow.set_trainable(mdl.likelihood.variance, False)
                    print(external_hyp["noisevar"][0])
                    print(tf.convert_to_tensor(external_hyp["noisevar"])[0])
                    print(tf.math.maximum(tf.convert_to_tensor(external_hyp["noisevar"])[0],tf.cast(tf.convert_to_tensor(2.0e-6),tf.double)))
                    mdl.likelihood.variance.assign(tf.math.maximum(tf.convert_to_tensor(external_hyp["noisevar"])[0],tf.cast(tf.convert_to_tensor(2.0e-6),tf.double)))
                    gpflow.utilities.print_summary(mdl)

                else:
                    try:
                        opt.minimize(mdl.training_loss, mdl.trainable_variables)
                        if self.logger.level == 10: 
                            gpflow.utilities.print_summary(mdl)
                    except:
                        self.logger.error("Hyperparameter optimization failed", exc_info=True)
                    self.logger.debug('Finished hyperparameter tuning')
            
            paths = mdl.generate_paths(num_samples=self.number_of_entropy_pts, num_bases=self.number_of_bases)
            mdl.set_paths(paths) 
  
            return mdl
        
        def fit_all_models(X: tf.Tensor, y: tf.Tensor, kernel,noise2):
            obj_gp_model = fit_model(X,y, kernel, noise2)
            self.stats_GP_length_scales_obj = np.concatenate([self.stats_GP_length_scales_obj, np.expand_dims(obj_gp_model.kernel.lengthscales.numpy(),0)],0)
            #stats_GP_length_scales_obj
            return obj_gp_model
        
        self.obj_gp_model = fit_all_models(self.X_scaled,self.y_scaled,kernel,noise2)

    def find_local_optimum_distribution(self):
        """
        Function finds local optima locations and values of all target GP paths
        """
        if self.low_level_CMAES:
            if self.use_random_start_pts_for_opt:
                raise NotImplementedError
            with torch.no_grad():
                a = time.time()
                x_vals_hist = []
                fvals_hist =[]
                for i in range(self.number_of_entropy_pts):
                    #define objective function
                    def cmaes_objective(x,i):
                        tf_tensor = tf.convert_to_tensor(x.numpy(),dtype=tf.double)
                        obj = self.obj_gp_model.predict_f_samples(Xnew=tf_tensor, sample_axis=0) + tf.reduce_sum(tf.cast(tf_tensor > tf.ones_like(tf_tensor),tf.float64))*100  + tf.reduce_sum(tf.cast(tf_tensor < tf.zeros_like(tf_tensor),tf.float64))*100
                        return torch.from_numpy(obj[i].numpy())
                        #return torch.from_numpy(obj[i])
                    #optimizers
                    x_init = torch.from_numpy(self.best_x_scaled.numpy())
                    #x_init = self.best_x_scaled
                    objective_lambda = lambda x: cmaes_objective(x,i)
                    optimizer = optimizers.CMAES(x_init,objective_lambda,maximization = False, verbose=False,sigma = 0.5)
                    for _ in range(50):
                        optimizer.step()
                    x_vals_hist.append(tf.stack(optimizer.params_history_list,0))
                    fvals_hist.append(tf.stack(optimizer.objective_history_list,0))
                    b = time.time()
                    eval_time = b-a
                    print(f"Sample No. {i} elapsed time: {eval_time}" )
                xvals_hist_stacked= tf.concat(x_vals_hist,axis=1)
                fvals_hist_stacked= tf.stack(fvals_hist,1)
                X = xvals_hist_stacked[-1,:,:]
                y = fvals_hist_stacked[-1,:]

        else:   
            if self.use_random_start_pts_for_opt:
                raise NotImplementedError
            else:
                self.logger.info('Starting local optimization of sample paths at current best location')
                x_start = self.best_x_scaled
                Xinit = tf.ones([self.number_of_entropy_pts, 1, self.problem_dim], dtype=floatx())*x_start
            
            Xvars = tf.Variable(Xinit, constraint=lambda x: tf.clip_by_value(x, tf.zeros((1,self.problem_dim),dtype=tf.double), tf.ones((1,self.problem_dim),dtype=tf.double)))
            
            previous_X = tf.Variable(Xvars)
            # set up 
            fvals_hist = []
            fgrad_hist = []
            xvals_hist = []

            @tf.function
            def closure(sample_axis=0,barrier_scaling = 0.01):
                """
                Passing sample_axis=0 indicates that the 0-th axis of Xnew 
                should be evaluated 1-to-1 with the individuals paths.
                """
                return self.obj_gp_model.predict_f_samples(Xnew=Xvars, sample_axis=sample_axis)
                

            
            learning_rates          = [None]
            steps                   = [500]
            barrier_fun_penalties   = [0]
            
            for i_optimization in range(len(learning_rates)):
                if self.low_level_adam:
                    if self.low_level_adam_beta05:
                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.002,beta_1=0.5)
                    else:
                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.002) # use adam with default parameters but 2 times increased learning rate
                else:
                    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001) 

                for step in range(steps[i_optimization]):
                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(Xvars)

                        fvals = closure()
                        
                    grads = tape.gradient(fvals, Xvars)

                    
                    optimizer.apply_gradients([(grads, Xvars)])#optimizer.update_step(grads, Xvars,0.001 ) # default learning rate as of https://github.com/keras-team/keras/blob/v3.3.3/keras/src/optimizers/adam.py#L115
                    
                    fvals_hist.append(tf.squeeze(fvals))
                    if self.use_gd_traces:
                        xvals_hist.append(tf.squeeze(Xvars))
                        if self.derivative_cond:
                            fgrad_hist.append(tf.squeeze(grads))

        
                
            X = tf.squeeze(Xvars,axis=1)
            y = self.obj_gp_model.predict_f_samples(Xnew=Xvars, sample_axis=0)
            y = tf.squeeze(y)
                   
            if self.use_gd_traces:
                xvals_hist_stacked = tf.stack(xvals_hist)
                fvals_hist_stacked = tf.stack(fvals_hist)
                fgrads_hist_stacked = tf.stack(fgrad_hist)

        if self.use_gd_traces:  
            diff_x = tf.experimental.numpy.diff(xvals_hist_stacked , axis=0)

            # Compute the quadratic (L2) norm along the last dimension (-1)
            norm_diff_x = tf.norm(diff_x, ord='euclidean', axis=-1)

            # Now calculate the cumulative sum over the first dimension (axis=0)
            cumsum_norm_diff_x = tf.cumsum(norm_diff_x, axis=0)

            # Find the maximum cumulative sum along the first axis for each entry in the second axis
            max_cumsum = tf.reduce_max(cumsum_norm_diff_x, axis=0)

            indices = np.zeros((self.number_of_entropy_pts,self.GD_trace_divisions))
            trace_points  = np.zeros((self.number_of_entropy_pts,self.GD_trace_divisions,self.problem_dim)) 
            trace_vals    = np.zeros((self.number_of_entropy_pts,self.GD_trace_divisions))
            if self.derivative_cond:
                x0_grads      = np.zeros((self.number_of_entropy_pts,self.problem_dim)) 
                trace_grads   = np.zeros((self.number_of_entropy_pts,self.GD_trace_divisions,self.problem_dim))

            for i in range(self.number_of_entropy_pts):
                if self.derivative_cond:
                    x0_grads[i,:] = fgrads_hist_stacked[0,i,:]

            # Divide the maximum cumulative sum into 10 equal parts (generate thresholds)
                thresholds = tf.cast(tf.linspace(0.0, 1.0, num=self.GD_trace_divisions+1),tf.double)*max_cumsum[i]  # 10 thresholds

                # Initialize a tensor to store the indices where cumulative sum crosses each threshold
                # indices = tf.TensorArray(dtype=tf.double, size=(self.number_of_entropy_pts,self.GD_trace_divisions))

                # Find the indices where cumulative sum crosses each threshold
                for ii in range(self.GD_trace_divisions):
                    threshold = thresholds[ii+1]  # Current threshold
                    condition = cumsum_norm_diff_x[:,i] >= threshold  # Boolean condition where cumsum crosses the threshold
                    index = tf.argmax(tf.cast(condition, tf.int32), axis=0)  # Find the first index where the condition is met
                    #indices[i,ii] = index
                    trace_points[i,ii,:] = xvals_hist_stacked[index,i,:] 
                    trace_vals[i,ii] = fvals_hist_stacked[index,i]
                    if self.derivative_cond:
                        trace_grads[i,ii,:] = fgrads_hist_stacked[index,i,:]    
                    


            #local_trace_dist_X_scaled = 
            self.current_point_y_scaled_dist = fvals_hist_stacked[0,:]
            self.trace_dist_X_scaled        = tf.convert_to_tensor(trace_points, dtype=tf.double)
            self.trace_dist_y_scaled        = tf.convert_to_tensor(trace_vals, dtype=tf.double)
            if self.derivative_cond:
                self.x0_grads_scaled        = x0_grads 
                self.trace_grads_scaled     = trace_grads




            self.model_based_regret_dist_scaled = self.current_point_y_scaled_dist - y  
            self.converged_indicator =  self.model_based_regret_dist_scaled < self.convergence_epsilon
            self.trace_dist_X = self.unscale_domain(tf.convert_to_tensor(trace_points, dtype=tf.double))
            self.trace_dist_y =  self.unscale(self.trace_dist_y_scaled,self.objective_scaling)

        
        self.local_opt_dist_X_scaled = X
        self.local_opt_dist_y_scaled = y


        self.local_opt_dist_X = self.unscale_domain(self.local_opt_dist_X_scaled) 
        self.local_opt_dist_y = self.unscale(self.local_opt_dist_y_scaled,self.objective_scaling)
       
        if self.logger.getEffectiveLevel() == logging.DEBUG:
            if not isinstance(self.handler, logging.FileHandler): # only add plots if not logging to file 
                fig, ax = plt.subplots(figsize=(7, 3))
                if self.low_level_CMAES:
                    for i in range(self.number_of_entropy_pts):
                        ax.plot(fvals_hist_stacked[:,i])
                else:
                    ax.plot(fvals_hist)
                plt.show(block=False)

                self.logger.warning("todo: add plots!")
            
        self.logger.info('Finished local optimization of sample paths')

    def prepare_acquisition_function(self,return_gradients = True):
        # Create a list of GPs with virtual data points, retrieve gradient information at virtual data point and set up GP with gradient information
        augmented_GPs = []
        derivative_GPs = []
        
        if self.use_gd_traces:
        # Compute derivatives for each local minimum when evaluating the corresponding path at that point
            a = time.time()
            if self.use_grad:
                raise NotImplementedError('Gradient information of final point not included, if we condition on traces ')
            if self.derivative_cond:
                for X_virt,ygradtrace_virt,ygradx0_virt  in zip(self.trace_dist_X_scaled,self.trace_grads_scaled,self.x0_grads_scaled):
                    X_virt_all = tf.concat([tf.expand_dims(self.best_x_scaled,0),X_virt],axis=0).numpy()
                    y_grad_virt_all = np.concatenate([np.expand_dims(ygradx0_virt,0),ygradtrace_virt],axis=0)
                    derivative_GPs.append(self.set_up_derivative_trace_GP(X_virt_all,y_grad_virt_all))
            else:
                for X_virt,y_virt in zip(self.trace_dist_X_scaled,self.trace_dist_y_scaled):
                    derivative_GPs.append(self.add_virt_obs(X_virt,y_virt))
            b=time.time()
            self.logger.debug(f'Setup of {self.local_opt_dist_X_scaled.shape[0]} augmented GPs took {b-a}')
        else: 
            # Set up a derivative-GP for every path and add the local optimum as a virtual point with computed derivatives
            a = time.time()
            if self.number_of_entropy_pts == 1:
                der_GP = self.set_up_derivative_GP(self.local_opt_dist_X_scaled, None, None, None)
                derivative_GPs.append(der_GP)
            else:
                for i, (X_virt, y_virt) in enumerate(zip(self.local_opt_dist_X_scaled,self.local_opt_dist_y_scaled)):
                    #der_GP = self.set_up_derivative_GP(tf.expand_dims(X_virt, axis=0), tf.expand_dims(y_virt, axis=0), tf.transpose(df_dx[i:i+1,:]) if self.use_grad else None, tf.transpose(ddf_ddx[i:i+1,:]) if self.use_grad else None) # Transposing is inefficient. Maybe change the base model code!
                    der_GP = self.set_up_derivative_GP(tf.expand_dims(X_virt, axis=0), None, None, None)
                    derivative_GPs.append(der_GP)
            b=time.time()
            self.logger.debug('Setup of %i derivative models took %.5fs', self.local_opt_dist_X_scaled.shape[0], b-a)

        if return_gradients:
            raise NotImplementedError('Gradient in acquisition function not yet supported?')
            def acquisition_function_with_gradients(augmented_GPs, x):
                # Use tf.GradientTape to compute gradients
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    neg_acq = -self.mutual_information_les(augmented_GPs, x)
            
                # Compute gradients of neg_acq with respect to x
                gradients = tape.gradient(neg_acq, x)
            
                return neg_acq, gradients
            
            self.negative_acquisition_function = lambda x: acquisition_function_with_gradients(augmented_GPs, x)
        else:
            def acquisition_function_with_gradients(virtual_GPs, x):
                neg_acq = -self.mutual_information_les(virtual_GPs, x)
                gradients = 0.0
                return neg_acq, gradients
            self.negative_acquisition_function = lambda x: acquisition_function_with_gradients(derivative_GPs, x)
    
    def mutual_information_les(self,augmented_GPs,X_query):
        
        # Calculate entropy of original prediction
        _,queryVar_ = self.obj_gp_model.predict_f(tf.expand_dims(X_query,axis=0))
        queryVar = tf.squeeze(queryVar_)
        entropy = 0.5 * tf.math.log(2 * np.pi * np.e * queryVar)
        cond_entropies = tf.Variable(tf.zeros((X_query.shape[0],self.number_of_entropy_pts),dtype=tf.double),dtype=tf.double)
        for i in range(len(augmented_GPs)):
            try:
                if self.use_gd_traces:
                    if self.derivative_cond:
                        prediction = augmented_GPs[i](jnp.array(X_query.numpy()),jnp.zeros_like(jnp.array(X_query.numpy())))
                        cond_query_var = tf.convert_to_tensor(prediction.variance, dtype=tf.float64)
                    
                    else:
                        _,cond_query_var_ = augmented_GPs[i].predict_f(tf.expand_dims(X_query,axis=0))
                        cond_query_var = tf.squeeze(cond_query_var_)
                else:
                    cond_query_var = augmented_GPs[i].computePosteriorVariance(X_query)
            except:
                self.logger.error('Instable model number %i', i)

            cond_entropies[:,i].assign(0.5 * tf.math.log(2 * np.pi * np.e * cond_query_var))      
        mutual_information = entropy - tf.reduce_mean(cond_entropies,axis=1)

        return mutual_information

    def set_up_derivative_GP(self, X_virt, y_virt=None, dy=None, ddy=None):
        """
        Function sets up GP with derivative information based on the current data, a given an existing GP model, one virtual observation 
        and derivative information at that location.
        """    
        # Possibly more efficient to store length scale and variances after optimization to avoid multiple accesses to model object
        lengthScales = self.obj_gp_model.kernel.lengthscales.numpy()
        s_f = tf.sqrt(self.obj_gp_model.kernel.variance.numpy())

        X_combined =  tf.concat([self.X_scaled,tf.reshape(X_virt,(1,self.problem_dim))],axis=0)
        return SEARDSecondOrderModel(zeroOrderX = X_combined, 
                                     zeroOrderY = self.y_scaled,
                                     firstOrderX = X_virt if self.use_grad else None,
                                     firstOrderY = dy,
                                     secondOrderX = X_virt if self.use_grad else None,
                                     secondOrderY = ddy,
                                     lengthScales = lengthScales,
                                     sigma_f = s_f,
                                     sigma_train = self._std_training_scaled)

    def set_up_derivative_trace_GP(self, Xgrad_virt, ygrad_virt):
        """
        Function sets up GP with derivative information based on the current data, a given an existing GP model, one virtual observation 
        and derivative information at that location.
        """    
        # Possibly more efficient to store length scale and variances after optimization to avoid multiple accesses to model object
        lengthScales = self.obj_gp_model.kernel.lengthscales.numpy()
        s_f = tf.sqrt(self.obj_gp_model.kernel.variance).numpy()
        s_noise = tf.sqrt(self.obj_gp_model.likelihood.variance).numpy()
        self.logger.info('Fit augmented GP with gradient information')



        params, predict = fit_derivative_gp(jnp.array(self.X_scaled.numpy()), jnp.array(self.y_scaled.numpy())[:,0], jnp.array(Xgrad_virt), jnp.array(ygrad_virt),
                                            log_amp=jnp.log(s_f), log_scale=jnp.log(lengthScales), log_noise=jnp.log(s_noise),
                                            optimise=False) # TODO


        if False: 
            random_samples = tf.random.uniform(shape=(10,self.problem_dim),
                                                     minval=self.lb*0,
                                                     maxval=self.ub/self.ub,
                                                     dtype=tf.double)
            
            _, predict_test = fit_derivative_gp(jnp.array(self.X_scaled.numpy()), jnp.array(self.y_scaled.numpy())[:,0],jnp.zeros((0,self.problem_dim)), jnp.zeros((0,self.problem_dim)),
                                            log_amp=jnp.log(s_f), log_scale=jnp.log(lengthScales), log_noise=jnp.log(s_noise),
                                            optimise=False) # TODO
            
            tiny_pred = predict_test(jnp.array(random_samples.numpy()),jnp.zeros_like(jnp.array(random_samples.numpy())))
            gpytorch_pred_mean, gpytorch_pred_var = self.obj_gp_model.predict_f(random_samples)
            
            
            delta_mean = np.max(np.absolute(np.array(tiny_pred.mean) - tf.squeeze(gpytorch_pred_mean).numpy()))
            delta_var = np.max(np.absolute(np.array(tiny_pred.variance) - tf.squeeze(gpytorch_pred_var).numpy()))

            print('Max Delta Mean Pred ' + str(delta_mean)) 
            print('Max Delta Var Pred ' + str(delta_var)) 
        return predict
    

    def add_virt_obs(self,X_virt,y_virt):
        if self.use_gd_traces:
            X_combined =  tf.concat([self.X_scaled,X_virt],axis=0)
            y_combined = tf.concat([self.y_scaled, tf.reshape(y_virt,(self.GD_trace_divisions,1))],axis=0)
        else:
            X_combined =  tf.concat([self.X_scaled,tf.reshape(X_virt,(1,self.problem_dim))],axis=0)
            y_combined = tf.concat([self.y_scaled, tf.reshape(y_virt,(1,1))],axis=0)

        return GPR(data=(X_combined,y_combined), kernel=self.obj_gp_model.kernel, likelihood=self.obj_gp_model.likelihood)

        
    def optimize_acquisition_function(self, method = 'best_entropy_pt'):
        self.logger.info('Beginning of acquisition function optimization')
        if method == 'best_entropy_pt':
            start_time = time.time()
            if self.use_gd_traces:
                query_pts_scaled = tf.reshape(self.trace_dist_X_scaled,(self.trace_dist_X_scaled.shape[0]*self.trace_dist_X_scaled.shape[1],self.problem_dim))
                query_pts   = tf.reshape(self.trace_dist_X,(self.trace_dist_X_scaled.shape[0]*self.trace_dist_X_scaled.shape[1],self.problem_dim))
            else:
                query_pts_scaled = self.local_opt_dist_X_scaled
                query_pts = self.local_opt_dist_X 
            acquisition_function_val, _ = self.negative_acquisition_function(query_pts_scaled)
            end_time = time.time()
            self.logger.debug('Number of query locations: %i', self.local_opt_dist_X.shape[0])
            self.logger.debug('Time per query evaluation: %.5f', (end_time - start_time)/(self.local_opt_dist_X.shape[0]))
            self.logger.debug('Total time: %.5f', end_time - start_time)
            minInd = tf.argmin(acquisition_function_val)
            self.logger.info('End of acquisition function optimization')
            return tf.expand_dims(query_pts[minInd,0:self.problem_dim],axis=0)
        else:
            raise NotImplementedError("Not implemented")
        
    

    def evaluate_new_sample(self, new_sample):
        self.X = tf.concat([self.X, [new_sample]], axis=0)
        self.y = tf.concat([self.y, [self.objective_function(new_sample)]], axis=0)

    def log_iteration_stats_to_file(self, iteration: int):
        self.stats_best_observed.append(self.best_y)
        self.stats_min_location.append(self.best_x)
        self.time_stamps.append(time.time())

        iteration_vec = tf.cast(tf.fill([self.local_opt_dist_X.shape[0], 1], iteration), tf.float32)
        path = tf.cast(tf.range(1,self.number_of_entropy_pts+1), tf.float32)
        data = tf.concat([iteration_vec, tf.reshape(path,[-1,1]), tf.cast(self.local_opt_dist_X, tf.float32), tf.reshape(tf.cast(self.local_opt_dist_y, tf.float32), [-1,1])], 1)
        if self.use_gd_traces:
            self.local_opt_distr_history = self.local_opt_distr_history.write(self.local_opt_distr_history.size(), data)
            self.model_based_regret_history = self.model_based_regret_history.write(self.model_based_regret_history.size(),tf.cast(self.model_based_regret_dist_scaled, tf.float32))
    
    def find_current_best(self):

        self.estimated_feasible = self.l

        min_ind = tf.argmin(self.y + tf.expand_dims(tf.cast(~self.estimated_feasible,dtype=tf.double)*(tf.reduce_max(self.y) - tf.reduce_min(self.y))*100,axis=1))

        self.best_x = self.X[int(min_ind),:]
        self.best_y = self.y[int(min_ind),:]
        self.best_x_scaled = self.X_scaled[int(min_ind),:]
        self.best_y_scaled = self.y_scaled[int(min_ind),:]

    def cluster_study(self):
        ## OPTIMIZATION
        start_time = time.time()
        self.randomly_select_initial_samples()
        self.evaluate_initial_samples()

        count = 0
        while self.X.shape[0] + 1 < self.total_samples:     
            self.logger.info("####### Iteration number %i starts #######", count)
            self.fit_gp_model()
            self.find_current_best()
            self.logger.info("Current best location: %s", self.best_x.numpy())
            self.logger.info("Current best objective value: %s", self.best_y.numpy())
            self.find_local_optimum_distribution()
            self.prepare_acquisition_function(return_gradients = False)
            best_x = self.optimize_acquisition_function(method=self.acquisition_function_opt_method)

            self.evaluate_single_sample(best_x)
            self.log_iteration_stats_to_file(count)
            count+=1

            self.clean_up_after_iteration()
          
        end_time = time.time()

        # Add final observation to data
        self.find_current_best()
        self.stats_best_observed.append(self.best_y)
        self.stats_min_location.append(self.best_x)
        self.time_stamps.append(end_time)

        self.close_logger()

        header = [f'x{i+1}' for i in range(self.problem_dim)]
        header.append('y')
        with open(self.results_filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            sampled_data = tf.concat([self.X, self.y], 1)
            writer.writerows(sampled_data.numpy())
            writer.writerow([f'Total runtime: {end_time-start_time}s'])

        header_ = copy.deepcopy(header)
        header_.append('timestamp')
        with open(self.optimizer_filepath, 'w', newline='') as file: 
            writer = csv.writer(file)
            writer.writerow(header_)
            best_hist = tf.concat([self.stats_min_location, self.stats_best_observed], 1)
            # Convert the list of timestamps to a TensorFlow tensor, reshape to [N, 1]
            time_tensor = tf.convert_to_tensor(self.time_stamps, dtype=best_hist.dtype)
            time_tensor = tf.reshape(time_tensor, [-1, 1])

            # Concatenate timestamps to best_hist => new shape [N, (problem_dim + 2)]
            best_hist_with_time = tf.concat([best_hist, time_tensor], axis=1)
            writer.writerows(best_hist_with_time.numpy())


        if self.use_gd_traces:
            header.insert(0, 'Iteration')
            header.insert(1, 'Path')
            header.append("rhat")
            with open(self.distribution_filepath, 'w', newline='') as file: 
                writer = csv.writer(file)
                writer.writerow(header)
                concatdata = tf.concat([self.local_opt_distr_history.stack(), tf.expand_dims(self.model_based_regret_history.stack(),-1)],2)
                writer.writerows(tf.reshape(concatdata, (-1, self.problem_dim + 4)).numpy()) # Problem dim + 3: all x values, y value , path and iteration, and local regret
              
        header = [f'x{i+1}' for i in range(self.problem_dim)]

        with open(self.length_scale_filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(self.stats_GP_length_scales_obj)



            # model_based_regret_history
            
    def clean_up_after_iteration(self):
        del self.local_opt_dist_y
        del self.local_opt_dist_X
        del self.local_opt_dist_y_scaled
        del self.local_opt_dist_X_scaled
        del self.negative_acquisition_function

        tf.keras.backend.clear_session()

        self.local_opt_dist_y = tf.zeros([0,0])
        self.local_opt_dist_X = tf.zeros([self.number_of_entropy_pts,self.problem_dim])
        self.local_opt_dist_y_scaled = 0   # y-values of local optimization of thompson paths (scaled)
        self.local_opt_dist_X_scaled = tf.zeros([self.number_of_entropy_pts,self.problem_dim]) # X-values of local optimization of thompson paths (scaled)
