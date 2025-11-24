import tensorflow as tf
from abc import ABC, abstractmethod
from local_bo.utils.enums import ComputationMode
   
class SecondOrderGPModel(ABC):
    '''
    Currently only standard GP model for testing!!!
    '''

    def __init__(self, 
                 zeroOrderX: tf.Tensor,
                 zeroOrderY: tf.Tensor,
                 firstOrderX: None,
                 firstOrderY: None,
                 secondOrderX = None,
                 secondOrderY = None,
                 sigma_train: float = 1e-8):
        
        # TODO: Input checking for consistency of dimensions of tensors
        self._DIMS = tf.shape(zeroOrderX)[1]

        self.zerothOrderX = zeroOrderX
        self.zerothOrderY = zeroOrderY
        self.firstOrderX = firstOrderX
        self.firstOrderY = firstOrderY
        self.secondOrderX = secondOrderX
        self.secondOrderY = secondOrderY

        if self.firstOrderX is not None: 
            if self.secondOrderX is not None: 
                self.computationCase = ComputationMode.SECOND_ORDER
            else:
                self.computationCase = ComputationMode.FIRST_ORDER
        else: 
            if self.secondOrderX is not None: 
                raise ValueError('Cannot process only zeroth and second order information.')
            self.computationCase = ComputationMode.ZEROTH_ORDER

        if not isinstance(sigma_train, (int,float)):
            raise ValueError(f'Expected numeric scalar for sigma_train')    
        else:
            self.sigma_train_sq = tf.cast(tf.square(sigma_train), dtype=tf.double)
    
    def computePosteriorMean(self, Xtest: tf.Tensor) -> tf.Tensor:
        # TODO: Add non-zero mean support
        """
        Computes the posterior mean of a GP trained on the training data provided when constructing the model or added later.
        Assumes zero mean. May be overridden by child classes to leverage knowledge about the kernel function for computational speed-up

        Args: 
            Xtest (Tensor): n x d matrix of test locations. 

        Returns: 
            Tensor (one-dimensional) containing the posterior mean at test locations.
        """

        match self.computationCase:
            case ComputationMode.ZEROTH_ORDER:
                KTrainChol = tf.linalg.cholesky(self.zeroOrderKernelMatrix(self.zerothOrderX, self.zerothOrderX, full=True) + self.sigma_train_sq * tf.eye(self.zerothOrderX.shape[0], dtype=tf.double))
                return tf.matmul(tf.transpose(tf.linalg.solve(KTrainChol, tf.transpose(self.zeroOrderKernelMatrix(Xtest, self.zerothOrderX, full=True)))), 
                         tf.linalg.solve(KTrainChol, self.zerothOrderY))
            case ComputationMode.FIRST_ORDER:
                Kxx1 = self.firstOrderKernelMatrix_fdf(Xtest, self.firstOrderX)
                Kx0x1 = self.firstOrderKernelMatrix_fdf(self.zerothOrderX, self.firstOrderX)
                Kx1x1_inv = tf.linalg.diag(tf.math.reciprocal(tf.linalg.diag_part(self.firstOrderKernelMatrix_dfdf(self.firstOrderX, self.firstOrderX)
                                                                                + self.sigma_train_sq * tf.eye(self.firstOrderX.shape[0], dtype=tf.double))))
                Kx1x0 = tf.transpose(Kx0x1) #self.firstOrderKernelMatrix_dff(self.firstOrderX, self.zerothOrderX)

                B4 = self.zeroOrderKernelMatrix(self.zerothOrderX, self.zerothOrderX, full=True) + self.sigma_train_sq * tf.eye(self.zerothOrderX.shape[0]) - (Kx0x1 @ Kx1x1_inv @ Kx1x0)
                B2 = self.zeroOrderKernelMatrix(Xtest, self.zerothOrderX, full=True) - (Kxx1 @ Kx1x1_inv @ Kx1x0)
                B4_inv = tf.linalg.inv(B4)

                return (Kxx1 - B2 @ B4_inv @ Kx0x1) @ Kx1x1_inv @ self.firstOrderY + B2 @ B4_inv @ self.zerothOrderY
            case ComputationMode.SECOND_ORDER:
                Kxx0 = self.zeroOrderKernelMatrix(Xtest, self.zerothOrderX, full=True)
                Kx0x0 = self.zeroOrderKernelMatrix(self.zerothOrderX, self.zerothOrderX, full=True)
                Kxx1 = self.firstOrderKernelMatrix_fdf(Xtest, self.firstOrderX)
                Kx1x1 = self.firstOrderKernelMatrix_dfdf(self.firstOrderX, self.firstOrderX)
                Kx0x1 = self.firstOrderKernelMatrix_fdf(self.zerothOrderX, self.firstOrderX)
                Kxx2 = self.secondOrderKernelMatrix_fddf(Xtest, self.secondOrderX)
                Kx2x1 = self.secondOrderKernelMatrix_ddfdf(self.secondOrderX, self.firstOrderX)
                Kx0x2 = self.secondOrderKernelMatrix_fddf(self.zerothOrderX, self.secondOrderX)
                Kx2x2_inv = tf.linalg.inv(self.secondOrderKernelMatrix_ddfddf(self.secondOrderX, self.secondOrderX) + self.sigma_train_sq * tf.eye(self._DIMS, dtype=tf.double))

                Astar = Kxx2 @ Kx2x2_inv
                A0 = Kx0x2 @ Kx2x2_inv
                A1 = tf.transpose(Kx2x1) @ Kx2x2_inv

                Y2 = Kx0x0 + self.sigma_train_sq * tf.eye(self.zerothOrderX.shape[0], dtype=tf.double) - A0 @ tf.transpose(Kx0x2)
                Y3_inv = tf.linalg.inv(Kx1x1 + self.sigma_train_sq * tf.eye(self._DIMS, dtype=tf.double) - A1 @ Kx2x1)
                Y4 = Kxx0 - Astar @ tf.transpose(Kx0x2)
                Y5 = Kx0x1 - A0 @ Kx2x1
                Y6 = Kxx1 - Astar @ Kx2x1

                Z2_inv = tf.linalg.inv(Y2 - Y5 @ Y3_inv @ tf.transpose(Y5))
                Z3 = Y4 - Y6 @ Y3_inv @ tf.transpose(Y5)

                Y6Y3 = Y6 @ Y3_inv
                Y5Y3 = Y5 @ Y3_inv
                Z3Z2 = Z3 @ Z2_inv

                dy2 = (Astar - Y6Y3 @ A1 - Z3Z2 @ (A0 - Y5Y3 @ A1)) @ self.secondOrderY
                dy1 = (Y6Y3 - Z3Z2 @ Y5Y3) @ self.firstOrderY
                dy0 = Z3Z2 @ self.zerothOrderY
                return dy2 + dy1 + dy0

    def computePosteriorVariance(self, Xtest: tf.Tensor, full: bool = False):
        """
        Computes the posterior variance of a GP trained on the training data provided when constructing the model or added later.
        Only computes the variance at test locations (no covariances) if full is False. 
        May be overridden by child classes to leverage knowledge about the kernel function for computational speed-up.

        Args: 
            Xtest (Tensor): n x d matrix of test locations. 
            full (boolean): Switch for computation of full covariance matrix

        Returns: 
            Tensor containing the posterior variances (optional: covariances) at test locations.
        """

        if full: 
            # TODO: Computation of full covariance matrix if required
            raise NotImplementedError('Currently no support for full covariance matrix')
        else: 
            raise NotImplementedError()           

    @abstractmethod
    def zeroOrderKernelMatrix(self, X1: tf.Tensor, X2: tf.Tensor, full:bool = False):
        """"
        Computes the zeroth order kernel matrix K(X1, X2).

        Args: 
            X1 (Tensor): n x d matrix of points defining the rows of K(X1, X2)
            X2 (Tensor): n x d matrix of points defining the columns of K(X1, X2)
            full (boolean): Switch for computation of main diagonal only

        Returns: 
            Either tensor containing the full covariance matrix K(X1, X2) or it's main diagonal
        """
        pass
    
    @abstractmethod
    def firstOrderKernelMatrix_dff(self, dX1: tf.Tensor, X2: tf.Tensor):
        """"
        Computes the first order kernel matrix ∇K(X1, X2), differentiated with respect to X1. 

        Args: 
            dX1 (Tensor): n x d matrix of first order points defining the rows of ∇K(X1, X2)
            X2 (Tensor): n x d matrix of points defining the columns of ∇K(X1, X2)

        Returns: 
            Tensor containing the full covariance matrix ∇K(X1, X2)
        """
        pass

    def firstOrderKernelMatrix_fdf(self, X1: tf.Tensor, dX2: tf.Tensor):
        """"
        Computes the first order kernel matrix K(X1, X2)∇, differentiated with respect to X2. 

        Args: 
            X1 (Tensor): n x d matrix of first order points defining the rows of K(X1, X2)∇
            dX2 (Tensor): n x d matrix of points defining the columns of K(X1, X2)∇

        Returns: 
            Tensor containing the full covariance matrix K(X1, X2)∇
        """
        return tf.transpose(self.firstOrderKernelMatrix_dff(dX2, X1))

    @abstractmethod
    def firstOrderKernelMatrix_dfdf(self, dX1: tf.Tensor, dX2: tf.Tensor):
        """"
        Computes the first order kernel matrix ∇K(X1, X2)∇, differentiated with respect to X1 and X2. 

        Args: 
            dX1 (Tensor): n x d matrix of first order points defining the rows of ∇K(X1, X2)∇
            dX2 (Tensor): n x d matrix of first order points defining the columns of ∇K(X1, X2)∇
            full (boolean): Switch whether only the main diagonal of the covariance matrix or all elements are to be computed

        Returns: 
            Tensor containing the full covariance matrix ∇K(X1, X2)∇
        """
        pass

    @abstractmethod
    def secondOrderKernelMatrix_ddff(self, ddX1: tf.Tensor, X2: tf.Tensor):
        """"
        Computes the second order kernel matrix ∇²K(X1, X2), differentiated with respect to X1. 

        Args: 
            ddX1 (Tensor): n x d matrix of second order points defining the rows of ∇²K(X1, X2)
            X2 (Tensor): n x d matrix of zeroth order points defining the columns of ∇²K(X1, X2)

        Returns: 
            Tensor containing the full covariance matrix ∇²K(X1, X2) 
        """
        pass

    def secondOrderKernelMatrix_fddf(self, X1: tf.Tensor, ddX2: tf.Tensor):
        """"
        Computes the second order kernel matrix K(X1, X2)∇², differentiated with respect to X2. 

        Args: 
            X1 (Tensor): n x d matrix of zeroth order points defining the rows of K(X1, X2)∇²
            ddX2 (Tensor): n x d matrix of second order points defining the columns of K(X1, X2)∇²

        Returns: 
            Tensor containing the full covariance matrix K(X1, X2)∇²
        """
        return tf.transpose(self.secondOrderKernelMatrix_ddff(ddX2, X1))
    
    @abstractmethod
    def secondOrderKernelMatrix_ddfdf(self, ddX1: tf.Tensor, dX2: tf.Tensor):
        """"
        Computes the second order kernel matrix ∇²K(X1, X2)∇, differentiated with respect to X1 and X2. 

        Args: 
            ddX1 (Tensor): n x d matrix of second order points defining the rows of ∇²K(X1, X2)∇
            X2 (Tensor): n x d matrix of first order points defining the columns of ∇²K(X1, X2)∇

        Returns: 
            Tensor containing the full covariance matrix ∇²K(X1, X2)∇
        """
        pass

    def secondOrderKernelMatrix_dfddf(self, dX1: tf.Tensor, ddX2: tf.Tensor):
        """"
        Computes the second order kernel matrix ∇K(X1, X2)∇², differentiated with respect to X2. 

        Args: 
            X1 (Tensor): n x d matrix of first order points defining the rows of ∇K(X1, X2)∇²
            ddX2 (Tensor): n x d matrix of second order points defining the columns of ∇K(X1, X2)∇²

        Returns: 
            Tensor containing the full covariance matrix ∇K(X1, X2)∇²
        """
        return tf.transpose(self.secondOrderKernelMatrix_ddfdf(ddX2, dX1, full=True))
    
    @abstractmethod
    def secondOrderKernelMatrix_ddfddf(self, ddX1: tf.Tensor, ddX2: tf.Tensor):
        """"
        Computes the second order kernel matrix ∇²K(X1, X2)∇², differentiated with respect to X1 and X2. 

        Args: 
            ddX1 (Tensor): n x d matrix of second order points defining the rows of ∇²K(X1, X2)∇²
            X2 (Tensor): n x d matrix of second order points defining the columns of ∇²K(X1, X2)∇²
            full (boolean): Switch whether only the main diagonal of the covariance matrix or all elements are to be computed

        Returns: 
            Tensor containing the full covariance matrix ∇²K(X1, X2)∇²
        """
        pass
    