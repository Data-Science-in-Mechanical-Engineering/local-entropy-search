from local_bo.second_order_model.base_model import SecondOrderGPModel
from local_bo.utils.decorators import validate_input_tensor
from local_bo.utils.enums import ComputationMode
from local_bo.utils.math_utils import cdist_tf
import tensorflow as tf

class SEARDSecondOrderModel(SecondOrderGPModel):
    def __init__(self, 
                 zeroOrderX: tf.Tensor, 
                 zeroOrderY = None, 
                 firstOrderX = None, 
                 firstOrderY = None, 
                 secondOrderX = None, 
                 secondOrderY = None,
                 sigma_train: float = 0.0,
                 lengthScales: tf.Tensor = None,
                 sigma_f: float = 1.0):
        
        super().__init__(zeroOrderX, zeroOrderY, firstOrderX, firstOrderY, secondOrderX, secondOrderY, sigma_train=sigma_train)

        if self.computationCase == ComputationMode.SECOND_ORDER:
            # Only support |X1| = |X2|= 1
            num_firstOrder, num_secondOrder = tf.shape(self.firstOrderX)[0], tf.shape(self.secondOrderX)[0]
            if num_firstOrder != 1:
                raise ValueError('Provided more than one first order observation.')
            if num_secondOrder != 1:
                raise ValueError('Provided more than one second order observation.')
            
            # Only support X1 = X2
            if tf.reduce_sum(tf.cast(tf.equal(self.firstOrderX,self.secondOrderX),dtype=tf.int16)).numpy() != tf.reduce_prod(tf.shape(self.firstOrderX)).numpy():
                raise ValueError('Expected first and second order observations to be equivalent.')

        if lengthScales is None:
            raise ValueError(f'No lengthscale vector was provided')
        if self._DIMS != lengthScales.shape[0] or lengthScales.ndim != 1:
            raise ValueError(f'Lengthscale vector invalid shape. Expected ({self._DIMS},) but received {lengthScales.shape}')
        self.lengthScales = lengthScales
        self.Lambda_rec = tf.math.reciprocal(tf.cast(tf.math.square(self.lengthScales), dtype=tf.double)) #tf.square returns float32
        
        if not isinstance(sigma_f, (int,float)):
            if tf.is_tensor(sigma_f) and sigma_f.shape == []:
                self.sigma_f_sq = tf.cast(tf.square(sigma_f), dtype=tf.double)
            else:
                raise ValueError(f'Expected numeric scalar for sigma_f')    
        else:
            self.sigma_f_sq = tf.cast(tf.square(sigma_f), dtype=tf.double)
    
    def zeroOrderKernelMatrix(self, X1: tf.Tensor, X2: tf.Tensor, full: bool = False) -> tf.Tensor:
        if not full:
            if X1.shape != X2.shape:
                raise ValueError(f'Cannot compute diagonal matrix for non-constant number of points. Dimension of X1: {X1.shape}. Dimensions of X2: {X2.shape}')
            return tf.fill([X1.shape[0]], self.sigma_f_sq)
        
        else: 
            if X1.ndim == 1:
                X1 = tf.reshape(X1, [-1, 1]) 
            if X2.ndim == 1: 
                X2 = tf.reshape(X2, [-1, 1])   
            X = tf.divide(X1, tf.reshape(self.lengthScales, [1, self._DIMS]))
            Y = tf.divide(X2, tf.reshape(self.lengthScales, [1, self._DIMS]))
            K = cdist_tf(X,Y)
            return tf.multiply(self.sigma_f_sq, tf.exp(-0.5*K))

    def firstOrderKernelMatrix_dff(self, dX1: tf.Tensor, X2: tf.Tensor):     
        if dX1.ndim == 1: 
            dX1 = tf.reshape(dX1,[-1,1])

        if X2.ndim == 1: 
            X2 = tf.reshape(X2,[-1,1])

        num_dX1 = dX1.shape[0]
        num_X2 = X2.shape[0]

        Kxy_m = tf.repeat(self.zeroOrderKernelMatrix(dX1, X2, full=True), repeats=self._DIMS, axis=0)
        X = tf.tile(tf.reshape(dX1, [-1, 1]),[1,num_X2])
        Y = tf.tile(tf.transpose(X2), [num_dX1, 1])
        return -tf.reshape(self.Lambda_rec,[self._DIMS, 1])*(X-Y)*Kxy_m         
    
    # Currently only supports the case of dX1 == dX2 and one point in dX1
    def firstOrderKernelMatrix_dfdf(self, dX1: tf.Tensor, dX2: tf.Tensor):
        return self.sigma_f_sq * tf.linalg.diag(self.Lambda_rec)
    
    def secondOrderKernelMatrix_ddff(self, ddX1: tf.Tensor, X2: tf.Tensor):
        if ddX1.ndim == 1: 
            ddX1 = tf.reshape(ddX1,[-1,1])

        if X2.ndim == 1: 
            X2 = tf.reshape(X2,[-1,1])

        num_ddX1 = ddX1.shape[0]
        num_X2 = X2.shape[0]

        Kxy_m = tf.repeat(self.zeroOrderKernelMatrix(ddX1, X2, full=True), repeats=self._DIMS, axis=0)
        X = tf.tile(tf.reshape(ddX1, [-1, 1]),[1,num_X2])
        Y = tf.tile(tf.transpose(X2), [num_ddX1, 1])
        L = tf.reshape(self.Lambda_rec,[self._DIMS, 1])
        return -(tf.ones_like(X)-L*tf.cast(tf.square(X-Y),dtype=tf.double))*Kxy_m*L
    
    # Currently only supports the case of dX1 == dX2 and one point in dX1
    def secondOrderKernelMatrix_ddfdf(self, ddX1: tf.Tensor, dX2: tf.Tensor):
        return tf.zeros([self._DIMS, self._DIMS])
    
    # Currently only supports the case of dX1 == dX2 and one point in dX1
    def secondOrderKernelMatrix_ddfddf(self, ddX1: tf.Tensor, ddX2: tf.Tensor, full: bool = False):
        tiled_lambda = tf.tile(tf.reshape(self.Lambda_rec,[-1,1]), [1,self._DIMS])
        return (tf.fill([self._DIMS, self._DIMS], self.sigma_f_sq)+2*self.sigma_f_sq*tf.eye(self._DIMS, dtype=tf.double)) * tiled_lambda * tf.transpose(tiled_lambda) # return (tf.fill([self._DIMS, self._DIMS], self.sigma_f_sq)+2*self.sigma_f_sq*tf.eye(self._DIMS, dtype=tf.double)) * tiled_lambda * tf.transpose(tiled_lambda)
    
    # Overrides parent class implementation
    # TODO: Optimize to avoid inverses!!
    @validate_input_tensor
    def computePosteriorMean(self, Xtest: tf.Tensor) -> tf.Tensor:
        match self.computationCase:
            case ComputationMode.ZEROTH_ORDER: 
                return super(SEARDSecondOrderModel, self).computePosteriorMean(Xtest) # Optimization is independent of kernel and therefore implemented in parent class
            
            case ComputationMode.FIRST_ORDER:
                Kx1x1_inv = tf.linalg.diag(tf.math.reciprocal(tf.linalg.diag_part(self.firstOrderKernelMatrix_dfdf(self.firstOrderX, self.firstOrderX)
                                                                          + self.sigma_train_sq * tf.eye(self.firstOrderX.shape[0], dtype=tf.double))))
                Kxx1 = self.firstOrderKernelMatrix_fdf(Xtest, self.firstOrderX)
                Kx0x1 = self.firstOrderKernelMatrix_fdf(self.zerothOrderX, self.firstOrderX)

                B2 = self.zeroOrderKernelMatrix(Xtest, self.zerothOrderX, full=True) - (Kxx1 @ Kx1x1_inv @ tf.transpose(Kx0x1))
                B4 = self.zeroOrderKernelMatrix(self.zerothOrderX, self.zerothOrderX, full=True) + self.sigma_train_sq * tf.eye(self.zerothOrderX.shape[0], dtype=tf.double) - (Kx0x1 @ Kx1x1_inv @ tf.transpose(Kx0x1))
                L = tf.linalg.cholesky(B4)
                B2_component = tf.transpose(tf.linalg.solve(L, tf.transpose(B2)))

                return (Kxx1 - B2_component @ tf.linalg.solve(L, Kx0x1)) @ Kx1x1_inv @ self.firstOrderY + B2_component @ tf.linalg.solve(L, self.zerothOrderY)

            case ComputationMode.SECOND_ORDER:
                Kxx0 = self.zeroOrderKernelMatrix(Xtest, self.zerothOrderX, full=True)
                Kx0x0 = self.zeroOrderKernelMatrix(self.zerothOrderX, self.zerothOrderX, full=True)
                Kxx1 = self.firstOrderKernelMatrix_fdf(Xtest, self.firstOrderX)
                Kx1x1 = self.firstOrderKernelMatrix_dfdf(self.firstOrderX, self.firstOrderX)
                Kx0x1 = self.firstOrderKernelMatrix_fdf(self.zerothOrderX, self.firstOrderX)
                Kxx2 = self.secondOrderKernelMatrix_fddf(Xtest, self.secondOrderX)
                Kx0x2 = self.secondOrderKernelMatrix_fddf(self.zerothOrderX, self.secondOrderX)
                Kx2x2_inv = tf.linalg.inv(self.secondOrderKernelMatrix_ddfddf(self.secondOrderX, self.secondOrderX) + self.sigma_train_sq * tf.eye(self._DIMS, dtype=tf.double))

                Astar = Kxx2 @ Kx2x2_inv
                A0 = Kx0x2 @ Kx2x2_inv

                Y2 = Kx0x0 + self.sigma_train_sq * tf.eye(self.zerothOrderX.shape[0], dtype=tf.double) - A0 @ tf.transpose(Kx0x2)
                Y3_inv = tf.linalg.diag(tf.math.reciprocal(tf.linalg.diag_part(Kx1x1 + self.sigma_train_sq * tf.eye(self._DIMS, dtype=tf.double))))
                Y4 = Kxx0 - Astar @ tf.transpose(Kx0x2)
                Y5 = Kx0x1
                Y6 = Kxx1

                Z2_inv = tf.linalg.inv(Y2 - Y5 @ Y3_inv @ tf.transpose(Y5))
                Z3 = Y4 - Y6 @ Y3_inv @ tf.transpose(Y5)

                Y6Y3 = Y6 @ Y3_inv
                Y5Y3 = Y5 @ Y3_inv
                Z3Z2 = Z3 @ Z2_inv

                dy2 = (Astar - Z3Z2 @ A0 ) @ self.secondOrderY
                dy1 = (Y6Y3 - Z3Z2 @ Y5Y3) @ self.firstOrderY
                dy0 = Z3Z2 @ self.zerothOrderY
                return dy2 + dy1 + dy0

    #Overrides parent class implementation
    @validate_input_tensor
    def computePosteriorVariance(self, Xtest: tf.Tensor, full: bool = False):
        if full: 
            # TODO: Computation of full covariance matrix if required
            raise NotImplementedError('Currently no support for full covariance matrix')
        else: 
            match self.computationCase:
                case ComputationMode.ZEROTH_ORDER:
                    KTrainChol = tf.linalg.cholesky(self.zeroOrderKernelMatrix(self.zerothOrderX, self.zerothOrderX, full=True) + self.sigma_train_sq * tf.eye(self.zerothOrderX.shape[0], dtype=tf.double))
                    z = tf.linalg.solve(KTrainChol, self.zeroOrderKernelMatrix(self.zerothOrderX, Xtest, full=True))
                    return self.zeroOrderKernelMatrix(Xtest, Xtest, full=False) - tf.reduce_sum(tf.cast(tf.square(z), dtype=tf.double), axis=0)
                
                case ComputationMode.FIRST_ORDER:
                    Kx1x1_inv = tf.linalg.diag(tf.math.reciprocal(tf.linalg.diag_part(self.firstOrderKernelMatrix_dfdf(self.firstOrderX, self.firstOrderX)
                                                                          + self.sigma_train_sq * tf.eye(self.firstOrderX.shape[0], dtype=tf.double))))
                    Kx1x0 = self.firstOrderKernelMatrix_dff(self.firstOrderX, self.zerothOrderX)
                    Kxx1 = self.firstOrderKernelMatrix_fdf(Xtest, self.firstOrderX)

                    B2 = self.zeroOrderKernelMatrix(Xtest, self.zerothOrderX, full=True) - (Kxx1 @ Kx1x1_inv @ Kx1x0)
                    B4 = self.zeroOrderKernelMatrix(self.zerothOrderX, self.zerothOrderX, full=True) + self.sigma_train_sq * tf.eye(self.zerothOrderX.shape[0], dtype=tf.double) - (tf.transpose(Kx1x0) @ Kx1x1_inv @ Kx1x0)

                    B1_diag = self.zeroOrderKernelMatrix(Xtest, Xtest, full=False) - tf.transpose(tf.reduce_sum(tf.math.multiply(Kxx1@Kx1x1_inv, Kxx1), axis=1))
                    z = tf.linalg.solve(tf.linalg.cholesky(B4), tf.transpose(B2))

                    return B1_diag - tf.reduce_sum(tf.cast(tf.square(z), dtype=tf.double), axis=0)
                
                case ComputationMode.SECOND_ORDER:
                    Kxx = self.zeroOrderKernelMatrix(Xtest, Xtest, full=False)
                    Kxx0 = self.zeroOrderKernelMatrix(Xtest, self.zerothOrderX, full=True)
                    Kx0x0 = self.zeroOrderKernelMatrix(self.zerothOrderX, self.zerothOrderX, full=True)
                    Kxx1 = self.firstOrderKernelMatrix_fdf(Xtest, self.firstOrderX)
                    Kx1x1 = self.firstOrderKernelMatrix_dfdf(self.firstOrderX, self.firstOrderX)
                    Kx0x1 = self.firstOrderKernelMatrix_fdf(self.zerothOrderX, self.firstOrderX)
                    Kxx2 = self.secondOrderKernelMatrix_fddf(Xtest, self.secondOrderX)
                    Kx0x2 = self.secondOrderKernelMatrix_fddf(self.zerothOrderX, self.secondOrderX)
                    L_Kx2x2 = tf.linalg.cholesky(self.secondOrderKernelMatrix_ddfddf(self.secondOrderX, self.secondOrderX) + self.sigma_train_sq * tf.eye(self._DIMS, dtype=tf.double))

                    Y1 = Kxx - tf.reduce_sum(tf.cast(tf.square(tf.linalg.solve(L_Kx2x2, tf.transpose(Kxx2))),dtype=tf.double), axis=0) 
                    temp = tf.linalg.solve(L_Kx2x2, tf.transpose(Kx0x2))
                    Y2 = Kx0x0 + self.sigma_train_sq * tf.eye(self.zerothOrderX.shape[0], dtype=tf.double) - tf.transpose(temp) @ temp
                    Y3_inv = tf.linalg.diag(tf.math.reciprocal(tf.linalg.diag_part(Kx1x1 + self.sigma_train_sq * tf.eye(self._DIMS, dtype=tf.double))))
                    Y4 = Kxx0 - tf.transpose(tf.linalg.solve(L_Kx2x2, tf.transpose(Kxx2))) @ tf.linalg.solve(L_Kx2x2, tf.transpose(Kx0x2))

                    Y6Y3 = Kxx1 @ Y3_inv
                    Y5Y3 = Kx0x1 @ Y3_inv
                    
                    Z1 = Y1 - tf.reduce_sum(tf.math.multiply(Kxx1@Y3_inv, Kxx1), axis=1)
                    L_Z2 = tf.linalg.cholesky(Y2 - Y5Y3 @ tf.transpose(Kx0x1))
                    Z3 = Y4 - Y6Y3 @ tf.transpose(Kx0x1)

                    return Z1 - tf.reduce_sum(tf.cast(tf.square(tf.linalg.solve(L_Z2, tf.transpose(Z3))),dtype=tf.double), axis=0)