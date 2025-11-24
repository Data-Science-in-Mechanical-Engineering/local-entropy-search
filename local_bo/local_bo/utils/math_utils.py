import tensorflow as tf

# Source: https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
@tf.function
def cdist_tf(A, B):
  """
  Computes pairwise distance matrix for two input matrices A and B. Tensorflow alternative to scipy.spatial.distance cdist function
  """
  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

# Source: https://stackoverflow.com/questions/38200982/how-to-compute-all-second-derivatives-only-the-diagonal-of-the-hessian-matrix
def calc_derivatives(f, x):
    """
    Calculates the diagonal entries of the Hessian of the function f
    (which maps rank-1 tensors to scalars) at coordinates x (rank-1
    tensors).
    
    Let k be the number of points in x, and n be the dimensionality of
    each point. For each point k, the function returns

      (d^2f/dx_1^2, d^2f/dx_2^2, ..., d^2f/dx_n^2) .

    Inputs:
      f (function): Takes a shape-(k,n) tensor and outputs a
          shape-(k,) tensor.
      x (tf.Tensor): The points at which to evaluate the Laplacian
          of f. Shape = (k,n).
    
    Outputs:
      A tensor containing the gradient of f at locations x. Shape: (k,n)
      
      A tensor containing the diagonal entries of the Hessian of f at
      points x. Shape = (k,n).
    """
    # Use the unstacking and re-stacking trick, which comes
    # from https://github.com/xuzhiqin1990/laplacian/
    with tf.GradientTape(persistent=True) as g1:
        # Turn x into a list of n tensors of shape (k,)
        x_unstacked = tf.unstack(x, axis=1)
        g1.watch(x_unstacked)

        with tf.GradientTape() as g2:
            # Re-stack x before passing it into f
            x_stacked = tf.stack(x_unstacked, axis=1) # shape = (k,n)
            g2.watch(x_stacked)
            f_x = f(x_stacked) # shape = (k,)
        
        # Calculate gradient of f with respect to x
        df_dx = g2.gradient(f_x, x_stacked) # shape = (k,n)
        # Turn df/dx into a list of n tensors of shape (k,)
        df_dx_unstacked = tf.unstack(df_dx, axis=1)

    # Calculate 2nd derivatives
    d2f_dx2 = []
    for df_dxi,xi in zip(df_dx_unstacked, x_unstacked):
        # Take 2nd derivative of each dimension separately:
        #   d/dx_i (df/dx_i)
        d2f_dx2.append(g1.gradient(df_dxi, xi))
    
    # Stack 2nd derivates
    d2f_dx2_stacked = tf.stack(d2f_dx2, axis=1) # shape = (k,n)
    del g1
    del g2
    return df_dx, d2f_dx2_stacked