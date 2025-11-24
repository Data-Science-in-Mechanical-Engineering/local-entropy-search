import functools
import tensorflow as tf

# TODO: Add further checking
def validate_input_tensor(func):
    @functools.wraps(func)
    def wrapper(self, tensor1, *args, **kwargs):
        if not (tf.is_tensor(tensor1)):
            raise ValueError("The first argument must be valid TensorFlow tensors.")
        
        if tensor1.ndim != 1:
            if tensor1.shape[1] != self._DIMS:
                raise ValueError(f'Input tensor has incorrect shape. Expected {self._DIMS} columns based on model dimensionality, received {tensor1.shape[1]}.')
        else:
            if self._DIMS != 1:
                raise ValueError(f'Input has incorrect shape. Expected {self._DIMS} columns based on model dimensionality, but received 1.')
        return func(self, tensor1, *args, **kwargs)
    return wrapper