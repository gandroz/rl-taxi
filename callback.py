import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class LogTensorBoard(tf.keras.callbacks.TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._train_step = 0
        self._train_dir = self.log_dir
        self._writers = {"train": tf.summary.create_file_writer(self._train_dir)}  # Resets writers.
        self._should_write_train_graph = False

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model
        
    def on_step_begin(self):
        pass
    
    def on_step_end(self, step, logs):
        self._log_epoch_metrics(step, logs)
        
    def _collect_learning_rate(self, logs):
        return logs
    

@keras_export("keras.optimizers.schedules.ExponentialDecay")
class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      decay_rate,
      lr_min,        
      name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.lr_min = lr_min
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "ExponentialDecay") as name:
            initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
              self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.decay_steps, dtype)
            decay_rate = math_ops.cast(self.decay_rate, dtype)
            lr_min = math_ops.cast(self.lr_min, dtype)
            global_step_recomp = math_ops.cast(step, dtype)

            p = global_step_recomp / decay_steps
            return lr_min + math_ops.multiply(
              (initial_learning_rate - lr_min), math_ops.pow(decay_rate, p), name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "lr_min": self.lr_min,
            "name": self.name
        }



# class LogTensorBoard(keras.callbacks.TensorBoard):

#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._train_step = 1
#         self._train_dir = os.path.join(self.log_dir, f'train_{int(time.time())}')
#         self._writers = {"train": tf.summary.create_file_writer(self._train_dir)}  # Resets writers.
#         self._should_write_train_graph = False

#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass

#     # Overrided, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self._log_epoch_metrics(self._train_step, logs)

#     # Overrided
#     # We train for one batch only, no need to save anything at epoch end
#     def on_batch_end(self, batch, logs=None):
#         pass

#     # Overrided, so won't close writer
#     def on_train_end(self, _):
#         pass
    
#     def _collect_learning_rate(self, logs):
#         return logs
    
#     def step_forward(self):
#         self._train_step += 1