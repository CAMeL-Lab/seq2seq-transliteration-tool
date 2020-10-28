"""This module defines the parent-most class for all the models in the project.
   The `BaseModel` class should only contain the most general functionality and
   abstractions that are desirable for all the other models; no matter what
   their functionality or computational graphs are."""

from abc import abstractmethod
import os

import tensorflow as tf
tf.set_random_seed(1)

from ai.utils import abstractclass


@abstractclass
class BaseModel(object):
  """Model abstraction to handle restoring variables and summary writers."""
  
  def __init__(self, restore=False, model_output_dir='default'):
    """The constructor builds the model's computational graph. Thus, it should
       always be called within a graph scope.
       Keyword arguments:
       `restore`: whether to override or use the model's pre-trained variables,
       `model_output_dir`: string for the model's checkpoint and summary outputs."""
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.build_graph()
    self.summary_op = tf.summary.merge_all()
    # Default configurations
    self.restore = restore
    self.model_output_dir = model_output_dir
    self.saver = None
    self.train_writer = None
    self.valid_writer = None
  
  @abstractmethod
  def build_graph(self):
    """Overridable. All placeholders and operations that define the
       computational graph should be defined in this method."""
    pass
  
  def start(self):
    """Initialize or restore the model, and create the summary writer. This
       method expects to be called within a session whose graph is that where
       the `build_graph` method was previously called."""
    sess = tf.get_default_session()
    train_dir = os.path.join(self.model_output_dir, 'train')
    valid_dir = os.path.join(self.model_output_dir, 'valid')
    # Check whether to restore the model
    self.saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(self.model_output_dir)
    if self.restore and ckpt:
      self.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      sess.run(tf.global_variables_initializer())
      # Flush the provided summary writer to avoid overwriting
      if tf.gfile.Exists(self.model_output_dir):
        tf.gfile.DeleteRecursively(self.model_output_dir)
      tf.gfile.MakeDirs(self.model_output_dir)
      tf.gfile.MakeDirs(train_dir)
      tf.gfile.MakeDirs(valid_dir)
    # Create the summary writers
    self.train_writer = tf.summary.FileWriter(train_dir, graph=sess.graph)
    self.valid_writer = tf.summary.FileWriter(valid_dir, graph=sess.graph)
  
  def save(self, filename='checkpoint.ckpt'):
    """Save the model's variables for later training or evaluation."""
    sess = tf.get_default_session()
    self.saver.save(
      sess, os.path.join(self.model_output_dir, filename),
      global_step=sess.run(self.global_step))
