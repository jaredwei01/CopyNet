# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from seq2seq import decoder
from seq2seq import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

import tensorflow as tf

__all__ = [
    "BasicDecoderOutput",
    "BasicDecoder",
    "CopyNetDecoder",
]


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
  pass


class BasicDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, initial_state, output_layer=None):
    """Initialize BasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        sample_id=tensor_shape.TensorShape([]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        dtypes.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)


class CopyNetDecoder(BasicDecoder):
    """
    copynet decoder, refer to the paper Jiatao Gu, 2016, 
    'Incorporating Copying Mechanism in Sequence-to-Sequence Learninag'
    https://arxiv.org/abs/1603.06393
    """
    def __init__(self, config, cell, helper, initial_state, 
                                    encoder_outputs, output_layer):
        """Initialize CopyNetDecoder.
        """
        if output_layer is None:
            raise ValueError("output_layer should not be None")
        assert isinstance(helper, helper_py.CopyNetTrainingHelper)
        self.encoder_outputs = encoder_outputs
        encoder_hidden_size = self.encoder_outputs.shape[-1].value
        self.copy_weight = tf.get_variable('copy_weight', 
                                [encoder_hidden_size, cell.output_size])
        super(CopyNetDecoder, self).__init__(cell, helper, initial_state, 
                                    output_layer=output_layer)

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size() + 
                        tf.convert_to_tensor(self.config.encoder_max_seq_len),
            sample_id=tensor_shape.TensorShape([]))
    
    def shape(self, tensor):
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])

    def _mix(self, generate_scores, copy_scores):
        # TODO is this correct? should verify the following code.
        """
        B is batch_size, V is vocab_size, L is length of every input_id
        print genreate_scores.shape     --> (B, V)
        print copy_scores.shape         --> (B, L)
        print self._helper.inputs_ids   --> (B, L)
        """
        # mask is (B, L, V)
        mask = tf.one_hot(self._helper.encoder_inputs_ids, self.config.vocab_size)

        # choice one, move generate_scores to copy_scores
        expanded_generate_scores = tf.expand_dim(generate_scores, 1) # (B,1,V)
        actual_copy_scores = copy_scores + tf.reduce_sum(
                                mask * expanded_generate_scores, 2)
        actual_generate_scores = generate_scores - tf.reduce_sum(
                                mask * expanded_generate_scores, 1)

        # choice two, move copy_scores to generate_scores
        '''
        expanded_copy_scores = tf.expand_dims(copy_scores, 2)
        acutual_generate_scores = generate_scores + tf.reduce_sum(
                                    mask * expanded_copy_scores, 1)
        acutual_copy_scores = copy_scores - tf.reduce_sum(
                                    mask * expanded_copy_scores, 2)
        '''

        mix_scores = tf.concat([actual_generate_scores, actual_copy_scores], 1)
        mix_scores = tf.nn.softmax(mix_scores, -1) # mix_scores is (B, V+L)

        # make sure mix_socres.shape is (B, V + encoder_max_seq_len)
        padding_size = self.config.encoder_max_seq_len - self.shape(copy_scores)[1]
        mix_scores = tf.pad(mix_scores, [[0, 0], [0, padding_size]])
                
        return mix_scores
    
    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.

        Args:
        time: scalar `int32` tensor.
        inputs: A (structure of) input tensors.
        state: A (structure of) state tensors and TensorArrays.
        name: Name scope for any created operations.

        Returns:
        `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            generate_scores = self._output_layer(cell_outputs)

            expand_cell_outputs = tf.expand_dims(cell_outputs, 1)
            copy_scores = tf.tensordot(self.encoder_outputs, self.copy_weight, 1)
            copy_scores = tf.nn.tanh(copy_scores)
            copy_scores = tf.reduce_sum(copy_scores * expand_cell_outputs, 2)

            mix_scores = self._mix(generate_scores, copy_scores)

            sample_ids = self._helper.sample(
                time=time, outputs=mix_scores, state=cell_state)
            # sample_ids are not always valid.. TODO
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=mix_scores,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = BasicDecoderOutput(mix_scores, sample_ids)
        return (outputs, next_state, next_inputs, finished)
