"""This module takes care of all data parsing"""

import io
import os
import re

import numpy as np
np.random.seed(1)

from ai.datasets import BaseDataset

def max_length_seq(pairs):
  """Get the maximum sequence length of the examples in the provided pairs."""
  return [max(map(len, seq)) for seq in zip(*pairs)]


class ALLDATASET(BaseDataset):
  """ALLDATASET parsing."""
  
  def __init__(self, train_input=None, train_output=None, dev_input=None, 
              dev_output=None, predict_input_file=None, max_input_length=None, max_label_length=None,
              parse_repeated=0, **kw):
    """Arguments:
       `max_input_length`: maximum sequence length for the inputs,
       `max_label_length`: maximum sequence length for the labels,
       `parse_repeated`: e.g. convert `abababab` to `<ab>4`
       Note on usage: to account for the _GO and _EOS tokens that the labels
       have inserted, if the maximum length sequences are in the labels, use
       two extra time steps if the goal is to not truncate anything."""
    super().__init__(**kw)
    self.train_input = train_input
    self.train_output = train_output
    self.dev_input = dev_input
    self.dev_output = dev_output
    self.predict_input_file = predict_input_file
    self.max_input_length = max_input_length
    self.max_label_length = max_label_length
    self.parse_repeated = parse_repeated
    # Prepare data
    print("Training input path is:", self.train_input)
    train_labels = self.maybe_flatten_gold(self.train_output)
    with io.open(self.train_input, encoding='utf-8') as train_file:
      self.train_pairs = self.make_pairs(train_file.readlines(), train_labels)
    # Lock the addition of new characters into the data-- this way, we simulate
    # a real testing environment with possible _UNK tokens.
    self.max_types = self.num_types()
    # Prepare validation data
    valid_labels = self.maybe_flatten_gold(self.dev_output)
    print("Dev input path is:", self.dev_input)
    with io.open(self.dev_input, encoding='utf-8') as valid_file:
      self.valid_pairs = self.make_pairs(valid_file.readlines(), valid_labels)
  
  def untokenize(self, tokens, join_str=''):
    result = super().untokenize(tokens, join_str=join_str)
    if not self.parse_repeated:
      return result
    repl = lambda m: m.group(1)[1:-1] * int(m.group(2))
    return re.sub(r'(<[^>]+>)([0-9]+)', repl, result)
  
  def maybe_flatten_gold(self, file_root, force=False):
    """Create and return the contents a provided filename that generates a
       parallel corpus to the inputs, following the corrections provided in the
       default gold file m2 format. Note that this step is necessary for
       seq2seq training, and code cannot be borrowed from the evaluation script
       because it never flattens the system output; instead, it finds the
       minimum number of corrections that map the input into the output."""
    print("Gold path is", file_root)
    if not force and os.path.exists(file_root):
      with io.open(file_root, encoding='utf-8') as gold_file:
        return gold_file.readlines()
  
  def pad_batch(self, batch):
    """Pad the given batch with zeros."""
    max_input_length = self.max_input_length
    max_label_length = self.max_label_length
    if max_input_length is None or max_label_length is None:
      max_input_length, max_label_length = max_length_seq(batch)
    for i in range(len(batch)):
      while len(batch[i][0]) < max_input_length:
        batch[i][0].append(self.type_to_ix['_PAD'])
      while len(batch[i][1]) < max_label_length:
        batch[i][1].append(self.type_to_ix['_PAD'])
    return batch
  
  # Override this to pad the batches
  def get_train_batches(self, batch_size):
    res = np.array(
      list(map(self.pad_batch, super().get_train_batches(batch_size))))
    #print(np.array(res).shape)
    return res

  def get_valid_batch(self, batch_size):
    """Draw random examples and pad them to the largest sequence drawn."""
    batch = []
    while len(batch) < batch_size:
      sequence = self.valid_pairs[np.random.randint(len(self.valid_pairs))]
      # Optionally discard examples past a maximum input or label length
      input_ok = self.max_input_length is None \
                 or len(sequence[0]) <= self.max_input_length
      label_ok = self.max_label_length is None \
                 or len(sequence[1]) <= self.max_label_length
      if input_ok and label_ok:
        batch.append(sequence)
    return zip(*self.pad_batch(batch))
  
  def shorten_repetitions(self, line):
    """If a pattern is seen at least 2 times contiguously, replace it with
       "pat...pat" (n times) -> "<pat>n"."""
    if not self.parse_repeated or self.parse_repeated < 2:
      return line
    repl = lambda m:'<{}>{}'.format(m.group(1), len(m.group()) // len(m.group(1)))
    return re.sub('(.+?)\1{%d,}' % (self.parse_repeated - 1), repl, line)
  
  def make_pairs(self, input_lines, label_lines):
    pairs = []
    for i in range(len(input_lines)):
      input_line = self.shorten_repetitions(input_lines[i][:-1])  # no newline
      label_line = self.shorten_repetitions(label_lines[i][:-1])  # no newline
      if len(input_line) <= self.max_input_length and \
         len(label_line) <= self.max_label_length - 1:  # eos token
        _input = self.tokenize(input_line)
        label = self.tokenize(label_line)
        label.append(self.type_to_ix['_EOS'])
        pairs.append((_input, label))
    return pairs
