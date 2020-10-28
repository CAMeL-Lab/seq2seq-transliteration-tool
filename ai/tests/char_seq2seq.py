import io
import os
import sys
import re
import timeit
import random
import argparse

import tensorflow as tf
import editdistance
import numpy as np
from scipy import exp
from scipy.special import lambertw

from ai.datasets import ALLDATASET
from ai.models import CharSeq2Seq

# INPUT FILES
tf.app.flags.DEFINE_string('train_input', None, "Path to train input file")
tf.app.flags.DEFINE_string('train_output', None, "Path to train output file")
tf.app.flags.DEFINE_string('dev_input', None, "Path to dev input file")
tf.app.flags.DEFINE_string('dev_output', None, "Path to dev output file")
tf.app.flags.DEFINE_string('test_input', None, "Path to test input file")

# HYPERPARAMETERS
tf.app.flags.DEFINE_float('lr', 0.0001, "Initial learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 2048, "Batch size.")
tf.app.flags.DEFINE_integer('embedding_size', 256, "Embedding dimensionality.")
tf.app.flags.DEFINE_integer('hidden_size', 256, "Number of hidden units.")
tf.app.flags.DEFINE_integer('rnn_layers', 2, "Number of RNN layers.")
tf.app.flags.DEFINE_boolean('bidirectional_encoder', True, "Whether to use a"
                            " bidirectional RNN in the encoder's 1st layer.")
tf.app.flags.DEFINE_string('bidirectional_mode', 'add', "Set to 'add',"
                           " 'concat' or 'project'.")
tf.app.flags.DEFINE_boolean('use_lstm', False, "Set to False to use GRUs.")
tf.app.flags.DEFINE_string('attention', 'luong', "'bahdanau' or 'luong'"
                           " (default is 'luong').")
tf.app.flags.DEFINE_float('dropout', .9, "Keep probability for dropout on the"
                          "RNNs' non-recurrent connections.")
tf.app.flags.DEFINE_float('max_grad_norm', 11.5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer('beam_size', 5, "Beam search size.")
tf.app.flags.DEFINE_float('initial_p_sample', .35, "Initial decoder sampling"
                          " probability (0=ground truth, 1=use predictions).")
tf.app.flags.DEFINE_float('final_p_sample', .35, "Final decoder sampling"
                          " probability (0=ground truth, 1=use predictions).")
tf.app.flags.DEFINE_integer('epochs_p_sample', 20, "Duration in epochs of"
                            " schedule sampling (determines rate of change).")
tf.app.flags.DEFINE_boolean('linear_p_sample', True, "False = sigmoid decay.")
tf.app.flags.DEFINE_integer('parse_repeated', 0, "Set to > 1 to compress"
                            " contiguous patterns in the data pipeline.")
tf.app.flags.DEFINE_float('epsilon', 1e-8, "Denominator constant.")
tf.app.flags.DEFINE_float('beta1', .9, "First order moment decay.")
tf.app.flags.DEFINE_float('beta2', .999, "Second order moment decay.")
tf.app.flags.DEFINE_string('fasttext_executable', None, "Path to Fasttext executable file")
tf.app.flags.DEFINE_string('word_embeddings', None, "Path to Fasttext .bin file")
tf.app.flags.DEFINE_boolean('train_word_embeddings', False, "Backprop on/off.")

# CONFIG
tf.app.flags.DEFINE_integer('max_sentence_length', 110, "Max. word length of"
                            " training examples (both inputs and labels).")
tf.app.flags.DEFINE_integer('num_steps_per_eval', 50, "Number of steps to wait"
                            " before running the graph with the dev set.")
tf.app.flags.DEFINE_integer('max_epochs', 40, "Number of epochs to run"
                            " (0 = no limit).")
tf.app.flags.DEFINE_string('predict_input_file', None, "Path to prediction input file")
tf.app.flags.DEFINE_string('predict_output_file', None, "Name of the output file with decoding results.")
tf.app.flags.DEFINE_boolean('restore', True, "Whether to restore the model.")
tf.app.flags.DEFINE_string('model_output_dir', None, "Name of the output directory.")

FLAGS = tf.app.flags.FLAGS

# Catch these very common errors :)
if not FLAGS.model_output_dir:
  raise ValueError(
    "Undefined model output directory. Please set the --model_output_dir flag.")
if not FLAGS.word_embeddings:
  raise ValueError(
    "Undefined FastText model. Please set the --word_embeddings flag.")


# Read the training and dev data files.
print("Building dynamic character-level ALLDATASET data...", flush=True)
DATASET = ALLDATASET(
  train_input=FLAGS.train_input, train_output=FLAGS.train_output,
  dev_input=FLAGS.dev_input, dev_output=FLAGS.dev_output, 
  predict_input_file=FLAGS.predict_input_file,
  parse_repeated=FLAGS.parse_repeated,
  shuffle=FLAGS.predict_input_file is None, max_input_length=FLAGS.max_sentence_length,
  max_label_length=FLAGS.max_sentence_length)


# Get all unique word embeddings from the given FastText model.
# The following files should remain the same in training and prediction to avoid an error due to unequal number of 
# word embedding vector lines.
cat_files = []
if FLAGS.train_input: cat_files.append(FLAGS.train_input)
if FLAGS.dev_input: cat_files.append(FLAGS.dev_input)
if FLAGS.test_input: cat_files.append(FLAGS.test_input)

cat_files = " ".join(cat_files)
unix_comm = (r"cat %s| grep -Po '(?<=^|\s)[^\s]*(?=\s|$)' | awk "
             r"'!seen[$0]++' | %s print-word-vectors "
             r"%s") % (cat_files, FLAGS.fasttext_executable, FLAGS.word_embeddings)

WORD_EMBEDDINGS = []
WORD_TO_IX = {}

if FLAGS.word_embeddings == 'concat':
  narrow_lines = os.popen(unix_comm.format('narrow')).read().splitlines()
  wide_lines = os.popen(unix_comm.format('wide')).read().splitlines()
  # The words should be the exact same for both lists, in the exact same order.
  for i in range(len(narrow_lines)):
    narrow_line = narrow_lines[i].split()
    word = tuple(DATASET.tokenize(narrow_line[0]))
    WORD_TO_IX[word] = i
    embedding = list(map(float, narrow_line[1:] + wide_lines[i].split()[1:]))
    WORD_EMBEDDINGS.append(embedding)
else:
  vec_lines = os.popen(unix_comm.format(FLAGS.word_embeddings)).read()
  for i, line in enumerate(vec_lines.splitlines()):
    line = line.split()
    word = tuple(DATASET.tokenize(line[0]))
    WORD_TO_IX[word] = i
    WORD_EMBEDDINGS.append(list(map(float, line[1:])))

# Space embedding is set randomly with standard normal initialization.
WORD_TO_IX[DATASET.type_to_ix[(' ',)]] = len(WORD_EMBEDDINGS)
WORD_EMBEDDINGS.append([
  random.normalvariate(0, 1) for _ in range(len(WORD_EMBEDDINGS[0]))])

def add_word_ids(batch):
  """Turn each character id to a pair (id, word_id)."""
  space_chid = DATASET.type_to_ix[(' ',)]
  space_like = [
    space_chid,
    DATASET.type_to_ix['_PAD'], DATASET.type_to_ix['_EOS'], DATASET.type_to_ix['<bos>'],
    DATASET.type_to_ix['<eos>'], DATASET.type_to_ix['<bow>'], DATASET.type_to_ix['<eow>']]
  new_batch = []
  for seq in batch:
    new_seq = []
    # This accumulates ids until we are ready to form the word
    char_ids = []
    for i, id_ in enumerate(seq):
      if id_ in space_like or i == len(seq) - 1:
        # Add the accumulated pairs
        if len(char_ids):
          try:
            word_id = WORD_TO_IX[tuple(char_ids)]
          except KeyError:
            # print("WARNING: Unknown word at index", i)
            # print(DATASET.untokenize(char_ids))
            word_id = WORD_TO_IX[space_chid]
          for char_id in char_ids:
            new_seq.append([char_id, word_id])
        # Add the space id and empty the char ids.
        new_seq.append([space_chid, WORD_TO_IX[space_chid]])
        char_ids = []
      else:
        char_ids.append(id_)
    new_batch.append(new_seq)
  return np.array(new_batch)


def untokenize_batch(id_batch):
  """Return the UTF-8 sequences of the given batch of ids."""
  return [DATASET.untokenize(DATASET.clean(s)) for s in id_batch]


def levenshtein(proposed, gold, normalize=False):
  """Return the normalized Levenshtein distance of the given strings."""
  lev_densities = []
  for x, y in zip(proposed, gold):
    score = editdistance.eval(x, y)
    if normalize:
      score /= len(y)
    lev_densities.append(score)
  return sum(lev_densities) / len(lev_densities)


def train():
  """Run a loop that continuously trains the model."""
  print("Building computational graph...", flush=True)
  graph = tf.Graph()
  
  with graph.as_default():
    
    tf.set_random_seed(1)
    random.seed(1)
    np.random.seed(1)
    # During training we use beam width 1. There are lots of complications on
    # the implementation, e.g. only tiling during inference.
    m = CharSeq2Seq(
      num_types=DATASET.num_types(),
      max_encoder_length=FLAGS.max_sentence_length,
      max_decoder_length=FLAGS.max_sentence_length,
      pad_id=DATASET.type_to_ix['_PAD'],
      eos_id=DATASET.type_to_ix['_EOS'],
      go_id=DATASET.type_to_ix['_GO'],
      batch_size=FLAGS.batch_size, embedding_size=FLAGS.embedding_size,
      hidden_size=FLAGS.hidden_size, rnn_layers=FLAGS.rnn_layers,
      bidirectional_encoder=FLAGS.bidirectional_encoder,
      bidirectional_mode=FLAGS.bidirectional_mode,
      use_lstm=FLAGS.use_lstm, attention=FLAGS.attention, 
      dropout=FLAGS.dropout, max_grad_norm=FLAGS.max_grad_norm, beam_size=1,
      epsilon=FLAGS.epsilon, beta1=FLAGS.beta1, beta2=FLAGS.beta2,
      word_embeddings=WORD_EMBEDDINGS,
      train_word_embeddings=FLAGS.train_word_embeddings, restore=FLAGS.restore,
      model_output_dir=FLAGS.model_output_dir)
  
  # Allow TensorFlow to resort back to CPU when we try to set an operation to
  # a GPU where there's only a CPU implementation, rather than crashing.
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  
  with tf.Session(graph=graph, config=sess_config) as sess:
    print("Initializing or restoring model...", flush=True)
    m.start()
    
    # If the model was not restored, initialize the variable hyperparameters.
    if sess.run(m.lr) == 0:
      sess.run(tf.assign(m.lr, FLAGS.lr))
    if sess.run(m.p_sample) == 0:
      sess.run(tf.assign(m.p_sample, FLAGS.initial_p_sample))
    
    # Get the number of epochs that have passed (easier by getting batches now)
    step = m.global_step.eval()
    batches = DATASET.get_train_batches(m.batch_size)
    epoch = step // len(batches)
    
    # Scheduled sampling decay
    i = FLAGS.initial_p_sample
    f = FLAGS.final_p_sample
    # The stopping point is based on the max epochs
    total_train_steps = len(batches) * FLAGS.epochs_p_sample
    if i != f and not FLAGS.linear_p_sample:
      k = total_train_steps / (float(lambertw(total_train_steps / 2)) * 2)
      expk = float(exp(-total_train_steps / k))
      delta_f = (f - i) * (1 + k) * (1 + k * expk) / (k - k * expk) - f
      delta_i = (f + delta_f) / (1 + k)
    
    while not FLAGS.max_epochs or epoch <= FLAGS.max_epochs:
      print("=====EPOCH {}=====".format(epoch), flush=True)
      while step < (epoch + 1) * len(batches):
        step = m.global_step.eval()
        
        # Scheduled sampling decay
        if i != f:
          # Linear decay
          if FLAGS.linear_p_sample:
            p = min(f, i + step * (f - i) / total_train_steps)
          # Inverse sigmoid decay
          else:
            expk = float(np.exp(-step / k))
            p = min(f, i - delta_i + (f + delta_f) / (1 + k * expk))
          
          sess.run(tf.assign(m.p_sample, p))
        
        # Gradient descent and backprop
        train_inputs = add_word_ids(batches[step % len(batches), :, 0])
        train_labels = batches[step % len(batches), :, 1]
        train_fd = {m.inputs: train_inputs, m.labels: train_labels}
        
        # Wrap into function to measure running time
        def train_step():
          try:
            sess.run(m.train_step, feed_dict=train_fd)
          except:
            print(batches[step % len(batches), :, 0].shape)
            print(train_inputs.shape)
            for seq in train_inputs: 
              if len(seq) != FLAGS.max_sentence_length:
                print("***BAD SEQ OF LENGTH", len(seq), "FOUND***")
                print(seq)
                print(DATASET.untokenize(batches[step % len(batches), :, 0]))
            raise

        print("Global step {0} ({1}s)".format(
          step, timeit.timeit(train_step, number=1)), flush=True)
        
        if step % FLAGS.num_steps_per_eval == 0:
          valid_inputs, valid_labels = DATASET.get_valid_batch(m.batch_size)
          valid_fd = {
            m.inputs: add_word_ids(valid_inputs),
            m.labels: valid_labels
          }
          
          # Run training and validation perplexity and samples
          
          lr, train_ppx, train_output, p_sample, train_ppx_summ = sess.run([
            m.lr,
            m.perplexity,
            m.output,
            m.p_sample,
            m.perplexity_summary,
          ], feed_dict=train_fd)
          
          valid_ppx, valid_output, infer_output, valid_ppx_summ = sess.run([
            m.perplexity,
            m.output,
            m.generative_output,
            m.perplexity_summary,
          ], feed_dict=valid_fd)
          
          # Convert data to UTF-8 strings for evaluation and display
          valid_inputs = untokenize_batch(valid_inputs)
          valid_labels = untokenize_batch(valid_labels)
          valid_output = untokenize_batch(valid_output)
          infer_output = untokenize_batch(infer_output)
          
          # Run evaluation metrics
          lev = levenshtein(infer_output, valid_labels)
          lev_density = levenshtein(infer_output, valid_labels, normalize=True)
          
          lev_summ = sess.run(
            m.lev_summary, feed_dict={m.lev: lev})
          lev_density_summ = sess.run(
            m.lev_density_summary, feed_dict={m.lev_density: lev_density})
          
          # Write summaries to TensorBoard
          m.train_writer.add_summary(train_ppx_summ, global_step=step)
          m.valid_writer.add_summary(valid_ppx_summ, global_step=step)
          m.valid_writer.add_summary(lev_summ, global_step=step)
          m.valid_writer.add_summary(lev_density_summ, global_step=step)
          
          # Display results to stdout
          print("  lr:", lr)
          print("  p_sample:", p_sample)
          print("  train_ppx:", train_ppx)
          print("  valid_ppx:", valid_ppx)
          print("  lev:", lev)
          print("  lev_density:", lev_density)
          print("Input:")
          print(valid_inputs[0])
          print("Target:")
          print(valid_labels[0])
          print("Output with ground truth:")
          print(valid_output[0])
          print("Greedily decoded output:")
          print(infer_output[0], flush=True)
      
      # Epoch about to be done - save, reshuffle the data and get new batches
      print("Saving model...")
      m.save()
      print("Model saved. Resuming training...", flush=True)
      batches = DATASET.get_train_batches(m.batch_size)
      epoch += 1


def decode():
  """Run a blind test on the file with path given by the `decode` flag."""
  with open(FLAGS.predict_input_file) as test_file:
    lines = test_file.readlines()
    # Get the largest sentence length to set an upper bound to the decoder.
    max_length = FLAGS.max_sentence_length
    # max_length = max([len(line) for line in lines])
  
  print("Building computational graph...", flush=True)
  graph = tf.Graph()
  with graph.as_default():
    
    tf.set_random_seed(1)
    random.seed(1)
    np.random.seed(1)

    m = CharSeq2Seq(
      num_types=DATASET.num_types(),
      max_encoder_length=max_length, max_decoder_length=max_length,
      pad_id=DATASET.type_to_ix['_PAD'],
      eos_id=DATASET.type_to_ix['_EOS'],
      go_id=DATASET.type_to_ix['_GO'],
      batch_size=1, embedding_size=FLAGS.embedding_size,
      hidden_size=FLAGS.hidden_size, rnn_layers=FLAGS.rnn_layers,
      bidirectional_encoder=FLAGS.bidirectional_encoder,
      bidirectional_mode=FLAGS.bidirectional_mode,
      use_lstm=FLAGS.use_lstm, attention=FLAGS.attention,
      beam_size=FLAGS.beam_size, word_embeddings=WORD_EMBEDDINGS,
      restore=True, model_output_dir=FLAGS.model_output_dir)
  
  with tf.Session(graph=graph) as sess:
    print("Restoring model...", flush=True)
    m.start()
    print(
      "Restored model (global step {})".format(m.global_step.eval()),
      flush=True)
    with io.open(FLAGS.predict_output_file, 'w', encoding='utf-8') as output_file:
      line_counter = 1
      for i, line in enumerate(lines):
        if line_counter % 500 == 0:
          print(f"Predicted {line_counter} out of {len(lines)} lines", flush=True)

        if line.endswith('\n'):
         line = line[:-1]

        number_of_chars = len(line)
        completely_divisble = number_of_chars % FLAGS.max_sentence_length == 0

        if number_of_chars < FLAGS.max_sentence_length:
          parts = [line]
        else:
          parts = []
          count = 0
          last_word_end_index = 0

          line_copy = line
          while len(line_copy) != 0 and count < len(line_copy):
            if count == FLAGS.max_sentence_length:
              if last_word_end_index == 0:
                parts.append(line_copy[: count])
                line_copy = line_copy[count:]
              else:
                parts.append(line_copy[: last_word_end_index])
                line_copy = line_copy[last_word_end_index:]
                  
                last_word_end_index = 0
                count = 0

            if line_copy[count] == " ":
              last_word_end_index = count

            count += 1

          if not completely_divisble:
            parts.append(line_copy)
        
        result = ""
        for part in parts:
          ids = DATASET.tokenize(part)
          while len(ids) < max_length:
            ids.append(DATASET.type_to_ix['_PAD'])
          ids = add_word_ids([ids])
          outputs = sess.run(m.generative_output, feed_dict={m.inputs: ids})
          top_line = untokenize_batch(outputs)[0]
          # Sequences of text will only be repeated up to 5 times.
          top_line = re.sub(r'(.+?)\1{5,}', lambda m: m.group(1) * 5, top_line)
          if top_line.strip() == "":
            result += "#"
          else:
            result += top_line
        output_file.write(result + '\n')
        line_counter += 1

if FLAGS.predict_input_file:
  decode()
else:
  train()

