import random
random.seed(1)

from ai.utils import abstractclass


@abstractclass
class BaseDataset(object):
  """Abstract class for parsing text datasets. Includes defaults for special
     types, tokenizing and untokenizing methods, dataset preparation, and
     batch generation.
     
     Note: if the dataset is expected to remain unchanged, it is good practice
     to use the `num_types` method once to see the number of types and add that
     value to the `max_types` keyword argument in the constructor to avoid
     adding out-of-vocabulary tokens and instead mapping them to '_UNK'."""
    
  def __init__(self, max_types=None, gram_order=1, shuffle=False):
    """Keyword arguments:
       `max_types`: an upper bound for the vocabulary size (no bound if falsy),
       `gram_order`: number of original types in the n-gram,
       `shuffle`: whether to shuffle the list of input/label pairs."""
    self.max_types = max_types
    self.gram_order = gram_order
    self.shuffle = shuffle
    self.train_pairs = []
    self.valid_pairs = []
    # Data structures to tokenize and untokenize data in O(1) time.
    # By default, we add four tokens:
    # '_PAD': padding for examples with variable size and model of fixed size,
    # '_EOS': end of string added before padding to aid the prediction process,
    # '_GO': go token added as the first decoder input for seq2seq models,
    # '_UNK': unknown token used to cover unknown types in the dataset.
    self.ix_to_type = ['_PAD', '_EOS', '_GO', '_UNK', '<bos>', '<eos>', '<bow>', '<eow>', '<boq>', '<eoq>', "[+]", "[-]"]
    # Allow to add any extra defaults without modifying both data structures.
    dictarg = lambda i: [self.ix_to_type[i], i]
    self.type_to_ix = dict(map(dictarg, range(len(self.ix_to_type))))
  
  def trainTags(self, input_list):
    """ Converts the train input into a list that has tags seperate from normal characters
    Example: input of "<bos><bow>mnfukha<eow>shwaya<spa>walahy<eos>" would be converted to
    ['<bos>', '<bow>', 'm', 'n', 'f', 'u', 'k', 'h', 'a', '<eow>', 's', 'h', 'w', 'a', 'y', 'a', 
    '<spa>', 'w', 'a', 'l', 'a', 'h', 'y', '<eos>']
    """
    tags = ['<bos>', '<eos>', '<bow>', '<eow>', '<wb>', "<boq>", "<eoq>"]
    line = []

    i = 0
    while i < len(input_list):
        if input_list[i] == "<" and input_list[i:i+5] in tags:
            line.append(input_list[i:i+5])
            i += 5

        elif input_list[i] == "<" and input_list[i:i+4] in tags:
            line.append(input_list[i:i+4])
            i += 4
            
        else:
            line.append(input_list[i])
            i += 1

    return line

  def goldTags(self, input_list):
    """ Converts the GOLD input into a list that has tags seperate from normal characters
    Example: input of "Ajyb[-]lkw" would be converted to
    ['A', 'j', 'y', 'b', '[-]', 'l', 'k', 'w']
    """
    tags = ['[+]','[-]']
    line = []
   
    i = 0
    while i < len(input_list):
        if input_list[i] == "[" and input_list[i:i+3] in tags:
            line.append(input_list[i:i+3])
            i += 3
        else:
            line.append(input_list[i])
            i += 1

    return line

  def tokenize(self, input_list):
    """Converts the argument list or string to a list of integer tokens, each
       representing a unique type. If the charachter is not registered, it will
       be added to the `type_to_ix` and `ix_to_type` attributes."""

    if "[+]" in input_list or "[-]" in input_list:
      input_list = self.goldTags(input_list)
    else:
      input_list = self.trainTags(input_list)

    result = []
    for i in range(len(input_list) - (self.gram_order - 1)):

      gram = tuple(input_list[i:i+self.gram_order])  # lists are unhashable

      if gram not in self.type_to_ix:
        if not self.max_types or self.num_types() < self.max_types:
          self.type_to_ix[gram] = len(self.ix_to_type)
          self.ix_to_type.append(gram)
        else:
          gram = '_UNK'  # pylint: disable=redefined-variable-type
      result.append(self.type_to_ix[gram])
    return result

  
  def clean(self, tokens):
    """Remove the _EOS token and everything after it, if found."""
    try:
      tokens = tokens[:list(tokens).index(self.type_to_ix['_EOS'])]
    except ValueError:
      pass
    return tokens
  
  def untokenize(self, tokens, join_str=' '):
    """Convert the argument list of integer ids back to a string."""
    return join_str.join([self.ix_to_type[t][0] for t in tokens])
  
  def get_train_batches(self, batch_size):
    """Group the pairs into batches, and allows randomized order."""
    if self.shuffle:
      random.shuffle(self.train_pairs)
    end = (len(self.train_pairs) // batch_size) * batch_size
    return [
      self.train_pairs[i:i+batch_size] for i in range(0, end, batch_size)]
  
  def num_types(self):
    """Return the number of unique n-grams in the dataset."""
    return len(self.ix_to_type)
  
  def num_pairs(self):
    """Return the number of train and valid example pairs in the dataset."""
    return (len(self.train_pairs), len(self.valid_pairs))
