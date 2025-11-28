from mxnet import gluon
from gluon.data import Vocab, Counter
counter = Counter(tokens)
vocab = Vocab(counter)
print(vocab['mxnet'], vocab.to_indices(['deep', 'learning']))

