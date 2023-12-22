import numpy as np

'''
Notation reference 

x: Input sequence
y: Output sequence
X: Indexing into the positions and sequence of input x
X_t: Indexing into positions in the middle of the sequence
T_x: Length of the input sequence
T_y: Length of the output sequence
X(i): Referring to the i-th training example in the sequence of training examples
Tx_i: Input sequence length for training example i
y_i(t): TIF element in the output sequence of the i-th training example
Ty_i: Length of the output sequence in the i-th training example
Vocabulary or Dictionary: List of words used in the representations
x-1, x_2, x_3, ...: One-hot representations of individual words in the sentence
Unknown Word (UNK): Token used to represent words not in the vocabulary
'''

def ReLU(z):
    return np.maximum(0, z)
def sigmoid(z):
    return (1/(1+np.exp(-z)))


def forward_prop_at_t(x_t, last_a, W_ax, W_aa, W_ya, b_a, b_y):
    a = np.dot(x_t, W_ax) + np.dot(last_a, W_aa) + b_a
    a = ReLU(a)
    yhat = np.dot(a, W_ya) + b_y
    yhat = sigmoid(yhat)
    return a, yhat

def forward_prop(hidden_units, x, W_ax, W_aa, W_ya, b_a, b_y):
    last_a = np.zeros(hidden_units)
    outputs = []
    for x_t in x:
        last_a, yhat = forward_prop_at_t(x_t, last_a, W_ax, W_aa, W_ya, b_a, b_y)
        outputs.append(yhat)
    return outputs


