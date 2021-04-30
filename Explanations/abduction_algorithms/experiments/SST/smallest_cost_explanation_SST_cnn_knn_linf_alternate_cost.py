"""
Take a model saved with Keras+Tensorflow and save the frozen graph in a .pb format.model.layers[idx].get_config()
Execute this with the command %run ./examples/minimal_subset_SST.py
"""
import argparse
import copy as cp
import numpy as np
import sys
import time
import tensorflow as tf
from keras import backend as K
from operator import itemgetter
from tensorflow.keras.models import load_model
from scipy.spatial import ConvexHull
import pickle 

from SST_REL_PATHS import MARABOUPY_REL_PATH, ABDUCTION_REL_PATH, KERAS_REL_PATH, FROZEN_REL_PATH, DATA_SAMPLES, EMBEDDINGS_REL_PATH, TRAIN_REL_PATH, RESULTS_PATH
sys.path.append(MARABOUPY_REL_PATH)
# import Marabou
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
# import abduction algorithms
sys.path.append(ABDUCTION_REL_PATH)
from abduction_algorithms_cost import logger, knn_smallest_cost_explanation_linf_alternate_cost
from abduction_algorithms import Entails
# import embeddings relative path
sys.path.append(TRAIN_REL_PATH)
from embedding import Embedding
from glove_utils import load_embedding, pad_sequences
from text_utils import test_SST


def str2bool(v):
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Parse epsilon and window size from command line
parser = argparse.ArgumentParser(description='Input string and epsilon can be passed as arguments.')
parser.add_argument('-i', '--input', dest='input', type=str, default='when Leguizamo finally plugged an irritating character late in the movie', help='input string')
parser.add_argument('-w', '--window-size', dest='wsize', type=int, default=5, help='window size')
parser.add_argument('-e', '--epsilon', dest='eps', type=float, default=0.1, help='epsilon for input bounds')
parser.add_argument('-u', '--uniform', dest='uniform', type=str2bool, default=True, help='cost function uniform=True, else False')
parser.add_argument('-k', '--kneighbors', dest='knn', type=int, default=7, help='Number of k-nearest-neighbors used to find the convex hull around each word from the input sequence')

args = parser.parse_args()
# Assign eps and wsize
window_size = args.wsize
eps = args.eps
uniform = args.uniform
input_without_padding = args.input
knn = args.knn

# Global Variables
verbose = True
prefix = ''  # may be set to 'robust-' to try IBP-trained models


# Global Variables
prefix = ''  # may be set to 'robust-' to try IBP-trained models
verbose = True
randomize_pickfalselits = False  # shuffle free variables before feeding them to pickfalselits
num_words = 25
ksize, emb_dims = int(num_words**0.5), 5
input_len = num_words*emb_dims
model_input_shape = (1, ksize, ksize, emb_dims)
input_bounds = [[-eps, eps] for _ in range(input_len)]
tf_model_path = KERAS_REL_PATH + '{}cnn2d-{}inp-keras-SST-{}d'.format(prefix, num_words, emb_dims)
frozen_graph_prefix, frozen_graph_path = FROZEN_REL_PATH, 'tf_model.pb'
log_results, logs_path = True, "MSA_NEW/MSA_NEW_SST_cnn_{}_inp_{}d_knn_linf/MSA_NEW_results_smallest_expl_SST_{}cnn_{}_inp_{}d_".format(num_words, emb_dims, prefix, num_words, emb_dims) +  "alternate_cost"  # write results on file+  


# Load model and test the input_ review
model = load_model(tf_model_path, compile=False)

# Load embedding
EMBEDDING_FILENAME = EMBEDDINGS_REL_PATH+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, index2word, index2embedding = load_embedding(EMBEDDING_FILENAME)
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

# Test accuracy of the model
acc = test_SST(load_model(tf_model_path, compile=True), index2embedding, word2index, [ksize, ksize, emb_dims], num_words, emb_dims, data_path='./../../../data/SST_2')
logger("Loss/Accuracy of the model on test set is {}".format(acc), True, "[logger]")

# Review + <pad>(s)
input_without_padding = input_without_padding.lower().split(' ') 
input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
x = embedding(input_)
inpforcost = input_


# Extract the k-nearest-neighbors
__Embedding = Embedding(word2index, index2word, index2embedding)
nearest_neighbors, eq_convex_hull, vertices_convex_hull = [], [], []
for i in input_:
    tmp = __Embedding.nearest_neighbors(i, knn, method='l2')
    nearest_neighbors += [[index2embedding[word2index[w]] for w in tmp[0]]]
    eq_convex_hull += [[eq.tolist() for eq in ConvexHull(nearest_neighbors[-1]).equations]]
    vertices_convex_hull += [[eq.tolist() for eq in ConvexHull(nearest_neighbors[-1]).vertices]]

# Extract max-min values (Marabu wants explicit bounds on each input...) from the vertices of the convex hull
for i in range(len(vertices_convex_hull)):
    vertices_convex_hull[i] = list(itemgetter(*vertices_convex_hull[i])(nearest_neighbors[i]))
vertices_convex_hull = np.array(vertices_convex_hull)
minmax_input_bounds = [[np.min(v, axis=0), np.max(v, axis=0)] for v in vertices_convex_hull]

# Check the Convex Hull is consistent (i.e., any point from the input is inside it)
# We leverage the convex hull normal form which, for an input x and matrices W,b is 
#   x*W - b <= \tolerance  // \threshold is close to zero 
for n in range(num_words):
    xx = x.reshape(1,num_words*emb_dims)
    xx = xx[:,n*window_size:(n+1)*window_size]
    for i, eq in enumerate(eq_convex_hull[n]):
        w,b = np.array(eq[:-1]).reshape(emb_dims,1), eq[-1]
        dp = np.dot(xx,np.array(w)) + b
        assert dp <= 1e-3, logger("The convex hull is NOT consistent! Error at equation {} (input {}), result of xW<=b is zero or negative, {}".format(i, n, dp), True, "[logger-ERROR]")
logger("The convex hull is consistent: each embedded point belongs to one of the respective {} facets equations".format(sum([len(eq) for eq in eq_convex_hull])), True, "[logger]")
 
input_shape = x.shape
prediction = model.predict(x)
input_ = x.flatten().tolist()
y_hat = np.argmax(prediction)
c_hat = np.max(prediction)
logger("Classifiation for the input is {} (confidence {})".format(y_hat, c_hat), verbose)

# Graph
filename = frozen_graph_prefix
model_without_softmax = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
tf.saved_model.save(model_without_softmax, frozen_graph_prefix)
output_constraints = [y_hat, (1 if y_hat==0 else 0), 1e-3]


weights_softmax = model.layers[-1].get_weights()

# ************ word_cost_func ***********
# word_cost_func['word'] = cost

with open('prec_dict.pkl','rb') as f:
    preds = pickle.load(f)

word_cost_func = dict()
for key in preds.keys():
    word_cost_func[key] = preds[key][int(y_hat)]


cost_func = dict()
m = x.flatten()
print(inpforcost)
for i in range(len(inpforcost)):
    for j in range(emb_dims):

        if m[i*emb_dims+j] == 0:
            cost_func[i*emb_dims+j] = 100
        else:
            cost_func[i*emb_dims+j] = word_cost_func.get(inpforcost[i])
      


h, exec_time = knn_smallest_cost_explanation_linf_alternate_cost(model, filename, x, minmax_input_bounds, [eq_convex_hull, minmax_input_bounds], output_constraints, num_words, uniform, weights_softmax, window_size, cost_func,  verbose=verbose)

#h, exec_time = smallest_cost_explanation_without_some_words(filename, x, eps, output_constraints, howManyWords, uniform, verbose=verbose)


# Report MSR found
logger("Minimum Cost Explanation found {}".format(h), True)
logger("Complementary set of Minimum Cost Explanation is {}".format([i for i in range(input_len) if i not in h]), True)
logger("Execution Time: {}".format(exec_time), True)

print("Explanation consists of  ", len(h)/emb_dims, " words.")


# Write on file the results
if log_results is True:
    logger("Writing results on {}".format(RESULTS_PATH + logs_path), verbose=True, log_type='logger')
    file_ = open(RESULTS_PATH + logs_path, "a")
    logs = "Input: {}\nKNN, Window Size: {}, {}\nOriginal output (confidence): {} ({})\nMinimal Expl: {}\nExec Time: {}\n\n".format(input_without_padding, knn, window_size, y_hat, c_hat, h, exec_time)
    file_.write(logs)
    file_.close()
