"""
Take a model saved with Keras+Tensorflow and save the frozen graph in a .pb format.model.layers[idx].get_config()
"""
import argparse
import copy as cp
import numpy as np
import sys
import time
import tensorflow as tf
from tensorflow.keras import backend as K
from operator import itemgetter
from tensorflow.keras.models import load_model
from scipy.spatial import ConvexHull

from SST_REL_PATHS import MARABOUPY_REL_PATH, ABDUCTION_REL_PATH, KERAS_REL_PATH, FROZEN_REL_PATH, DATA_SAMPLES, EMBEDDINGS_REL_PATH, TRAIN_REL_PATH, RESULTS_PATH
sys.path.append(MARABOUPY_REL_PATH)
# import Marabou
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
# import abduction algorithms
sys.path.append(ABDUCTION_REL_PATH)
from abduction_algorithms_HS_cost import logger, freeze_session, knn_smallest_explanation_linf_with_cost
# import embeddings relative path
sys.path.append(EMBEDDINGS_REL_PATH)
sys.path.append(TRAIN_REL_PATH)
from embedding import Embedding
from glove_utils import load_embedding, pad_sequences
from text_utils import test_SST

# routines for adversarial attacks (fgsm, pgd, ..)
import adversarial_attacks as adv

def str2bool(v):
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Parse epsilon and window size from command line
parser = argparse.ArgumentParser(description='Input string, window size and epsilon can be passed as arguments.')
parser.add_argument('-i', '--input', dest='input', type=str, default='it is probably not easy to make such a worthless film', help='input string')
parser.add_argument('-w', '--window-size', dest='wsize', type=int, default=25, help='window size')
parser.add_argument('-e', '--epsilon', dest='eps', type=float, default=0.1, help='epsilon for input bounds')
parser.add_argument('-k', '--kneighbors', dest='knn', type=int, default=25, help='Number of k-nearest-neighbors used to find the convex hull around each word from the input sequence')
parser.add_argument('-u', '--uniform', dest='uniform', type=str2bool, default=True, help='cost function uniform=True, else False')
parser.add_argument('-a', '--adv', dest='adv_attacks', type=int, default=0, help='Number of adversarial attacks (0 means disabling )')
parser.add_argument('-x', '--excl', dest='excl', type=str, default='', help='comma separated words to exclude')
args = parser.parse_args()
# Assign eps and wsize
window_size = args.wsize
eps = args.eps
input_without_padding = args.input
knn = args.knn
adv_sims = int(args.adv_attacks)
uniform = args.uniform
excl_list = args.excl.split(',')

# Global Variables
prefix = ''
verbose = True
randomize_pickfalselits = False  # shuffle free variables before feeding them to pickfalselits
emb_dims = 5
input_len = emb_dims*10
num_words = int(input_len/emb_dims)
model_input_shape = (1, emb_dims*num_words)
tf_model_path = KERAS_REL_PATH + '{}fc-{}inp-16hu-keras-SST-{}d'.format(prefix, num_words, emb_dims)
frozen_graph_prefix = FROZEN_REL_PATH + 'tf_model_{}.pb'.format(np.random.randint(0, 1e7))  # random seed to prevent conflict between multiple instances
log_results, logs_path = True, "HS-clear/HS_CLEAR_SST_fc_{}_inp_{}d_knn_linf/HS_CLEAR_results_smallest_cost_exclude_expl_SST_{}fc_{}_inp_{}d".format(num_words, emb_dims, prefix, num_words, emb_dims)  # write results on file
HS_maxlen = 1e7  # max size of GAMMA in smallest_explanation

# Load model and test the input_ review
model = load_model(tf_model_path, compile=False)

# Load embedding
EMBEDDING_FILENAME = EMBEDDINGS_REL_PATH+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, index2word, index2embedding = load_embedding(EMBEDDING_FILENAME)
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

# Test accuracy of the model
acc = test_SST(load_model(tf_model_path, compile=True), index2embedding, word2index, [emb_dims*num_words], num_words, emb_dims, data_path='./../../../data/SST_2')
logger("Loss/Accuracy of the model on test set is {}".format(acc), True, "[logger]")

# Review + <pad>(s)
input_without_padding = input_without_padding.lower().split(' ') 
input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
orig_input = input_.copy()
x = embedding(input_)

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

# ************ word_cost_func ***********
# Where the word cost function is constructed, this can be constructed however you like.
if uniform:
    print('Uniform cost.')

word_cost_func = dict()
for word in orig_input:
    if not word in word_cost_func.keys():
        # if its a pad or not in the embedding, give it a high cost
        #if word == '<PAD>' or word2index[word] <= 2 or word == 'arliss' or word == 'howard':
        if word in excl_list:
            print()
            print('*****************')
            print('excluding:',word)
            print('*****************')
            print()
            if uniform:
                word_cost_func[word] = 1
            else:
                word_cost_func[word] = 100
        else:
            word_cost_func[word] = 1

# ***************************************

# Graph
filename = frozen_graph_prefix
model_without_softmax = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
tf.saved_model.save(model_without_softmax, frozen_graph_prefix)
output_constraints = [y_hat, (1 if y_hat==0 else 0), 1e-3]

# Start Main
# Args and procedures for parallelizing search of an attack
# args are in order: model, input_, y_hat, num_classes, targeted=True,
#                    loss="categorical_cross_entropy", mask=[], 
#                    minmax_input_bounds=[epsilons], epochs=100, feature_size=window_size
fgsm_args = (model, x, y_hat, 2, False, "categorical_cross_entropy", [], minmax_input_bounds, 1)
adv_args = (adv.get_adversarial_FGSM, fgsm_args)

weights_softmax = model.layers[-1].get_weights()
h, exec_time, GAMMA = knn_smallest_explanation_linf_with_cost(model, filename, orig_input, x, word_cost_func, minmax_input_bounds, y_hat, [eq_convex_hull, minmax_input_bounds], output_constraints, window_size, weights_softmax,
                                                    adv_attacks=(True if adv_sims>0 else False), adv_args=adv_args, sims=adv_sims, randomize_pickfalselits=randomize_pickfalselits, HS_maxlen=HS_maxlen, verbose=verbose)

expl_in_words = []
for i in range(num_words):
    tmp_idx = i*emb_dims
    if tmp_idx in h:
        expl_in_words.append(orig_input[i])

# Report MSR found
print("Minimum Size Explanation found {} (size {})".format(h, len(h)/window_size))
print(str(expl_in_words))
print("Complementary set of Minimum Size Explanation is {}".format([i for i in range(input_len) if i not in h]))
print("Execution Time: {}".format(exec_time))

# Write on file the results
if log_results is True:
    logger("Writing results on {}".format(RESULTS_PATH + logs_path), verbose=True, log_type='logger')
    file_ = open(RESULTS_PATH + logs_path, "a")
    if uniform:
        cost_str = 'uniform'
    else:
        cost_str = 'exclude ' + str(excl_list) 
    logs = "Input: {}\nKNN, Window Size, Adv_attacks: {}, {}, {}\nCost: {}\nOriginal output (confidence): {} ({})\nMinimal Expl: {} ({} words)\nExpl words: {}\nExec Time: {}\n\n".format(input_without_padding, knn, window_size, (True if adv_sims>0 else False), cost_str, y_hat, c_hat, h, int(len(h)/window_size), expl_in_words, exec_time)
    file_.write(logs)
    file_.close()
