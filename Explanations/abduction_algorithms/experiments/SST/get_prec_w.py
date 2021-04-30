import argparse
import copy as cp
import numpy as np
import sys
import time
import tensorflow as tf
import enchant
from keras import backend as K
from tensorflow.keras.models import load_model

import pickle

from SST_REL_PATHS import MARABOUPY_REL_PATH, ABDUCTION_REL_PATH, KERAS_REL_PATH, FROZEN_REL_PATH, DATA_SAMPLES, EMBEDDINGS_REL_PATH, TRAIN_REL_PATH, RESULTS_PATH
sys.path.append(MARABOUPY_REL_PATH)
# import Marabou
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
# import abduction algorithms
sys.path.append(ABDUCTION_REL_PATH)
from abduction_algorithms_HS_cost import logger, freeze_session, Entails, smallest_explanation_with_cost
# import embeddings relative path
sys.path.append(EMBEDDINGS_REL_PATH)
sys.path.append(TRAIN_REL_PATH)
from glove_utils import load_embedding, pad_sequences

# routines for adversarial attacks (fgsm, pgd, ..)
import adversarial_attacks as adv

emb_dims = 5
input_len = emb_dims*50
num_words = int(input_len/emb_dims)
#ksize = int(num_words**0.5)
#model_input_shape = (1, ksize, ksize, emb_dims)
model_input_shape = (1, emb_dims*num_words)
tf_model_path = KERAS_REL_PATH + 'fc-{}inp-16hu-keras-SST-{}d'.format(num_words, emb_dims)

# Load model and test the input_ review
model = load_model(tf_model_path, compile=False)

# Load embedding
EMBEDDING_FILENAME = EMBEDDINGS_REL_PATH+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

predictions = []
inputs = ['you will probably love it', 'a delightful coming-of-age story', 'insanely hilarious !', 'Yet the act is still charming here', 'It is amusing and that is all it needs to be', 'The extent to which it succeeds is impressive', 'Strange funny twisted brilliant and macabre', 'Still this flick is fun and host to some truly excellent sequences', 'A rigorously structured and exquisitely filmed drama about a father and son connection that is a brief shooting star of love', 'An operatic sprawling picture that is entertainingly acted magnificently shot and gripping enough to sustain most of its 170-minute length', 'More romantic more emotional and ultimately more satisfying than the teary-eyed original', 'Whether or not you are enlightened by any of Derrida is lectures on the other and the self Derrida is an undeniably fascinating and playful fellow', 'Exquisitely nuanced in mood tics and dialogue this chamber drama is superbly acted by the deeply appealing veteran Bouquet and the chilling but quite human Berling', 'It is a much more emotional journey than what Shyamalan has given us in his past two movies and Gibson stepping in for Bruce Willis is the perfect actor to take us on the trip', 'You really have to salute writer-director Haneke ( he adapted Elfriede Jelinek novel ) for making a film that is not nearly as graphic but much more powerful brutally shocking and difficult to watch', 'Star/producer Salma Hayek and director Julie Taymor have infused Frida with a visual style unique and inherent to the titular character paintings and in the process created a masterful work of art of their own', 'This is wild surreal stuff , but brilliant and the camera just kind of sits there and lets you look at this and its like you are going from one room to the next and none of them have any relation to the other', 'This odd poetic road movie spiked by jolts of pop music pretty much takes place in Morton ever-watchful gaze -- and it as a tribute to the actress and to her inventive director that the journey is such a mesmerizing one', 'Still this flick is fun and host to some truly excellent sequences', 'It is a satisfying summer blockbuster and worth a look', 'it is probably not easy to make such a worthless film', 'this is one baaaaaaaaad movie', 'the gorgeously elaborate continuation of The Lord of the Rings trilogy is so huge that a column of words can not adequately describe co-writer/director Peter Jackson expanded vision of J.R.R. Tolkien Middle-earth', 'when Leguizamo finally plugged an irritating character late in the movie', 'at least one scene is so disgusting that viewers may be hard pressed to retain their lunch', 'Like its title character Esther Kahn is unusual but unfortunately also irritating', 'Staggeringly dreadful romance', 'an incredibly irritating comedy about thoroughly vacuous people ... manages to embody the worst excesses of nouvelle vague without any of its sense of fun or energy', 'the film just might turn on many people to opera in general an art form at once visceral and spiritual wonderfully vulgar and sublimely lofty and as emotionally grand as life', 'as vulgar as it is banal', 'just dreadful', 'a crushing disappointment', 'it is not life-affirming its vulgar and mean but I liked it', 'the draw for Big Bad Love is a solid performance by Arliss Howard', 'there are far worse messages to teach a young audience which will probably be perfectly happy with the sloppy slapstick comedy', 'the main story ... is compelling enough but it is difficult to shrug off the annoyance of that chatty fish', 'a painfully funny ode to bad behavior', '... spellbinding fun and deliciously exploitative', 'ah what the hell', 'this one is not nearly as dreadful as expected']

padded_inputs = []
# Review + <pad>(s)
for input_without_padding in inputs:
    input_without_padding = input_without_padding.lower().split(' ') 
    input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
    orig_input = input_.copy()
    padded_inputs.append(orig_input)
    x = embedding(input_)
    input_shape = x.shape
    prediction = model.predict(x)
    input_ = x.flatten().tolist()
    y_hat = np.argmax(prediction)
    c_hat = np.max(prediction)
    predictions.append(y_hat)

pred_dict = dict()
for idx, padded_input in enumerate(padded_inputs):
    for word in padded_input:
        if not word in pred_dict.keys():
            # [0             ,  0               ]
            # bad preds (0's),  good preds (1's)]
            pred_dict[word] = [0,0]
        pred_dict[word][int(predictions[idx])] += 1


for key in pred_dict.keys():
    s = sum(pred_dict[key])
    new_0 = pred_dict[key][0] / s
    new_1 = pred_dict[key][1] / s
    pred_dict[key] = [new_0,new_1]

print(pred_dict)

with open("prec_dict.pkl", "wb") as f:
    pickle.dump(pred_dict,f)

