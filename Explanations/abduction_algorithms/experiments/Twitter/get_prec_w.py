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

from Twitter_REL_PATHS import MARABOUPY_REL_PATH, ABDUCTION_REL_PATH, KERAS_REL_PATH, FROZEN_REL_PATH, DATA_SAMPLES, EMBEDDINGS_REL_PATH, TRAIN_REL_PATH, RESULTS_PATH
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
prefix = ""
emb_dims = 5
input_len = emb_dims*25

num_words = int(input_len/emb_dims)
ksize, emb_dims = int(num_words**0.5), 5
model_input_shape = (1, ksize, ksize, emb_dims)
tf_model_path = KERAS_REL_PATH + '{}cnn2d-{}inp-keras-Twitter-{}d'.format(prefix, num_words, emb_dims)

# Load model and test the input_ review
model = load_model(tf_model_path, compile=False)

# Load embedding
EMBEDDING_FILENAME = EMBEDDINGS_REL_PATH+'custom-embedding-Twitter.{}d.txt'.format(emb_dims)
word2index, index2word, index2embedding = load_embedding(EMBEDDING_FILENAME)
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

predictions = []
inputs =  ['cant sleep but im still feelin like a piece of shit',
 'I couldnt bear to watch it  And I thought the UA loss was embarrassing ...',
 'spencer is not a good guy',
 'My nap was interrupted so many times today Going out for Japanese with the arents again ...',
 'Unfortunately one of those moments wasnt a giant squid monster',
 'aw sorry to hear that',
 'nah I havent received my stimulus yet',
 'i just seen ur tweet plz write bak if u get this i havnt got one reply bak',
 'Awwwww and you were trying to go to sleep 3 hours ago',
 'I am so sorry to hear that It is always sad when we lose those close to us as we loved them',
 'I am doing the time warp without you and am sad',
 'this is true lol but it is still a slap in the face after such a warm end of march',
 'I hate converting movies just to put em on my itouch',
 'gross i have a pimple',
 'time to come back to flawda for double dates! no seriously i am sorry to hear that',
 'I am up way to late to be working for a client 12:10 AM  #fb',
 'blegghhhh i have to go to work',
 'I emailed you yesterday and u never responded',
 'Just got done watching the new House episode Definitely one of the saddest episodes ever',
 'Man ... taxes suck Im horrified that i did something wrong on them TurboTax decided to keep around a lot of the stuff I turned off',
 'Is up and off to get in the shower Hope everything runs smoothly today',
 'Have a great time tonight guys looks like it will be massive I have girlfriend night but wish you all the best',
 'Awww no but yay for being there already!', 
 'Great! Enjoy the riverside if you can',
 'morning!! beautiful isnt it! what you got planned for today?', 
 'great site and i like the angel part',
 'showed me http://www.fmylife.com/ and its quite funny',
 'Pleasure to meet you as well!  It is always good to be connected with a heart specialist!',
 'Wow!! I actually want to read right now ...  Weird!!!  i know someone would be happy!! Hahaha ;)',
 'toothache subsiding thank god for extra strength painkillers',
 'Has summer finally arrived? Hurrah for sunshine',
 'awe my friend Kyle is amazing! thas my baby right there', 
 'You are a gentleman! Thank you for your kind words',
 'Cant wait till I have a friend who is expecting so I can give them such a beautifully presented gift box of lovelyness',
 'so your entire day was spent doing chores ay??!! that sounds like sooo much fun', 
 'is delighted by the beautiful weather', 
 'Guess who got a job',
 'watching dvd all day long hmm discover some new hottie boys',
 'Eating breakfast Then later I am gonna freshen myself up cause we are going to City Center today',
 'Just got home from school we only had 2 hours']

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

