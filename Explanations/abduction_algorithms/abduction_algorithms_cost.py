"""
Implementation of Algorithm 1 of the paper "Minimum Satisfying Assignments for SMT" CAV 2013
---
Take a model saved with Keras+Tensorflow and save the frozen graph in a .pb format.model.layers[idx].get_config()
@Author:
"""

import itertools
import multiprocessing as mp
import numpy as np
import random
import tensorflow as tf
import time
from collections import deque
import random

from keras import backend as K
from pysat.examples.hitman import Hitman
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import linear as keras_linear
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import sys
sys.path.append('./../../../Marabou/')
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore

import marabou_rnn as marabou_rnn


from abduction_algorithms import Entails, freeze_session, logger, idx2word
level = 0
howmanyentails = 0


def create_initial_assignment(input_):
    key = []
    for i in range(len(input_)):
        key.append(i)
    initialAssign = dict(zip(key,input_))
    return initialAssign

def create_cost_function(initial_assignment, uniform=True):
    keyList = list(initial_assignment.keys())
    valList = []

    if uniform == True:
        for i in initial_assignment.values():
            valList.append(1)
    else:
        for i in initial_assignment.values():
            if i == 0.0:
                valList.append(100)
            else:
                valList.append(1)

    cost_function = dict(zip(keyList,valList))
    return cost_function

def find_highest_cost_word (candidateVariables, C):
    newC = {k: v for k, v in sorted(C.items(), key=lambda item: item[1], reverse=True)}

    new_list = [list(newC.keys())[i:i+5] for i in range(0, len(list(newC.keys())), 5)]
    return list(new_list)
        
        
    

def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def find_mus(listOfBounded, filename, candidateVariables, assignment, C, bound, input_bounds, output_constraints, weights_softmax, norm, window_size, verbose):
    """
    Input:
        listOfBounded: a list with variables which are bounded by forall operator in the original algorithm (variables without assinnment in input)
        network: Marabou network model
        candidateVariables: variables which are candidates for beeing in MUS
        assignment: original const. input for nn
        C: cost function
        bound: a bound for MUS algorithm
        input_bounds: bounds for variables which are in the listOfBounded
        output_constraints: constraints for the input of nn
    """
    global level
    global howmanyentails
    level += 1
    print()
    print("Level: ", level)

    if len(candidateVariables) == 0 or computeCost(C, candidateVariables) <= bound:
        level -= 1
        return []

    candidateVariablesCopy = candidateVariables.copy()
    network = Marabou.read_tf(filename, modelType='savedModel_v2', savedModelTags=['serving_default'])

    #if k == -1:
    candidateVariablesCopy.reverse()
    v = candidateVariablesCopy.pop()
    candidateVariablesCopy.reverse()
    #else:
    #    v = candidateVariablesCopy.pop(k)
    listOfBoundedCopy = listOfBounded.copy()
    listOfBoundedCopy.append(v)
    flatListoOfBoundedCopy =  [y for x in listOfBoundedCopy for y in x]

    const_ranges = [n for n in range(len(assignment)) if n not in flatListoOfBoundedCopy]
    best = []
    howmanyentails= howmanyentails+1
    res = Entails(const_ranges, network, list(assignment.values()),
                  input_bounds, output_constraints, weights_softmax,
                  window_size, norm,
                  verbose)

    network.clear()
    if len(res) == 0:
        Y = find_mus(listOfBoundedCopy, filename, shrink(filename, listOfBoundedCopy, assignment, candidateVariablesCopy, input_bounds, output_constraints, weights_softmax, window_size, norm, verbose), assignment, C, bound - computeCost(C, v), input_bounds, output_constraints,  weights_softmax, norm, window_size, verbose)
        #comment 119 if you dont want to use shrink
        #Y = find_mus(listOfBoundedCopy, filename, candidateVariablesCopy, assignment, C, bound - computeCost(C, v), input_bounds, output_constraints,  weights_softmax, norm, window_size, verbose)
        cost = computeCost(C, Y) + computeCost(C, v)
        if cost > bound:
            best = Union(Y, v)
            bound = cost

    Y = find_mus(listOfBounded, filename, candidateVariablesCopy, assignment, C, bound, input_bounds, output_constraints, weights_softmax, norm, window_size, verbose)

    if computeCost(C, Y) > bound:
        best = Y.copy()
    level -= 1
    return best


def computeCost(C, variables):
    assert(variables != None)
    cost = 0
    listOfVars = np.array(variables).flatten().tolist()
    for v in listOfVars:
        cost = cost + C[v]
    return cost

def candidateVarsPreparing(listOfAllVars, howManyWords):
    assert(listOfAllVars != None)
    nplistOfWords = np.array_split(listOfAllVars, howManyWords)
    listOfWords = []
    for i in nplistOfWords:
        listOfWords.append(i.tolist())
    return listOfWords

def smallest_cost_explanation(model, filename, numpy_input, eps, output_constraints, howManyWords, uniform, weights_softmax, norm, window_size, verbose=True):
    global n

    start_time = time.time()

    input_ = numpy_input.flatten().tolist()
    input_len = len(input_)

    input_bounds = [[input_[i]-eps, input_[i]+eps] for i in range(input_len)]

    assignment = create_initial_assignment(input_)

    costFunction = create_cost_function(assignment, uniform)

    candidateVariables = list(assignment.keys())
    candidateWords = candidateVarsPreparing(candidateVariables, howManyWords)
    listOfBounded = []
    mus = find_mus(listOfBounded, filename, candidateWords, assignment, costFunction, 0, input_bounds, output_constraints, weights_softmax, norm, window_size, verbose)

    result = []
    for v in list(assignment.keys()):
        if v not in mus:
            result.append(v)

    exec_time = time.time() - start_time
    print("Entails worked ", howmanyentails, " times.")
    return result, exec_time



def shrink(filename, listOfBounded, assignment, candidateVariables, input_bounds, output_constraints, weights_softmax,
                      window_size, norm, verbose):
    global howmanyentails
    essential_variables = candidateVariables.copy()
    listOfBoundedCopy = listOfBounded.copy()
    for j in candidateVariables:
        essential_variables.remove(j)
        network = Marabou.read_tf(filename, modelType='savedModel_v2',
                                  savedModelTags=['serving_default'])
        listOfBoundedCopy.append(j)
        flatListOfBounded =  [y for x in listOfBoundedCopy for y in x]
        const_ranges = [n for n in range(len(assignment)) if n not in flatListOfBounded]
        howmanyentails = howmanyentails+1

        res = Entails(const_ranges, network, list(assignment.values()), input_bounds, output_constraints, weights_softmax, window_size, norm, verbose)
        network.clear()
        if len(res) == 0:
            essential_variables.append(j)
        listOfBoundedCopy.remove(j)

    return essential_variables

def knn_smallest_cost_explanation(model, filename, numpy_input, eps, convex_hull_constraints, output_constraints, howManyWords, uniform, weights_softmax, window_size, verbose=True):
    global n

    start_time = time.time()

    input_ = numpy_input.flatten().tolist()
    input_len = len(input_)


    assignment = create_initial_assignment(input_)

    costFunction = create_cost_function(assignment, uniform)

    candidateVariables = list(assignment.keys())
    candidateWords = candidateVarsPreparing(candidateVariables, howManyWords)
    listOfBounded = []
    mus = find_mus(listOfBounded, filename, candidateWords, assignment, costFunction, 0, convex_hull_constraints, output_constraints, weights_softmax, "knn", window_size, verbose)

    result = []
    for v in list(assignment.keys()):
        if v not in mus:
            result.append(v)

    exec_time = time.time() - start_time
    print("Entails worked ", howmanyentails, " times.")
    return result, exec_time
    
    
    
def knn_smallest_cost_explanation_linf(model, filename, numpy_input, eps, convex_hull_constraints, output_constraints, howManyWords, uniform, weights_softmax, window_size, verbose=True):
    global n

    start_time = time.time()

    input_ = numpy_input.flatten().tolist()
    input_len = len(input_)

    assignment = create_initial_assignment(input_)

    costFunction = create_cost_function(assignment, uniform)

    candidateVariables = list(assignment.keys())
    candidateWords = candidateVarsPreparing(candidateVariables, howManyWords)
    listOfBounded = []
    mus = find_mus(listOfBounded, filename, candidateWords, assignment, costFunction, 0, convex_hull_constraints, output_constraints, weights_softmax, 'knn-linf', window_size, verbose)

    result = []
    for v in list(assignment.keys()):
        if v not in mus:
            result.append(v)

    exec_time = time.time() - start_time
    print("Entails worked ", howmanyentails, " times.")
    return result, exec_time
    
    
def knn_smallest_cost_explanation_linf_alternate_cost(model, filename, numpy_input, eps, convex_hull_constraints, output_constraints, howManyWords, uniform, weights_softmax, window_size, cost_func, verbose=True):
    global n
    input_ = numpy_input.flatten().tolist()
    input_len = len(input_)
    assignment = create_initial_assignment(input_)
    candidateVariables = list(assignment.keys())
    candidateWords = candidateVarsPreparing(candidateVariables, howManyWords)
    candidateWords = find_highest_cost_word(candidateWords, cost_func)

    
    listOfBounded = []
    start_time = time.time()
    mus = find_mus(listOfBounded, filename, candidateWords, assignment, cost_func, 0, convex_hull_constraints, output_constraints, weights_softmax, 'knn-linf', window_size, verbose)

    result = []
    for v in list(assignment.keys()):
        if v not in mus:
            result.append(v)

    exec_time = time.time() - start_time
    print("Entails worked ", howmanyentails, " times.")
    return result, exec_time
    
def rnn_smallest_cost_explanation_linf(model, numpy_input, eps, convex_hull_constraints, y_hat, howManyWords, uniform,  window_size, verbose=True):
    global n

    start_time = time.time()

    input_ = numpy_input.flatten().tolist()
    input_len = len(input_)

    assignment = create_initial_assignment(input_)

    costFunction = create_cost_function(assignment, uniform)

    candidateVariables = list(assignment.keys())
    candidateWords = candidateVarsPreparing(candidateVariables, howManyWords)
    listOfBounded = []
    mus = find_mus_rnn(listOfBounded, model, candidateWords, assignment, costFunction, 0, convex_hull_constraints, y_hat, 'knn-linf', window_size, numpy_input, verbose)

    result = []
    for v in list(assignment.keys()):
        if v not in mus:
            result.append(v)

    exec_time = time.time() - start_time
    print("Entails worked ", howmanyentails, " times.")
    return result, exec_time
    
    
def find_mus_rnn(listOfBounded, model, candidateVariables, assignment, C, bound, input_bounds, y_hat, norm, window_size, numpy_input, verbose):
    """
    Input:
        listOfBounded: a list with variables which are bounded by forall operator in the original algorithm (variables without assinnment in input)
        network: Marabou network model
        candidateVariables: variables which are candidates for beeing in MUS
        assignment: original const. input for nn
        C: cost function
        bound: a bound for MUS algorithm
        input_bounds: bounds for variables which are in the listOfBounded
        output_constraints: constraints for the input of nn
    """
    global level
    global howmanyentails
    level += 1
    print()
    print("Level: ", level)

    if len(candidateVariables) == 0 or computeCost(C, candidateVariables) <= bound:
        level -= 1
        return []

    candidateVariablesCopy = candidateVariables.copy()
   
    candidateVariablesCopy.reverse()
    v = candidateVariablesCopy.pop()
    candidateVariablesCopy.reverse()
   
    listOfBoundedCopy = listOfBounded.copy()
    listOfBoundedCopy.append(v)
    flatListoOfBoundedCopy =  [y for x in listOfBoundedCopy for y in x]

    const_ranges = [n for n in range(len(assignment)) if n not in flatListoOfBoundedCopy]
    best = []
    howmanyentails= howmanyentails+1
    res = marabou_rnn.Entails_knn_linf(const_ranges, model, numpy_input, y_hat, window_size, input_bounds)

    
    if len(res) == 0:
        Y = find_mus_rnn(listOfBoundedCopy, model, shrink_rnn(model, listOfBoundedCopy, assignment, candidateVariablesCopy, input_bounds, y_hat, norm, window_size, numpy_input, verbose), assignment, C, bound - computeCost(C, v), input_bounds, y_hat, norm, window_size, numpy_input, verbose)
        
        #Y = find_mus_rnn(listOfBoundedCopy, model, candidateVariablesCopy, assignment, C, bound - computeCost(C, v), input_bounds, y_hat, norm, window_size, numpy_input, verbose)
        cost = computeCost(C, Y) + computeCost(C, v)
        if cost > bound:
            best = Union(Y, v)
            bound = cost

    Y = find_mus_rnn(listOfBounded, model, candidateVariablesCopy, assignment, C, bound, input_bounds, y_hat, norm, window_size, numpy_input, verbose)

    if computeCost(C, Y) > bound:
        best = Y.copy()
    level -= 1
    return best        

def shrink_rnn(model, listOfBounded, assignment, candidateVariables, input_bounds, y_hat, norm, window_size, numpy_input, verbose):
    global howmanyentails
    essential_variables = candidateVariables.copy()
    listOfBoundedCopy = listOfBounded.copy()
    for j in candidateVariables:
        essential_variables.remove(j)
        listOfBoundedCopy.append(j)
        flatListOfBounded =  [y for x in listOfBoundedCopy for y in x]
        const_ranges = [n for n in range(len(assignment)) if n not in flatListOfBounded]
        howmanyentails = howmanyentails+1
        res = marabou_rnn.Entails_knn_linf(const_ranges, model, numpy_input, y_hat, window_size, input_bounds)
        if len(res) == 0:
            essential_variables.append(j)
        listOfBoundedCopy.remove(j)

    return essential_variables



def knn_smallest_cost_explanation_linf_alternate_cost_exclude_words(model, filename, numpy_input, listOfIdWordsToExclude, eps, convex_hull_constraints, output_constraints, howManyWords, uniform, weights_softmax, window_size, cost_func, verbose=True):
    global n

    input_ = numpy_input.flatten().tolist()
    input_len = len(input_)
    assignment = create_initial_assignment(input_)
    candidateVariables = list(assignment.keys())
    candidateWords = candidateVarsPreparing(candidateVariables, howManyWords)
    candidateWords = find_highest_cost_word(candidateWords, cost_func)
    candidateWordsExclude = []
    listOfBounded = listOfIdWordsToExclude
    for i in candidateWords:
        if not i in listOfBounded: 
            candidateWordsExclude.append(i)
            
    print("listOfBounded")
    print(listOfBounded)
    print("candidateWordsExclude")
    print(candidateWordsExclude)

    
 
    start_time = time.time()
    mus = find_mus(listOfBounded, filename, candidateWords, assignment, cost_func, 0, convex_hull_constraints, output_constraints, weights_softmax, 'knn-linf', window_size, verbose)

    result = []
    for v in list(assignment.keys()):
        if v not in mus:
            result.append(v)

    exec_time = time.time() - start_time
    print("Entails worked ", howmanyentails, " times.")
    return result, exec_time

