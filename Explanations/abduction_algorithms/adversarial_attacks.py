import copy as cp
import keras.backend as K
import numpy as np
import random as rn
import tensorflow as tf
from tqdm import tqdm
from os import path

def idx2word(indices, window_size, input_len):
    """
    When a feature is selected by MinimumHS, the entire window is selected
    """
    res = []
    for idx in indices:
        for r in range(0, input_len, window_size):
            if idx<r+window_size and idx>=r and (idx not in res):
                res.extend([el for el in range(r, min(r+window_size, input_len))])
    return res


def get_adversarial_FGSM(model, input_, y_hat, num_classes, 
                         targeted=True, loss="categorical_cross_entropy", mask=[], eps=1e-3, epochs=100):
    """
    Return a list of adversarial attacks for different values of eps
    Input:
        model:tf.keras.model
            keras model with tensorflow backend
        input_:numpy.array
            input
        y_hat:int
            true class
        num_classes:int
            number of output classes
        targeted:boolean
            (optional) targeted or untargeted attack
        loss:string
            (optional) name of the Keras Backend loss used
        mask:list
            (optional) list of variables to be excluded from the formulation of the attack
        eps:float/list
            (optional) epsilon used to generate the attack at each iteration. If it is a list, is specifies a value
             for each entry of a flattened input.
        epochs:int
            (optional) max number of iterations to formulate the attack 
    Output:
        success:boolean
            True if the attack is successfull (i.e. it changes the classification)
        adv:numpy.array or dictionary
            attack found (or the best found, if unsuccessful)
    """  
    if len(mask) == 0:
        m = 1
    else:
        input_shape = input_.shape
        m = np.ones(np.prod(input_shape))
        m[mask] = 0.
        m = m.reshape(input_shape)
    # cache the gradient for next iterations
    path_blob_hash = './__pycache__/gradient_FGSM' + str(hash(str(model)+str(input_)+str(y_hat))) + '.npy'
    if path.exists(path_blob_hash):
        delta = np.load(path_blob_hash, allow_pickle=True)
        noise = np.zeros_like(input_)
        # compute and save hash to speedup re-utilization
        if isinstance(eps, list):
            eps = 0.95*np.array([(eps[i][1] if d > 0 else eps[i][0]) for i,d in enumerate(delta.flatten())]).reshape(input_.shape)
        for _ in range(epochs):
            noise += eps*delta*m
            if np.argmax(model.predict(input_ + noise)) != y_hat:
                return True, input_+noise
    else: 
        if loss == "categorical_cross_entropy":
            loss_object = K.categorical_crossentropy
        else:
            raise NotImplementedError("{} loss has not been implemented.".format(loss))
        #sess = K.get_session()
        target = tf.reshape(K.one_hot(y_hat, num_classes), (1,2))
        t = (-1 if targeted is True else 1)  # targeted vs. untargeted
        noise = np.zeros_like(input_)
        input_tf = tf.cast(input_, tf.float32)  # tf representation of the input
        with tf.GradientTape() as tape:
            tape.watch(input_tf)
            prediction = model(input_tf)
            loss = loss_object(target, prediction)
        grads = tape.gradient(loss, input_tf)
        delta = t*tf.sign(grads).numpy()
        # compute and save hash to speedup re-utilization
        print("[logger] Storing hash for current input in {}".format(path_blob_hash))
        print("value: {}".format(delta))
        np.save(path_blob_hash, delta)
        # If eps is a list, we are doing *real* PGD
        if isinstance(eps, list):
            eps = 0.95*np.array([(eps[i][1] if d > 0 else eps[i][0]) for i,d in enumerate(delta.flatten())]).reshape(input_.shape)
        for _ in range(epochs):
            noise += eps*delta*m
            if np.argmax(model.predict(input_ + noise)) != y_hat:
                return True, input_+noise
    # attack was unsuccessful (using cache or not)
    return False, input_+noise

def get_adversarial_SparseRS(model, input_, input_bounds, y_hat, num_classes,
                            k=1, method="max-bound", num_queries=1000, mask=[], PGDargs=(False, 1e-3, 100), feature_size=1):
    """
    Return an adversarial attack that exploits exactly k dimensions in the input space.
    The method performs a random search, i.e., each new iteration is accepted if and only if the confidence
     of the model is decreased wrt the previous point.
    Input:
        model:tf.keras.model
            keras model with tensorflow backend
        input_:numpy.array
            input in the correct shape for model.predict (i.e., (1, dim1, dim2, .., dimn))
        input_bounds:numpy.array
            bounds for each input in the correct shape for model.predict (i.e., (1, dim1, dim2, .., dimn))
        y_hat:int
            true class
        num_classes:int
            number of output classes
        window_size:int
            embedding dimension and size of each feature
        k:int
            (optional) number of dimensions checked
        method:string
            (optional) method to extract the perturbation on each dimension. This defines the constraints for the attacks. 
             Default is "max-bound" that uses the maximum value of the correspective input_bounds value.
             Other methods available are "min-bound" (uses min-value), "maxmin-bound" that uses a random mix,
             and "PGD"
        num_queries:int
            (optional) max number of queries
        mask:list
            (optional) list of variables to be excluded from the formulation of the attack
        PGDargs:float
            (optional) targeted, epsilon and epochs used to generate the PGD attack at each iteration
        feature_size:int
            size of each feature, also known as window_size
    Output:
        success:boolean
            True if the attack is successfull (i.e. it changes the classification)
        adv:numpy.array
            attack found (or the best found, if unsuccessful)
    """
    target_class, confidence = np.argmax(model.predict(input_)), np.max(model.predict(input_))  # calculate confidence of the classifier
    input_shape = input_.shape
    input_len = len(input_.flatten())
    best_attack = cp.copy(input_).flatten()
    ###
    vars_ = [i for i in range(0, input_len, feature_size) if i not in mask]  # should check dim of vars_ >= k
    step, found = 0, False
    while step <= num_queries and found != True:
        perturbed_input = cp.copy(input_).flatten()
        ###
        idx_perturbations = rn.sample(vars_, np.random.randint(0,len(vars_)))  # select k indices to sample perturbations
        idx_perturbations = idx2word(idx_perturbations, feature_size, input_len)
        # extract best perturbations
        if method == "max-bound":
            perturbed_input[idx_perturbations] = [input_bounds[i][1] for i in idx_perturbations]
        elif method == "min-bound":
            perturbed_input[idx_perturbations] = [input_bounds[i][0] for i in idx_perturbations]
        elif method == "minmax-bound":
            perturbed_input[idx_perturbations] = [input_bounds[i][rn.randint(0,1)] for i in idx_perturbations]
        elif method == "PGD":  
            # TODO: implement control on the solution such that it satisfies bounds on any Lp-norm
            is_legit_attack, perturbed_input = get_adversarial_FGSM(model, input_, y_hat, num_classes, targeted=PGDargs[0], mask=[i for i in range(input_len) if i not in idx_perturbations], eps=PGDargs[1], epochs=PGDargs[2])
        else:
            raise NotImplementedError("{} is not recognized as perturbation method.".format(method))
        # acceptance step for the newly generated point
        new_confidence = model.predict(perturbed_input.reshape(input_shape))[0][target_class]
        if new_confidence < confidence and is_legit_attack:
            best_attack = cp.copy(perturbed_input)
            confidence = new_confidence
        step += 1
        if new_confidence < 0.49 and is_legit_attack:  # threshold is larger than 0.5 to void numerical errors and guarantee strong attacks
            found = True
    return found, best_attack.reshape(input_shape)

def optimize_sparseRS(model, input_, input_bounds, y_hat, num_classes, 
                      k=10, sims=100, mask=[], PGDargs=(False, 1e-3, 100), feature_size=1):
    """
    Return an adversarial attack that exploits at most k dimensions in the input space.
     PGD method is used to find the best attack at each stage and if found and till a budget is
     consumed, another attack strictly smaller is searched.
    Input:
        model:tf.keras.model
            keras model with tensorflow backend
        input_:numpy.array
            input in the correct shape for model.predict (i.e., (1, dim1, dim2, .., dimn))
        input_bounds:numpy.array
            bounds for each input in the correct shape for model.predict (i.e., (1, dim1, dim2, .., dimn))
        y_hat:int
            true class
        num_classes:int
            number of output classes
        k:int
            (optional) initial number of dimensions that are checked for a sparse attack
        sims:int
            (optional) size of the population of attacks generated at each stage
        mask:list
            (optional) list of variables to be excluded from the formulation of the attack
        PGDargs:float
            (optional) targeted, epsilon and epochs used to generate the PGD attack at each iteration
        feature_size:int
            size of each feature, also known as window_size
    Output:   
        min_k:int
            minimum number of inputs that have been changed by the best attack     
        best_attack:numpy.array
            best sparse attack found
        population:list
            list of all the attacks found for each number of features modified in [k, min_k]
    """
    print("[logger]: Searching for Sparse Adversarial Attacks with minimum size {}".format(k))
    min_k = k
    best_attack, population = None, []
    while min_k > 0:
        num_attacks = 0
        print("[logger]: Gathering attacks with {} variables.".format(min_k))
        for _ in tqdm(range(sims)):
            found, attack = get_adversarial_SparseRS(model, input_, input_bounds, y_hat, num_classes, k=min_k, 
                                                     method="PGD", num_queries=1, mask=mask, PGDargs=PGDargs, feature_size=feature_size)
            if found is True:
                #print("Attack found with size {}, pop_len={}".format(min_k, len(population)))
                num_attacks += 1
                population.append(attack)
                best_attack = attack
        # Break if at the previous stage no attacks were found
        if num_attacks == 0:
            break
        min_k -= 1
    print("[logger]: {} attacks found".format(len(population)))
    # Keep only strong attacks
    false_alarms, new_population = 0, []
    for p in population:
        if model.predict(p)[0][y_hat] > 0.5:
            false_alarms += 1
        else:
            new_population.append(p)
    population = new_population
    print("[logger]: SparseRS routine: total attacks found {}, minimal size {}, attacks removed {}".format(len(population), min_k+1, false_alarms))
    return min_k, best_attack, population
        
