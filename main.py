import torch
import torch.nn as nn
import numpy as np
import random

# TODO: alpha - Tune this learning rate
LEARNING_RATE = 0.001

# TODO: gamma - Tune this discounting factor
DISCOUNTING_FACTOR = 0.9

# TODO: Implement these functions
def get_validation_set():
    pass

def train_model(data):
    pass

def pretrain_policy():
    pass

def oracle(x):
    pass

def accuracy(clf, validation_set):
    pass

# TODO: Get validation set
VALIDATION_SET = get_validation_set()

def update_policy(al_policy, clf, train_set, batch, episodes, validation_set=VALIDATION_SET, learning_rate=LEARNING_RATE):
    """
    Algorithm 2: Update Agent

    inputs:
    al_policy:      pi_phi          - the current AL agent policy
    clf:            f               - the trained classifier since last batch update
    train_set:      D               - the training set
    batch:          N               - the batch of newly labeled samples (z_1, y_1), ..., (z_T, y_T)
    episodes:       E               - the number of training episodes

    outputs:
    new_al_policy:  pi'             - the updated AL agent policy
    """
    # M <- {} - initialize the memory to be an empty set
    memory = []
    acc_old = accuracy(clf, validation_set)
    for e in range(episodes):
        proxy_clf = clf
        random.shuffle(batch)
        log_probs = []

        for (z_t, y_t) in batch:
            prediction = proxy_clf(z_t)
            # TODO: should it be a different distribution for each z_t?
            bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(al_policy(prediction)))
            a_t = bernoulli.sample()
            log_probs.append(bernoulli.log_prob(a_t))

            if (a_t == 1):
                memory = memory.append((z_t, y_t))
                proxy_clf = train_model(train_set.append(memory))
                acc = accuracy(proxy_clf, validation_set)
                
                # reward signal r_t which is subsequently used to update the agent
                r_t = (acc-acc_old)/acc_old

            else:
                cf_clf = train_model(train_set.append(memory.append((z_t, y_t))))
                acc_cf = accuracy(cf_clf, validation_set)
                r_t = -(acc_cf-acc_old)/acc_old
                # proxy_clf remains the same
                # acc remains the same

        # TODO: al_policy here should really be the parameters of the policy
        policy = al_policy + learning_rate * gradient(al_policy)

    return policy # TODO: Not sure if this is the policy we want to return

def rmal_al(init_train_set, data_stream, budget, batch_size, episodes):
    """
    Algorithm 1: RMAL-AL algorithm

    inputs:
    init_data:      D_0             - initial dataset
    data_stream:    {x_1, ..., x_N} - data stream
    budget:         b               - the number of samples u_t selected for training until time t
    batch_size:     T               - the number of samples required for retraining
    episodes:       E               - the number of training episodes for the udate_agent() function

    outputs:
    clf:            f               - the final classifier
    """
    # pi <- random - initialize AL agent policy = random
    # TODO: not sure what this policy should be
    al_policy = torch.rand(1)

    # D <- D_0 - initialize the training set to be the initial dataset
    train_set = init_train_set

    # N <- {} - initialize the batch to be an empty set
    batch = []

    # u <- 0 - number of training samples accumulated by the active learner
    u = 0

    clf = train_model(train_set)
    proxy_clf = clf
    pretrain_policy()
    for t in range(len(data_stream)):
        # clf is a regressor which outputs (0, 1]
        prediction = clf(data_stream[t])
        
        # there is a probability of "prediction" that the label of x_t is 1
        # TODO: should it be a different distribution for each x_t?
        bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(al_policy(prediction))) 
        
        # a_t = 1 means AL policy decides to query the label of x_t
        a_t = bernoulli.sample()
        
        if (u/t < budget and a_t == 1):
            y = oracle(data_stream[t])
            batch = batch.append((data_stream[t], y))
            clf = train_model(train_set.append(batch))
            u += 1
            if (len(batch) == batch_size):
                al_policy = update_policy(al_policy, proxy_clf, train_set, batch, episodes)
                train_set = train_set.append(batch)
                batch = []
                proxy_clf = clf
        else:
            # clf remains the same
            pass

    pass


if __name__ == "__main__":
    rmal_al(None, None, None, None, None)
