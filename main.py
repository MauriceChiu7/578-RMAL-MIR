import torch
import torch.nn as nn
import numpy as np

def train_model(data):
    pass

def pretrain_policy():
    pass

def oracle(x):
    pass

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
def rmal_al(init_train_set, data_stream, budget, batch_size, episodes):
    # pi <- random - initialize AL agent policy = random
    prediction = torch.rand(1)

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
        bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(prediction)) 
        
        # a_t = 1 means AL policy decides to query the label of x_t
        a_t = bernoulli.sample()
        
        if (u/t < budget and a_t == 1):
            y = oracle(data_stream[t])

    pass


if __name__ == "__main__":
    rmal_al(None, None, None, None, None)
