import argparse
import csv
import gym
import os
import pybullet
import math
from Model.cartpole_model import NN
import numpy as np
import torch
from utility import *

MAX_ITERATIONS = 200 # Number of updates to your gradient
# N_EPISODES = 500 # Number of episodes/rollouts
GAMMA = 0.99 # Discounting factor

def loss_f1(const_return, log_probs):
    # log_probs = torch.FloatTensor(log_probs) # WILL LOSE grad_fn
    # log_probs = torch.tensor(log_probs) # WILL LOSE grad_fn
    # if args.verbose: print(f"===> log_probs[0]:\t", log_probs[0])

    # print("\nCalculating loss with function 1\n")
    loss = -1 * torch.sum(
            torch.mul(
                log_probs, 
                const_return
            )
        )
    # print(loss)
    return loss

def loss_f2(args, returns, log_probs):
    # log_probs = torch.FloatTensor(log_probs)
    loss = 0
    if args.model == '2':
        # print("\nCalculating loss with function 2\n")
        loss = -1 * torch.sum(
                torch.mul(
                    log_probs,
                    returns
                )
            )
    elif args.model == '3':
        # print("\nCalculating loss with function 3\n")
        baseline = torch.mean(returns)
        sigma = torch.std(returns)
        loss = -1 * torch.sum(
                torch.mul(
                    log_probs, 
                    torch.div(
                        torch.sub(
                            returns, 
                            baseline
                        ), 
                        sigma
                    )
                )
            )
    else:
        loss = None
        print("Wrong input. Terminates")
        exit(0)
    # print(loss)
    return loss

def getReturn(rollout, gamma):
    ret = torch.tensor(0.)
    power = 0
    for i in range(len(rollout)):
        ret += (gamma**power)*rollout[i][3]
        power += 1
    if args.verbose: print(f"===> ret:\t\t", ret)
    return ret

def getReturn_t(rollout, snapshot, gamma):
    # Trajectory. A list of tuples [(s0, a0, s1, r1), (s1, a1, s2, r2), ..., (sH-1, aH-1, sH, rH)]
    ret = torch.tensor(0.)
    rollout = rollout[snapshot:]
    power = snapshot
    for i in range(len(rollout)):
        ret += (gamma**power)*rollout[i][3]
        power += 1
    if args.verbose: print(f"===> ret:\t\t", ret)
    return ret

def main(args):
    try:
        env.reset()
    except NameError:
        env = gym.make('CartPole-v1')

    # ___Set Seed___

    # ___Build the Models___
    if torch.cuda.is_available():
        print("cuda available")
        torch.cuda.set_device(args.device)
    else: 
        print("cuda NOT available")

    policy = NN()

    model_path = f"./models"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print(f"learning rate: {args.learning_rate}")
    # policy.parameters() = policy.fc.parameters() They are the same thing!
    if torch.cuda.is_available():
        policy.cuda()
        optimizer = torch.optim.Adam(list(policy.fc.parameters()), lr=args.learning_rate)
    
    optimizer = torch.optim.Adam(list(policy.fc.parameters()), lr=args.learning_rate)

    #___Load Previously Trained Model If Start Epoch > 0___ Skipping
    # if args.start_epoch > 0:
    #     load_opt_state(policy, os.path.join(args.model_path, model_name))
    
    print("training...")
    avgRewards = []
    if args.verbose: print(f"===> avgRewards:\t", avgRewards)
    for iter in range(MAX_ITERATIONS):
        totalRewards = []
        totalLoss = torch.tensor(0.)
        # print(f"Iteration {iter} of {MAX_ITERATIONS}")
        for e in range(args.episodes):
            if args.verbose: print(f"Episode {e} of {args.episodes}, iteration {iter} of {MAX_ITERATIONS}")
            t = 0
            rollout = [] # Trajectory. A list of tuples [(s0, a0, s1, r1), (s1, a1, s2, r2), ..., (sH-1, aH-1, sH, rH)]
            log_probs = [] # the distribution

            # get initial observation (agent's initial state s0)
            prev_state = env.reset() 

            terminate = False
            while not terminate:
                if not args.fast: env.render()
                # if args.verbose: print(obs)  
                # [ 0.04457668 -0.03367179 -0.01488337 -0.03034673  ]   obs -> state t
                # [ cart pos   cart vel    pole angle  pole ang vel ]

                # ask NN to make a prediction (softmax applied)
                out = policy.forward(prev_state)
                if args.verbose: print(f"===> out:\t\t", out)

                # takes discrete probabilities and create a probability distribution.
                distribution = torch.distributions.categorical.Categorical(out)

                # get a sample from distribution
                action = distribution.sample()

                action_prob = out[(action)]
                if args.verbose: print(f"===> action_prob:\t", action_prob)

                log_of_ap = torch.log(action_prob)
                if args.verbose: print(f"===> log_of_ap:\t\t", log_of_ap)
                
                # logp = distribution.log_prob(action)
                # if args.verbose: print(f"===> logp:\t\t", logp)

                # log_probs.append(logp)
                log_probs.append(log_of_ap)
                # if args.verbose: print(f"===> log_probs: \t", log_probs) # correct!

                # exit(0)
                
                if args.verbose: print(f"action taken: action")
                next_state, reward, done, info = env.step(int(action)) # obs -> state t+1
                
                # Store the states, actions, new states, and rewards
                snapshot = (prev_state, action, next_state, reward)

                rollout.append(snapshot)
                
                prev_state = next_state

                terminate = done
                t += 1

            if args.verbose: print(f"Episode finished after {t} timesteps.")

            const_return = torch.tensor(0.)
            if args.verbose: print(f"===> const_return:\t", const_return)
            returns = []
            if args.model == '1':
                # ___Calculate Constant Return___
                const_return = getReturn(rollout, GAMMA)
                if args.verbose: print(f"===> const_return:\t", const_return)
            else:
                # ___Calculate Return w.r.t. Time___
                for snapshot in range(len(rollout)):
                    ret = getReturn_t(rollout, snapshot, GAMMA)
                    if args.verbose: print(f"===> ret:\t\t", ret)
                    returns.append(ret)
                if args.verbose: print(f"===> returns:\t\t", returns)
            
            # Bam!

            # ___Calculate Total Rewards per Episode___
            totalReward = torch.tensor(0.)
            for i in range(len(rollout)):
                totalReward += rollout[i][3]
            
            totalRewards.append(totalReward)
            if args.verbose: print(f"===> totalRewards:\t", totalRewards)

            # Double Bam!
            
            loss = torch.tensor(0.)
            log_probs = torch.stack(log_probs)
            
            if len(returns) > 0:
                returns = torch.stack(returns)

            if args.model == '1':
                # const_return = to_var(torch.FloatTensor(const_return))
                if args.verbose: print(f"===> const_return:\t", const_return)
                if args.verbose: print(f"===> log_probs:\t", log_probs)
                loss = loss_f1(const_return, log_probs)
                if args.verbose: print(f"===> loss:\t\t", loss)
            else:
                # returns = to_var(torch.FloatTensor(returns))
                if args.verbose: print(f"===> returns:\t\t", returns)
                loss = loss_f2(args, returns, log_probs)
                if args.verbose: print(f"===> loss:\t\t", loss)
                
            totalLoss += loss
            
            # Triple Bam!
        if args.verbose: print(f"===> totalLoss:\t\t", totalLoss)

        avgLoss = totalLoss / args.episodes
        # avgLoss = torch.FloatTensor(avgLoss)
        if args.verbose: print(f"===> avgLoss:\t\t", avgLoss)
        # print("\n\n==================\n\n==================\n\n==================\n\n")

        sumTotalRewards = 0
        for tr in totalRewards:
            sumTotalRewards += tr
        avgReward = sumTotalRewards / args.episodes
        avgRewards.append(avgReward)

        if (iter+1) % 10 == 0:
            model_name = f"cartpole_q_{args.model}_episode_{args.episodes}_epoch_{iter+1}.pkl"
            save_state(policy, optimizer, os.path.join(model_path, model_name))
        
        optimizer.zero_grad()
        avgLoss.backward()
        optimizer.step()

        # print(f"\npolicy.state_dict:\n{policy.state_dict()}\n")
        # print("optimizer.step() called!")

        print(f"Iter: {iter},\tavgLoss: {avgLoss},\tavgReward: {avgReward}")

    avgRewardFileName = f"average_reward_model_{args.model}_episodes_{args.episodes}.csv"
    with open(avgRewardFileName, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(avgRewards)
    if args.verbose: print(f"\n...average rewards wrote to file: {avgRewardFileName}\n")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 4')
    parser.add_argument('-m', '--model', default='1', choices=['1', '2', '3'], \
        help='Enter 1 for question 1, part 1. Enter 2 for question 1 part 2. Enter 3 for question 1, part 3')
    parser.add_argument('-e', '--episodes', default=500, type=int, help='Number of episodes to execute.')
    parser.add_argument('-l', '--learning-rate', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('-f', '--fast', action='store_true', help='Set to disable live animation.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs.')

    args = parser.parse_args()

    main(args)