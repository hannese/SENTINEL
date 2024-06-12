#!/usr/bin/env python
from __future__ import print_function

import argparse
import random
from random import choice
import numpy as np
from collections import deque
import time
import math
import distributions

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Convolution1D, Dense, Flatten, Dropout, Input, \
    AveragePooling2D, Lambda, Merge, Activation, Embedding, Lambda
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K
from keras.utils import np_utils

import itertools as it
from time import sleep
import tensorflow as tf

import gym
from gym import register
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec

tableau10 = np.array([(31,119,180), (255,127,14),(44,160,44), (214,39,40), (148,103,189),
                      (140,86,75), (227,119,194),(127,127,127),(188,189,34),(23,190,207)]) * 1. / 255

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['figure.figsize'] = 10, 8
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def preprocess_img(img): # downsample and grayscale it
    return np.mean(img[::2, ::2], axis=2).astype(np.uint8)

class Networks(object):

    @staticmethod
    def cnn_value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        """Model Value Distribution
        With States as inputs and output Probability Distributions for all Actions
        """

        state_input = Input(shape=(input_shape))
        normalized = Lambda(lambda x: x / 255.0)(state_input)
        cnn_feature = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu')(normalized)
        cnn_feature = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')(cnn_feature)
        cnn_feature = Convolution2D(64, 3, 3, activation='relu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, activation='relu')(cnn_feature)
        dropout = Dropout(0.5)(cnn_feature)
        final_layer = cnn_feature

        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(final_layer))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model

    '''
    One common estimator layer
    One common action layer
    '''
    @staticmethod
    def ensemble_value_distribution_network(input_shape, num_atoms, action_size, ensemble_size, shared_weights=True):
        models = []

        state_input = Input(shape=(input_shape,), dtype='float32')

        if shared_weights:
            common_ensemble_layer = Dense(32, activation='relu')(state_input)
            for m in range(ensemble_size):
                ensemble_models = []
                dropout1 = Dropout(rate=0.01)(common_ensemble_layer)
                hidden_layer1 = Dense(32, activation='relu')(dropout1)
                for a in range(action_size):
                    dropout2 = Dropout(rate=0.01)(hidden_layer1)
                    hidden_layer2 = Dense(128, activation='relu')(dropout2)
                    #hidden_layer2 = Dense(32, activation='relu')(dropout2)
                    distribution = Dense(num_atoms, activation='softmax')(hidden_layer2)
                    model = Model(inputs=state_input, output=distribution)
                    #adam = Adam(lr=0.00025, epsilon=0.01 / 32)  # from CDQN paper
                    adam = Adam(lr=0.01, epsilon=0.01 / 32)  # from CDQN paper
                    model.compile(loss='categorical_crossentropy', optimizer=adam)
                    ensemble_models.append(model)
                models.append(ensemble_models)
        else:
            for m in range(ensemble_size):
                ensemble_models = []
                for a in range(action_size):
                    hidden_layer1 = Dense(256, activation='relu')(state_input)
                    dropout1 = Dropout(rate=0.5)(hidden_layer1)
                    hidden_layer2 = Dense(256, activation='relu')(dropout1)
                    dropout2 = Dropout(rate=0.5)(hidden_layer2)
                    hidden_layer3 = Dense(256, activation='relu')(dropout2)
                    distribution = Dense(num_atoms, activation='softmax')(hidden_layer3)
                    model = Model(inputs=state_input, output=distribution)
                    adam = Adam(lr=0.01/32) 
                    model.compile(loss='categorical_crossentropy', optimizer=adam)
                    ensemble_models.append(model)
                models.append(ensemble_models)
        return models

    @staticmethod
    def factored_value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        """Model Value Distribution
        With States as inputs and output Probability Distributions for all Actions
        """

        state_input = Input(shape=(input_shape,))

        distribution_list = []

        for i in range(action_size):
            hidden_feature = Dense(input_shape, activation='relu')(state_input)
            final_dense = Dense(2056, activation='relu')(hidden_feature)
            distribution_list.append(Dense(num_atoms, activation='softmax')(final_dense))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model

class SentinelAgent:

    # helper functions for plotting etc

    # marginal over estimators
    # ensemble x action x 1 x atoms
    def _get_marginal(self, res):
        _, _, z = res
        return np.sum(np.array(z), axis=0) / self.ensemble_size

    # expectation for each action
    def _EX(self, res):
        _, _, z = res

        if self.KL_lambda is not None and self.ensemble_size > 1:
            # get marginal
            ptheta_z = self._get_marginal(res)
            # calculate weights
            weights = np.zeros((self.ensemble_size, self.action_size))
            for m in range(self.ensemble_size):
                inner = ptheta_z * np.log(z[m, :, :] / ptheta_z)
                np.nan_to_num(inner) # handle if marginal is zero 0 * log x / 0 -> 0
                weights[m] = np.exp(-self.KL_lambda * -np.sum(inner, axis=1))
            weighted_q = np.sum(
                np.sum(np.multiply(z, np.array(self.z)), axis=2) * weights / np.sum(weights, axis=0),
                                axis=0)
            return weighted_q

        else:
            q = np.mean(np.sum(np.multiply(z, np.array(self.z)), axis=2), axis=0)
            return q

    # average aleatory
    def _ALEATORY(self, res):

        _, _, z = res
        ACVaR = np.zeros((self.action_size))

        for m in range(self.ensemble_size):

            alpha = self.utility_params[0]

            for a in range(self.action_size):
                CDF = np.cumsum(z[m][a])
                idx = np.where(CDF >= alpha)[0][0]
                C = np.dot(z[m][a].flatten()[:idx], self.z[:idx])
                if idx == 0:
                    C += self.z[idx] * alpha
                else:
                    C += self.z[idx] * (alpha - CDF[idx - 1])
                ACVaR[a] += C / self.ensemble_size

        return ACVaR

    # epistemic
    def _EPISTEMIC(self, res):

        _, _, z = res
        q = np.sum(np.multiply(z, np.array(self.z)), axis=2) # ensemble x actions x 1 x atoms
        if self.KL_lambda is not None and self.ensemble_size > 1:
            # get marginal
            ptheta_z = self._get_marginal(res)
            # calculate weights
            weights = np.zeros((self.ensemble_size, self.action_size))
            for m in range(self.ensemble_size):
                inner = ptheta_z * np.log(z[m, :, :] / ptheta_z)
                np.nan_to_num(inner) # handle if marginal is zero 0 * log x / 0 -> 0
                weights[m] = np.exp(-self.KL_lambda * -np.sum(inner, axis=1))

            weighted_qs = zip(q, weights / np.sum(weights, axis=0))

        else:
            weighted_qs = zip(q, np.ones(self.ensemble_size) / self.ensemble_size)

        sorted_qs = np.sort(np.array(list(weighted_qs)), axis=0)
        CDF = np.cumsum(sorted_qs[:, 1, :], axis=0)
        CA = np.zeros((self.action_size))
        for a in range(self.action_size):
            idx = np.where(CDF[:, a] >= self.utility_params[1])[0][0]
            CA[a] = np.dot(sorted_qs[:, 0, a], sorted_qs[:, 1, a])
            if idx == 0:
                CA[a] += sorted_qs[idx, 1, a] * self.utility_params[1]
            else:
                CA[a] += sorted_qs[idx, 1, a] * (self.utility_params[1] - CDF[idx - 1, a])
        return CA

    def _COMPOSITE(self, res):

        _, _, z = res

        # do cvar
        aleatory_dist = np.zeros((self.ensemble_size, self.action_size))
        for m in range(self.ensemble_size):
            for a in range(self.action_size):
                if self.utility == "composite":
                    aleatory = distributions.cvar(z[m][a], self.z, self.utility_params[0])
                elif self.utility == "wang":
                    aleatory = distributions.wang(z[m][a], self.z, self.utility_params[0])
                elif self.utility == "evar":
                    aleatory = distributions.evar(z[m][a], self.z, self.utility_params[0])
                elif self.utility == "stdcvar":
                    aleatory = distributions.std(z[m][a], self.z, self.utility_params[0])
                elif self.utility == "cvarstd":
                    aleatory = distributions.cvar(z[m][a], self.z, self.utility_params[0])
                else:
                    raise RuntimeError("unexpected utility")
                aleatory_dist[m][a] = aleatory

        if self.KL_lambda is not None and self.ensemble_size > 1:
            # get marginal
            ptheta_z = self._get_marginal(res)
            # calculate weights
            weights = np.zeros((self.ensemble_size, self.action_size))
            for m in range(self.ensemble_size):
                inner = ptheta_z * np.log(z[m, :, :] / ptheta_z)
                np.nan_to_num(inner) # handle if marginal is zero 0 * log x / 0 -> 0
                weights[m] = np.exp(-self.KL_lambda * -np.sum(inner, axis=1))

            weighted_qs = zip(aleatory_dist, weights / np.sum(weights, axis=0))

        else:
            weighted_qs = zip(aleatory_dist, np.ones(self.ensemble_size) / self.ensemble_size)

        weighted_qs = np.array(list(weighted_qs))
        masses = weighted_qs[:, 1, :]
        boundary = weighted_qs[:, 0, :]
        idx = np.argsort(boundary, axis=0)
        boundary = np.take_along_axis(boundary, idx, axis=0)
        masses = np.take_along_axis(masses, idx, axis=0)
        CA = np.zeros((self.action_size))
        for a in range(self.action_size):
            if self.utility == "composite":
                epistemic = distributions.cvar(masses[:, a], boundary[:, a], self.utility_params[1])
            elif self.utility == "wang":
                epistemic = distributions.wang(masses[:, a], boundary[:, a], self.utility_params[1])
            elif self.utility == "evar":
                epistemic = distributions.evar(weighted_qs[:, 1, a], weighted_qs[:, 0, a], self.utility_params[1])
                #epistemic = distributions.evar(masses[:, a], boundary[:, a], 0.05)
            elif self.utility == "stdcvar":
                epistemic = distributions.cvar(masses[:, a], boundary[:, a], self.utility_params[1])
            elif self.utility == "cvarstd":
                epistemic = distributions.std(masses[:, a], boundary[:, a], self.utility_params[1])
            else:
                raise RuntimeError("unexpected utility")
            CA[a] = epistemic

        return CA

    def __init__(self, state_size, action_size, num_atoms, ensemble_size, max_steps, gamma,
                 utility="linear", utility_params=(None, None), batch_size=32, max_samples=1000,
                 learning_rate=0.00025, grad_step=0.01/32, epochs=1, KL_lambda=1.0,
                 v_min=-10, v_max=10, max_memory=100000):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size

        # these are hyper parameters for the DQN
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.05
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.learning_rate = learning_rate # from CDQN
        self.grad_step = grad_step # from CDQN
        self.gamma = gamma
        self.epochs = epochs
        self.observe = self.max_samples
        self.explore = math.floor(max_steps * 0.25)

        self.KL_lambda = KL_lambda # 1% weight to completely wrong model
        #self.KL_lambda = None

        # 0 1 3 6 10 ...
        self.updates = []
        tmp = np.arange(max_steps)

        for i in range(1, max_steps):
            tmp2 = np.sum(tmp[0:i])
            if tmp2 > max_steps:
                break
            self.updates.append(tmp2)

        print("Total updates: " + str(len(self.updates) * self.ensemble_size))
        self.utility = utility
        self.utility_params = utility_params

        # Initialize Atoms
        self.num_atoms = num_atoms # 51 for C51
        self.v_max = v_max
        self.v_min = v_min
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Create replay memory using deque
        self.memory = deque()
        self.memory_weights = [None] * ensemble_size
        for m in range(ensemble_size):
            self.memory_weights[m] = deque()
        self.max_memory = max_memory # number of previous transitions to remember

        # Models for value distribution
        self.models = [None] * ensemble_size
        self.target_models = [None] * ensemble_size

        # Performance Statistics
        self.returns = []

    def _plot_model(self, model):
        """
        Attempts to plot architecture, requires pydot, pydotplus and graphviz.
        Requires brew install graphviz for Mac
        Requires specific .msi for Windows
        Works fine on *nix
        """
        tf.keras.utils.plot_model(model, to_file='model.pdf', show_shapes=True)

    def update_target_model(self, model):
        """
        After some time interval update the target model to be same with model
        """
        for a in range(self.action_size):
            self.target_models[model][a].set_weights(self.models[model][a].get_weights())

    def get_action(self, state, debug=False):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            if self.utility == "linear":
                action_idx = self.get_optimal_action(state, debug)
            elif self.utility == "aleatory":
                action_idx = self.get_aleatory_action(state, debug)
            elif self.utility == "epistemic":
                action_idx = self.get_epistemic_action(state, debug)
            elif self.utility in ["composite", "wang", "evar", "stdcvar", "cvarstd"]:
                action_idx = self.get_composite_action(state, debug)
            else:
                raise NotImplementedError("Unknown decision maker")

        return action_idx

    def get_optimal_action(self, state, debug=False):
        """Get optimal utility action for a state
        """
        s_ = state[np.newaxis, :]
        z = np.zeros((self.ensemble_size, self.action_size, self.num_atoms))
        for m in range(self.ensemble_size):
            for a in range(self.action_size):
                z[m, a] = self.models[m][a].predict(s_).flatten()
        action_idx = np.argmax(self._EX((None, None, z)))

        if debug:
            return action_idx, self.z, z
        else:
            return action_idx

    def get_aleatory_action(self, state, debug=False):
        """Get optimal aleatoric utility action for a state
        """
        s_ = state[np.newaxis, :]
        z = np.zeros((self.ensemble_size, self.action_size, self.num_atoms))
        for m in range(self.ensemble_size):
            for a in range(self.action_size):
                z[m, a] = self.models[m][a].predict(s_).flatten()
        action_idx = np.argmax(self._ALEATORY((None, None, z)))

        if debug:
            return action_idx, self.z, z
        else:
            return action_idx

    def get_epistemic_action(self, state, debug=False):
        s_ = state[np.newaxis, :]
        z = np.zeros((self.ensemble_size, self.action_size, self.num_atoms))
        for m in range(self.ensemble_size):
            for a in range(self.action_size):
                z[m, a] = self.models[m][a].predict(s_).flatten()
        action_idx = np.argmax(self._EPISTEMIC((None, None, z)))

        if debug:
            return action_idx, self.z, z
        else:
            return action_idx

    def get_composite_action(self, state, debug=False):
        s_ = state[np.newaxis, :]
        z = np.zeros((self.ensemble_size, self.action_size, self.num_atoms))
        for m in range(self.ensemble_size):
            for a in range(self.action_size):
                z[m, a] = self.models[m][a].predict(s_).flatten()
        action_idx = np.argmax(self._COMPOSITE((None, None, z)))

        if debug:
            return action_idx, self.z, z
        else:
            return action_idx

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))

        for m in range(self.ensemble_size):
            #self.memory_weights[m].append(random.random())
            self.memory_weights[m].append(np.random.binomial(1, 1./3, 1)[0])

        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()
            for m in range(self.ensemble_size):
                self.memory_weights[m].popleft()

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self, model):
        for m in range(self.ensemble_size):
            for a in range(self.action_size):
                self.models[m][a].training = True

        num_samples = min(self.max_samples, len(self.memory))
        p = self.memory_weights[model] / np.sum(self.memory_weights[model])
        tmp_memory = np.array(self.memory)

        if np.count_nonzero(p) < num_samples:
            replay_samples = tmp_memory[np.random.choice(len(self.memory), size=num_samples,
                                                         p=p, replace=True)]
        else:
            replay_samples = tmp_memory[np.random.choice(len(self.memory), size=num_samples,
                                                         p=p, replace=False)]
        if True: # not a tuple
            state_inputs = np.zeros(((num_samples,) + (self.state_size,)))
            next_states = np.zeros(((num_samples,) + (self.state_size,)))
        else:
            state_inputs = np.zeros(((num_samples,) + (self.state_size)))
            next_states = np.zeros(((num_samples,) + (self.state_size)))

        m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(self.action_size)]
        action, reward, done = [], [], []

        for i in range(num_samples):
            state_inputs[i] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            next_states[i] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        z_ = [self.target_models[model][a].predict(next_states) for a in range(self.action_size)]

        # Get Optimal Actions for the next states (from distribution z)
        if self.utility == "linear":
            optimal_action_idxs = [self.get_optimal_action(next_s) for next_s in next_states]
        elif self.utility == "aleatory":
            optimal_action_idxs = [self.get_aleatory_action(next_s) for next_s in next_states]
        elif self.utility == "epistemic":
            optimal_action_idxs = [self.get_epistemic_action(next_s) for next_s in next_states]
        elif self.utility in ["composite", "wang", "evar", "stdcvar", "cvarstd"]:
            optimal_action_idxs = [self.get_composite_action(next_s) for next_s in next_states]
        else:
            raise NotImplementedError("Unknown decision maker")

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            if done[i]: # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                    m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

        accumulated_loss = 0
        for a in range(self.action_size):
            mask = np.intersect1d(np.where((np.array(action) == a)), np.nonzero(np.sum(m_prob[a], axis=1)))
            if len(mask) < 2:
                continue
            loss = self.models[model][a].fit(state_inputs[mask], m_prob[a][mask], batch_size=self.batch_size,
                                             epochs=self.epochs, verbose=0)
            #print("loss: ", loss.history['loss'][-1], loss.history['loss'][-1]-loss.history['loss'][0])
            accumulated_loss += loss.history['loss'][-1]

        for m in range(self.ensemble_size):
            for a in range(self.action_size):
                self.models[m][a].training = False

        return accumulated_loss

    # another training scheme
    def train_ensemble(self):

        for m in range(self.ensemble_size):
            self.models[m].training = True

        for m in range(self.ensemble_size):
            self.models[m].training = False

    # load the saved model
    def load_model(self, name):
        for m in range(len(self.models)):
            for a in range(self.action_size):
                self.models[m][a].load_weights(name+"."+str(m)+"_"+str(a)+".npy")

    # save the model which is under training
    def save_model(self, name):
        for m in range(len(self.models)):
            for a in range(self.action_size):
                self.models[m][a].save_weights(name+"."+str(m)+"_"+str(a)+".npy")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--env', type=str, default="CartPole-v0")
    parser.add_argument('--max_steps', type=int, default=100000, help="Steps in environment")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--ensembles', type=int, default=4, help="Number of ensembles")
    parser.add_argument('--utility', type=str, default="linear", help="linear / aleatory / epistemic / composite / wang / evar")


    args = parser.parse_args()
    _env = args.env
    env = None
    
    max_steps = args.max_steps
    utility = args.utility
    ensembles = args.ensembles
    gamma = args.gamma
    
    if _env == "CartPole-v0":
        env = gym.make("CartPole-v0")
        state_size = 4
        action_size = 2
        v_min = 0
        v_max = (1-gamma ** 200) / (1-gamma) # ~= 86.6
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    #
    num_experiments = 5
    num_atoms = 51
    alpha1 = 0.99
    alpha2 = 0.99
   
    timesteps = []
    rolling_returns = []
    print("Starting experiment: ", env)
    for i in range(num_experiments):
        if _env == "CartPole-v0":
            env = gym.make("CartPole-v0")
            
        agent = SentinelAgent(state_size, action_size, num_atoms, ensembles, max_steps, gamma, utility=utility,
                          utility_params=(alpha1, alpha2), v_min=v_min, v_max=v_max)
        
        agent.models = Networks.ensemble_value_distribution_network(state_size, num_atoms, action_size, ensembles)
        agent.target_models = Networks.ensemble_value_distribution_network(state_size, num_atoms, action_size, ensembles)
        
        print(agent.models[0][0].summary())
        
        episodic_reward = 0
        rolling_episodic = 0
        s = env.reset()
        for t in range(max_steps): 
            a = agent.get_action(s)
            s_, r, done, info = env.step(a)
            episodic_reward += r
            s_ = s_.flatten()
            agent.replay_memory(s, a, r, s_, done, t)
            
            if done:
                rolling_episodic = 0.98 * rolling_episodic + 0.02 * episodic_reward
                timesteps.append(t)
                rolling_returns.append(rolling_episodic)
                print("timestep: "+str(t) + " return: " + "{:.3f}".format(episodic_reward) + " rolling return: "+"{:.3f}".format(rolling_episodic) + " epsilon: "+"{:.3f}".format(agent.epsilon))
                episodic_reward = 0
                s = env.reset().flatten()
            else:
                s = s_
                
            if t > agent.observe and t in agent.updates[0::1]:
                previous_loss = 0
                for m in range(ensembles):
                    previous_loss += agent.train_replay(m)


            if t > agent.observe and t in agent.updates[0::5]:  # update target every 5th time
                for m in range(ensembles):
                    agent.update_target_model(m)
        K.clear_session()
        np.save("results/sentinel_"+str(utility)+"_"+str(ensembles)+"_timesteps"+str(i)+".npy", np.array(timesteps))   
        np.save("results/sentinel_"+str(utility)+"_"+str(ensembles)+"_rolling_returns"+str(i)+".npy", np.array(rolling_returns))  

    