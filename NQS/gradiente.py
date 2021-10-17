#!/usr/bin/env python
# coding: utf-8

# In[3]:

import math

import numpy as np

# In[4]:

 
class Gradiente:
    def __init__(self, learning_rate, gamma, nqs):
        self.nqs = nqs
        self.n_hidden = nqs.n_hidden
        self.n_visible = nqs.n_visible
        self.eta = learning_rate
        self.gamma = gamma
        self.parameter = (
            self.n_hidden + self.n_visible + (self.n_visible * self.n_hidden)
        ) #dimension of the parameters 
        self.shift = np.zeros(self.parameter)

        self.epsilon = 1e-8
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.m = np.zeros(self.parameter)
        self.s = np.zeros(self.parameter)
        self.squared = np.zeros(self.parameter)

    def parameter_shift(self, gradient):
        # The function computes the shift with which the network
        # parameters should be updated according to
        # the simple gradient descent algoirthm.
        #the parameters are the weights and bias
        self.shift = self.gamma * self.prev_shift + self.eta * gradient
        self.prev_shift = self.shift

        return -self.shift

    def adam(self, gradient, iteration):
        self.m = self.beta_1 * self.prev_m + (1 - self.beta_1) * gradient
        for i in range(0, self.parameter):
            self.squared[i] = gradient[i] * gradient[i]

        self.s = self.beta_2 * self.prev_s + (1 - self.beta_2) * self.squared
        self.prev_m = self.m
        self.prev_s = self.s

        self.m = self.m / (1 - math.pow(self.beta_1, iteration))
        self.s = self.s / (1 - math.pow(self.beta_2, iteration))
        # print(prev_m[1],)
        for i in range(0, self.parameter):
            self.shift[i] = self.m[i] / (np.sqrt(self.s[i]) + self.epsilon)
        return -self.eta * self.shift

    def setup(self):
        self.prev_m = np.zeros(self.parameter)
        self.prev_s = np.zeros(self.parameter)
        self.prev_shift = np.zeros(self.parameter)      


# In[ ]:
