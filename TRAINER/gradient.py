#!/usr/bin/env python
# coding: utf-8

# In[3]:

import math

import numpy as np
from attr import validators as vldts



# In[4]:

@attr.s
class Gradient:
    #def __init__(self, learning_rate, gamma, n_hidden, n_visible):
    
    gamma = attr.ib(validator=vldts.instance_of(float)) # gamma
    eta = attr.ib(validator=vldts.instance_of(float)) # learning_rate    
    n_hidden = attr.ib(validator=vldts.instance_of(int)) # nqs.n_hidden
    n_visible = attr.ib(validator=vldts.instance_of(int)) # nqs.n_visible
    
    bias_shift_ =  attr.ib(init=False, repr=False) # np.zeros(parameter)
    epsilon_ = attr.ib(default=1e-8) # 1e-8
    beta_1_ = attr.ib(default=0.9) # 0.9
    beta_2_ = attr.ib(default=0.99) # 0.99
    bias_m_ =  attr.ib(init=False, repr=False) # np.zeros(parameter)
    bias_s_ =  attr.ib(init=False, repr=False) # np.zeros(parameter)
    bias_squared_ =  attr.ib(init=False, repr=False) # np.zeros(parameter)
        
    @n_parameter_.default
    def _parameter__default(self):
        return (
            self.n_hidden + self.n_visible + (self.n_visible * self.n_hidden)
        ) #dimension of the parameters 

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
