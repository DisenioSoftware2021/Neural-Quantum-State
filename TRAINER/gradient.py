#!/usr/bin/env python
# coding: utf-8

# In[3]:
import math

import attr
from attr import validators as vldts

import numpy as np


# In[4]:


@attr.s
class Gradient:
    # def __init__(self, learning_rate, gamma, n_hidden, n_visible):

    eta = attr.ib(validator=vldts.instance_of(float))  # learning_rate
    gamma = attr.ib(validator=vldts.instance_of(float))  # gamma
    n_hidden = attr.ib(validator=vldts.instance_of(int))  # nqs.n_hidden
    n_visible = attr.ib(validator=vldts.instance_of(int))  # nqs.n_visible

    epsilon_ = attr.ib(default=1e-8)  # 1e-8
    beta_1_ = attr.ib(default=0.9)  # 0.9
    beta_2_ = attr.ib(default=0.99)  # 0.99
    # import ipdb; ipdb.set_trace()
    n_parameter_ = attr.ib(init=False, repr=False)  # np.zeros(n_parameter_)
    shift = attr.ib(init=False, repr=False)  # np.zeros(n_parameter_)
    # m =  attr.ib(init=False, repr=False) # np.zeros(n_parameter_)
    # s =  attr.ib(init=False, repr=False) # np.zeros(n_parameter_)
    # squared =  attr.ib(init=False, repr=False) # np.zeros(n_parameter_)
    prev_m = attr.ib(init=False, repr=False)  # np.zeros(n_parameter_)
    prev_s = attr.ib(init=False, repr=False)  # np.zeros(n_parameter_)
    prev_shift = attr.ib(init=False, repr=False)  # np.zeros(n_parameter_)

    @n_parameter_.default
    def n_parameter_default(self):
        return (
            self.n_hidden + self.n_visible + (self.n_visible * self.n_hidden)
        )

    @shift.default
    def shift_default(self):
        return np.zeros(self.n_parameter_)

    @prev_m.default
    def prev_m_default(self):
        return np.zeros(self.n_parameter_)

    @prev_s.default
    def prev_s_default(self):
        return np.zeros(self.n_parameter_)

    @prev_shift.default
    def prev_shift_default(self):
        return np.zeros(self.n_parameter_)

    def parameter_shift(self, gradient):
        # The function computes the shift with which the network
        # parameters should be updated according to
        # the simple gradient descent algoirthm.
        # the parameters are the weights and bias
        self.shift = self.gamma * self.prev_shift + self.eta * gradient
        self.prev_shift = self.shift

        return -self.shift

    def adam(self, gradient, iteration):
        squared = np.zeros(self.n_parameter_)
        m = np.zeros(self.n_parameter_)
        s = np.zeros(self.n_parameter_)

        m = self.beta_1_ * self.prev_m + (1 - self.beta_1_) * gradient
        for i in range(0, self.n_parameter_):
            squared[i] = gradient[i] * gradient[i]

        s = self.beta_2_ * self.prev_s + (1 - self.beta_2_) * squared
        self.prev_m = m
        self.prev_s = s

        m = m / (1 - math.pow(self.beta_1_, iteration))
        s = s / (1 - math.pow(self.beta_2_, iteration))
        # print(prev_m[1],)
        for i in range(0, self.n_parameter_):
            self.shift[i] = m[i] / (np.sqrt(s[i]) + self.epsilon_)
        return -self.eta * self.shift

    # def setup(self):
    # self.prev_m = np.zeros(self.n_parameter_)
    # self.prev_s = np.zeros(self.n_parameter_)
    # self.prev_shift = np.zeros(self.n_parameter_)


# In[ ]:
