#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# In[ ]:


class NQSpositive:
    def __init__(self, nqs):
        self.nqs = nqs

        self.n_hidden = nqs.n_hidden
        self.n_visible = nqs.n_visible
        self.a = nqs.a
        self.w = nqs.w
        self.x = nqs.x
        self.sigma = nqs.sigma
        self.h = nqs.h

        self.normal_distance = np.zeros([])

    def prob_h_given_x(self):
        # The function returns the conditional probability of h_j=1
        # given the value of x
        self.sigmoid = self.nqs.sigmoid

        return self.sigmoid

    def prob_x_given_h(self, i):
        # The function returns the conditional probability distribution
        # of each value of x given the value of h
        x_mean = self.nqs.a[i] + np.dot(self.nqs.w[i, :], self.h)
        return np.random.normal(x_mean, self.sigma)

    def gibbs(self, seed_2):
        # The function samples a new configuration of positions
        # according to the Gibbs method.
        # It gives the value of new hidden units according to the
        # sigmoid function comparing this with a random value
        # between 0 and 1

        self.sigmoid = self.nqs.sigmoidQ
        for j in range(0, self.n_hidden):
            if self.sigmoid[j] > np.random.rand():
                self.h[j] = 1.0
            else:
                self.h[j] = 0.0

        for i in range(0, self.n_visible):
            self.x[i] = NQSpositive.prob_x_given_h(self, i)
        self.sigmoid = self.nqs.sigmoid(self.x)
        return self.x, self.h
