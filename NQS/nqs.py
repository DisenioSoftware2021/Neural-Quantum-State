#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


class NQS:
    """Represents a quantum state using 
    the restricted Boltzmann machine."""

    def __init__(self, n_hidden, n_dim, n_particles, sigma, d):

        self.n_dim = n_dim
        self.n_visible = n_dim * n_particles
        self.n_hidden = n_hidden
        self.n_particles = n_particles

        self.a = np.zeros(self.n_visible)
        self.b = np.zeros(n_hidden)

        self.x = np.zeros(self.n_visible)
        self.h = np.zeros(n_hidden)

        self.sigma = sigma
        self.sigma2 = self.sigma * self.sigma
        self.inverse_distances = np.zeros([n_particles, n_particles])
        self.sigmoidQ = np.zeros(n_hidden) #funcion sigmoide
        self.derivative_sigmoidQ = np.zeros(n_hidden)
        self.w = np.zeros(
            [self.n_visible, n_hidden]
        )  # self.sigma*(np.random.rand(self.n_visible, n_hidden)-0.5)
        self.derivative_psi = np.zeros(
            self.n_visible + self.n_hidden + (self.n_visible * self.n_hidden)
        )
        self.positive = 0.5
        self.calogeno = 0.0
        self.d        = 0.0

    def initi(self, inicial, seed):

        if inicial == "normal":

            sigma_init = 0.1
            for i in range(self.n_visible):
                self.a[i] = np.random.normal(sigma_init)
            for j in range(self.n_hidden):
                self.b[j] = np.random.normal(sigma_init)
            for i in range(self.n_visible):
                for j in range(self.n_hidden):
                    self.w[i, j] = np.random.normal(sigma_init)
        elif str(inicial) == "uniform":

            for i in range(self.n_visible):
                self.a[i] = np.random.rand() - 1
            for j in range(self.n_hidden):
                self.b[j] = np.random.rand() - 1
            for i in range(self.n_visible):
                for j in range(self.n_hidden):
                    self.w[i, j] = np.random.rand() - 1
        else:
            exit
        for i in range(self.n_visible):
            self.x[i] = np.random.rand() - 0.5

        return self.a, self.x, self.b, self.w

    def psi(self, x):
        # Wave function
        factor1 = np.dot(x - self.a, x - self.a)
        factor1 = np.exp(-factor1 / (2.0 * self.sigma2))
        factor2 = 1.0
        self.q1 = self.b + (np.matmul(x, self.w)) / self.sigma2
        for j in range(self.n_hidden):
            factor2 *= 1 + np.exp(self.q1[j])
        return np.sqrt(factor1 * factor2)

    def sigmoid(self, x):
        self.Q = self.b + (np.matmul(x, self.w)) / self.sigma2
        for j in range(self.n_hidden):
            self.sigmoidQ[j] = 1 / (1 + np.exp(-self.Q[j]))
        # print(self.sigmoidQ)
        return self.sigmoidQ

    def derivative_sigmoid_Q(self):
        for j in range(self.n_hidden):
            self.derivative_sigmoidQ[j] = np.exp(self.Q[j]) / (
                (1 + np.exp(self.Q[j])) * (1 + np.exp(self.Q[j]))
            )
        return self.derivative_sigmoidQ

    def laplacian(self, x):
        # The functions compute the laplacian of the wave function

        laplacian = 0.0

        for i in range(self.n_visible):
            derivative1_ln_psi = ((-x[i] + self.a[i]) / self.sigma2) + np.dot(
                self.w[i, :], self.sigmoidQ
            ) / self.sigma2
            sum_term = 0.0

            for j in range(self.n_hidden):
                sum_term += self.w[i, j] * self.w[i, j] * self.derivative_sigmoidQ[j]
            derivative2_ln_psi = -1.0 / self.sigma2 + sum_term / (self.sigma2 * self.sigma2)

            derivative1_ln_psi *= self.positive
            derivative2_ln_psi *= self.positive

            laplacian += -derivative1_ln_psi * derivative1_ln_psi - derivative2_ln_psi
        return laplacian

    def laplacian_alfa(self, x):
        #The function calculates 1 / psi * derivative_psi / dalpha_i,
        #  this is the derived wave function with respect 
        # to the network parameters a_i, b_j and w_ij

        for k in range(self.n_visible):
            self.derivative_psi[k] = (x[k] - self.a[k]) / self.sigma2
            # print(k)
        for k in range(self.n_visible, self.n_visible + self.n_hidden):
            self.derivative_psi[k] = self.sigmoidQ[k - self.n_visible]
            # print(k,self.sigmoidQ[k-self.n_visible],k-self.n_visible)
        k = self.n_visible + self.n_hidden
        for i in range(self.n_visible):
            for j in range(self.n_hidden):
                self.derivative_psi[k] = self.x[i] * self.sigmoidQ[j] / self.sigma2
                k = k + 1
        return self.derivative_psi * self.positive

    def inverse_distance(self, x):
        # Computes the coulombian interaction term
        p1 = 0

        # Loop over each particle
        for i1 in range(self.n_visible - self.n_dim, self.n_dim):
            p2 = p1 + 1
            # Loop over each particles that particle r hasn't been paired with
            for i2 in range(i1 + self.n_dim, self.n_visible, self.n_dim):
                # if i2>self.n_visible:
                #   break
                distance_particle = 0 #particle distance squared
                # Loop over dimensions
                for d in range(self.n_dim):
                    distance_particle += (x[i1 + d] - x[i2 + d]) * (x[i1 + d] - x[i2 + d])
                self.inverse_distances[p1, p2] = 1.0 / np.sqrt(distance_particle)
        return self.inverse_distances

    def calogero(self,x):
        #calculates the term of calogero model
        distance_particle=((x[0]-x[1])*(x[0]-x[1]))+self.d #particle distance squared
        g=2+2*self.d
        self.calogeno=g/(distance_particle)
        
        return self.calogeno
        