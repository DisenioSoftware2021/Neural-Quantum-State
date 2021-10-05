#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


class NQS:
    """Represents a quantum state using 
    the restricted Boltzmann machine."""

    def __init__(self, n_hidden, n_dim, n_particles, sig):

        self.n_dim = n_dim
        self.n_visible = n_dim * n_particles
        self.n_hidden = n_hidden
        self.n_particles = n_particles

        self.a = np.zeros(self.n_visible)
        self.b = np.zeros(n_hidden)

        self.x = np.zeros(self.n_visible)
        self.h = np.zeros(n_hidden)

        self.sig = sig
        self.sig2 = self.sig * self.sig
        self.m_inverse_distances = np.zeros([n_particles, n_particles])
        self.sigmoidQ = np.zeros(n_hidden)
        self.m_der_sigmoid = np.zeros(n_hidden)
        self.w = np.zeros(
            [self.n_visible, n_hidden]
        )  # self.sig*(np.random.rand(self.n_visible, n_hidden)-0.5)
        self.dPsi = np.zeros(
            self.n_visible + self.n_hidden + (self.n_visible * self.n_hidden)
        )
        self.positive = 0.5
        self.calogeno = 0.0

    def initi(self, inicial, seed):

        if inicial == "normal":

            sigma_init = 0.1
            for i in range(0, self.n_visible):
                self.a[i] = np.random.normal(0, sigma_init)
            for j in range(0, self.n_hidden):
                self.b[j] = np.random.normal(0, sigma_init)
            for i in range(0, self.n_visible):
                for j in range(0, self.n_hidden):
                    self.w[i, j] = np.random.normal(0, sigma_init)
        elif str(inicial) == "uniforme":

            for i in range(0, self.n_visible):
                self.a[i] = np.random.rand() - 1
            for j in range(0, self.n_hidden):
                self.b[j] = np.random.rand() - 1
            for i in range(0, self.n_visible):
                for j in range(0, self.n_hidden):
                    self.w[i, j] = np.random.rand() - 1
        else:
            exit
        for i in range(0, self.n_visible):
            self.x[i] = np.random.rand() - 0.5

        return self.a, self.x, self.b, self.w

    def psi(self, x):
        # Funcion de onda
        factor1 = np.dot(x - self.a, x - self.a)
        factor1 = np.exp(-factor1 / (2.0 * self.sig2))
        factor2 = 1.0
        self.Q1 = self.b + (np.matmul(x, self.w)) / self.sig2
        for j in range(0, self.n_hidden):
            factor2 *= 1 + np.exp(self.Q1[j])
        return np.sqrt(factor1 * factor2)

    def sigmoid(self, x):
        self.Q = self.b + (np.matmul(x, self.w)) / self.sig2
        for j in range(0, self.n_hidden):
            self.sigmoidQ[j] = 1 / (1 + np.exp(-self.Q[j]))
        # print(self.sigmoidQ)
        return self.sigmoidQ

    def mdersigmoidQ(self):
        for j in range(0, self.n_hidden):
            self.m_der_sigmoid[j] = np.exp(self.Q[j]) / (
                (1 + np.exp(self.Q[j])) * (1 + np.exp(self.Q[j]))
            )
        return self.m_der_sigmoid

    def Laplacian(self, x):
        # La funcion calcula el laplaciano de la funcion de onda

        laplacian = 0.0

        for i in range(0, self.n_visible):
            d1lnPsi = ((-x[i] + self.a[i]) / self.sig2) + np.dot(
                self.w[i, :], self.sigmoidQ
            ) / self.sig2
            sum_term = 0.0

            for j in range(0, self.n_hidden):
                sum_term += self.w[i, j] * self.w[i, j] * self.m_der_sigmoid[j]
            d2lnPsi = -1.0 / self.sig2 + sum_term / (self.sig2 * self.sig2)

            d1lnPsi *= self.positive
            d2lnPsi *= self.positive

            laplacian += -d1lnPsi * d1lnPsi - d2lnPsi
        return laplacian

    def laplacianalfa(self, x):
        #The function calculates 1 / psi * dPsi / dalpha_i,
        #  this is the derived wave function with respect 
        # to the network parameters a_i, b_j and w_ij

        for k in range(0, self.n_visible):
            self.dPsi[k] = (x[k] - self.a[k]) / self.sig2
            # print(k)
        for k in range(self.n_visible, self.n_visible + self.n_hidden):
            self.dPsi[k] = self.sigmoidQ[k - self.n_visible]
            # print(k,self.sigmoidQ[k-self.n_visible],k-self.n_visible)
        k = self.n_visible + self.n_hidden
        for i in range(0, self.n_visible):
            for j in range(0, self.n_hidden):
                self.dPsi[k] = self.x[i] * self.sigmoidQ[j] / self.sig2
                k = k + 1
        return self.dPsi * self.positive

    def Inversedistance(self, x):
        # Computes the coulombian interaction term
        p1 = 0

        # Loop over each particle
        for i1 in range(0, self.n_visible - self.n_dim, self.n_dim):
            p2 = p1 + 1
            # Loop over each particles that particle r hasn't been paired with
            for i2 in range(i1 + self.n_dim, self.n_visible, self.n_dim):
                # if i2>self.n_visible:
                #   break
                dSqd = 0
                # Loop over dimensions
                for d in range(0, self.n_dim):
                    dSqd += (x[i1 + d] - x[i2 + d]) * (x[i1 + d] - x[i2 + d])
                self.m_inverse_distances[p1, p2] = 1.0 / np.sqrt(dSqd)
        return self.m_inverse_distances

    def calogero(self, x):
        #   calculates the term of the calogero model
        dsqd = ((x[0] - x[1]) * (x[0] - x[1])) + 0.05

        self.calogeno = 1.0 / (dsqd)

        return self.calogeno
