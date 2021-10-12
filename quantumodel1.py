#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

import NQS

import hamiltonian as hm

import probcondicional1 as cond


# In[3]:


class QuantumModel:
    def __init__(self, nqs, ham, condi):

        self.nqs = nqs
        self.ham = ham
        self.condi = condi

        self.n_dim = nqs.n_dim
        self.n_hidden = nqs.n_hidden
        self.n_visible = nqs.n_visible
        self.a = nqs.a
        self.w = nqs.w
        self.b = nqs.b
        self.x = condi.x

        self.parameters = (
            self.n_visible + self.n_hidden + self.n_visible * self.n_hidden
        )
        self.sigma = nqs.sigma

        self.energy_gradient = np.zeros(self.parameters)

    def zzero(self):
        # The function initializes variables that I am going to add from zero.

        self.local_energy = 0
        self.local_energy_squared = 0
        self.accept_count = 0

        self.derivate_psi1 = np.zeros(self.parameters)
        self.energy_derivative_psi = np.zeros(self.parameters)

    def set_up_sampling(self, x):
        # The function sets up the model for a Monte Carlo simulation.

        self.sigmoid = self.nqs.sigmoid(x)

        self.psi = self.nqs.psi(x)
        self.loc_energy = self.ham.local_energy(self.nqs)

        self.lap = self.nqs.laplacian_alfa(x)

        return self.psi, self.loc_energy, self.lap

    def accumulator(self, x):

        self.lap = self.nqs.laplacian_alfa(x)
        self.loc_energy = self.ham.local_energy(self.nqs)

        self.local_energy += self.loc_energy
        self.local_energy_squared += self.loc_energy * self.loc_energy

        self.derivate_psi1 += self.lap
        self.energy_derivative_psi += self.lap * self.loc_energy

        return (
            self.local_energy,
            self.local_energy_squared,
            self.derivate_psi1,
            self.energy_derivative_psi,
        )

    def average_value(self, sample_number):
        self.local_energy = self.local_energy / sample_number
        self.local_energy_squared = self.local_energy_squared / sample_number
        # self.acceptcount =self.acceptcount/nrosamples
        self.derivate_psi1 = self.derivate_psi1 / sample_number
        self.energy_derivative_psi = self.energy_derivative_psi / sample_number

        self.variance = (
            self.local_energy_squared - self.local_energy * self.local_energy
        ) / sample_number

        self.loc_energy_gradient = 2 * (self.energy_derivative_psi - self.local_energy * self.derivate_psi1)
        self.loc_energy_gradient_norm = np.sqrt(
            np.dot(self.loc_energy_gradient, self.loc_energy_gradient)
        )

    def shift_parameters(self, shift):
        # The function updates the network parameters by adding a given shift.
        # It is used by the gradient descent method.
        for i in range(self.n_visible):
            self.a[i] = self.a[i] + shift[i]
        for j in range(self.n_hidden):
            self.b[j] = self.b[j] + shift[self.n_visible + j]
        k = self.n_visible + self.n_hidden
        for i in range(self.n_visible):
            for j in range(self.n_hidden):
                self.w[i, j] = self.w[i, j] + shift[k]
                k = k + 1

    def new_parameters(self, best):
        for i in range(self.n_visible):
            self.a[i] = best[i]
        for j in range(self.n_hidden):
            self.b[j] = best[j + self.n_visible]
        k = self.n_visible + self.n_hidden
        for i in range(self.n_visible):
            for j in range(self.n_hidden):
                self.w[i, j] = best[k]
                k = k + 1

    def changes(self, best):
        return best

    def get_gradient(self):
        return self.loc_energy_gradient

    def get_gradient_norm(self):
        return self.loc_energy_gradient_norm


# In[ ]:
