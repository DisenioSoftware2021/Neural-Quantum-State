# In[1]:


import time

import numpy as np

import condprobability1 as cond
import gradient as gradi
import hamiltonian as hm
import mcmethod as mcm
import nqs
import quantumodel1 as quam
import trainer

# import GA1


# In[2]:
# The parameters that the algorithm needs are established.
# harmonic oscillator frequency: omega,
# variance of initial gaussian: sigma,
# type of interaction, "harmonic_oscillator",
# "coulomb" or "calogero": coulomb_interaction,
# initial distribution of weights and bias: "normal" or "uniform",
# number of particles: n_particles,
# number of dimensions: n_dimensions,
# (number of visible units=n_particles+n_dimensions)
# number of hidden units: n_hidden.

# model
omega = 1.0
sigma = 1 / np.sqrt(2 * omega)
coulomb_interaction = "harmonic_oscillator"
nqs_initialization = "normal"
n_particles = 2
n_dimensions = 2
n_hidden = 4
seed_1 = np.random.seed()
d = 0
# Method
# The number of Gibbs samples to be performed is indicated
number_of_samples = int(1e5)
seed_2 = np.random.seed()
account = 0

# Trainer
# The type of optimization, "adam" or "simple",
# is chosen: minimizer_type.
# The number of times the method will be performed
# is indicated: n_iterations.
# The value of the learning rate is specified: learning_rate.
# The gamma value associated with the moment is indicated: gamma,
# (if the type of optimization chosen is "adam", gamma=0.0).
minimizer_type = "adam"
n_iterations = 200
learning_rate = 0.5
gamma = 0.0


# In[3]:
# The classes are called and the program is executed with the parameters
# specified above.

qs = nqs.NQS(n_hidden, n_dimensions, n_particles, sigma, d)
grad = gradi.Gradient(learning_rate, gamma, qs)

nqs_positive = cond.NQSpositive(qs)
ham = hm.Hamiltonian(omega, coulomb_interaction)
qm = quam.QuantumModel(qs, ham, nqs_positive)
mc = mcm.MCMethod(qm, np.random.seed())
qs.initi(nqs_initialization, seed_1)
t = trainer.Trainer(qs, ham, qm, mc, grad, n_iterations, minimizer_type)
tic = time.perf_counter()
t.train(number_of_samples)
toc = time.perf_counter()
print(account)
print(f"Duracion total= {toc - tic:0.4f} seconds")
