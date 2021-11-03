#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[3]:


class Trainer:
    def __init__(self, nqs, hm, qm, mc, grad, nro_iterations, optimizer):
        self.nqs = nqs
        self.hm = hm
        self.qm = qm
        self.mc = mc
        self.grad = grad
        self.nro_iterations = nro_iterations
        self.optimizer = optimizer

    def train(self, nro_samples):
        # The function trains the network

        self.grad.setup()

        for i in range(0, self.nro_iterations):

            self.mc.run_mc(nro_samples, i)

            if self.optimizer == "simple":
                shift = self.grad.parameter_shift(
                    self.qm.get_gradient()
                )  # I give as a parameter the energy gradient, and I calculate DGS
                self.qm.shift_parameters(
                    shift
                )  # I update the weights and bias

            elif self.optimizer == "adam":
                shift = self.grad.adam(
                    self.qm.get_gradient(), i + 1
                )  # I give as a parameter the energy gradient, and I calculate Adam
                self.qm.shift_parameters(
                    shift
                )  # I update the weights and bias
            else:
                exit
                # self.mc.run_mc(int(1e6),1)
        print(self.mc.account)


# In[ ]:
