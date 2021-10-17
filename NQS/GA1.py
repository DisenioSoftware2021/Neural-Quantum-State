#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy 
import NQS
import quantumodel1 as quam
import MCMethod1 as MCM
import hamiltonian as hm
import Probcondicional1 as cond


# In[24]:


class genetic:
    def __init__(self,n_hidden,n_visible,n_gen):
        #self.nqs=nqs
        #self.qm=qm
        self.n_hidden=n_hidden
        self.n_visible=n_visible
        self.n_gen=n_gen
        self.parameter=self.n_hidden+self.n_visible+(self.n_visible*self.n_hidden)#+self.n_hidden
        self.gen=numpy.zeros([self.n_gen,self.parameter])
        #self.fitness_function = self.qm.valormedio.locengradientnorm
        self.fitness=0.
        
    def poblacion_inicial(self):
        sigma_init=0.1#1./numpy.sqrt(2)
        for l in range(0,self.n_gen): 
        
            for i in range(0,self.n_visible):
                self.gen[l,i]=numpy.random.normal(0,sigma_init)
            for j in range(0,self.n_hidden):
                self.gen[l,j+self.n_visible]=numpy.random.normal(0,sigma_init)
            k=self.n_visible+self.n_hidden
            for i in range(0,self.n_visible):
                for j in range(0,self.n_hidden):
                    self.gen[l,k]=numpy.random.normal(0,sigma_init)
                    k=k+1
            #for i in range(0,self.n_visible):
             #   self.gen[l,i+self.n_hidden+self.n_visible+(self.n_visible*self.n_hidden)]=numpy.random.rand()-0.5
        return self.gen


# In[ ]:




