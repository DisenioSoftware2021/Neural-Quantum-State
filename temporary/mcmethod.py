#!/usr/bin/env python
# coding: utf-8

# In[4]:

# In[5]:


class MCMethod:
    def __init__(self, qm, seed):
        self.condi = qm.condi
        self.qm = qm
        self.seed = seed
        self.account = 0

    def run_mc(self, sample_number, nro):
        self.qm.zzero()
        self.qm.set_up_sampling(self.qm.condi.x)

        effective_n_samples = 0
        # file=open("gd_05_large.txt","a")
        equilibration = False

        for sample in range(0, sample_number):
            self.qm.condi.gibbs(self.seed)
            equilibration = False
            # Begins to calculate the energy values

            if sample > (0.1 * sample_number):
                equilibration = True
            if equilibration is True:
                self.qm.accumulator(self.qm.condi.x)
                # file.writelines(str(self.qm.local_energy))

                effective_n_samples += 1

        self.qm.average_value(
            effective_n_samples
        )  # Obtain the mean value of the energy and calculate the variance and gradient

        print(
            f"valor de E= {self.qm.local_energy:0.9f}",
            self.qm.loc_energy_gradient_norm,
            nro,
        )
        # file.write(f"{self.qm.local_energy:0.7f}"+"  "+f"{self.qm.var:0.7e}"+"  "+
        # f"{np.sqrt(self.qm.var):0.7e}"+"  "+f"{self.qm.loc_energy_gradient_norm:0.7f}"+"  "
        # +f"{np.abs(self.qm.local_energy-2):0.7f}"+"  "+"  "+str(nro)+"\n")

        if (
            self.qm.local_energy < 2.0
        ):  # Counter that allows you to see when you are below the desired energy value

            self.account += 1
