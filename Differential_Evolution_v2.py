import numpy as np
import matplotlib.pyplot as plt

#%% set random seed 
np.random.seed(0)

#%% Constants
Np = 10 # population number
D = 2 # number of dimensions
Cr = 0.5
F = 1

# generate initial population
x = np.random.rand(Np,D)

#%% Test functions

# function given by PhD : f(\text{x$\_$})\text{:=}\sum _{i=1}^{\text{Dim}} \left(20 \sin \left(\frac{1}{2} \pi  (x[[i]]-2 \pi )\right)+(x[[2]]-2 \pi )^2\right)
f = lambda x: np.sum(20*np.sin(0.5*np.pi*(x-2*np.pi))+(x-2*np.pi)**2)
#Ackley function (as found on wikipedia)
f_Ack = lambda x: -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2)))-np.exp(0.5*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])))+20

#%% Define DE class
class Differential_Evolution:
    
    def __init__(self, obj_function, Cr, F):
        self.population_ = None
        self.score_ = None
        self.objective_ = obj_function
        self.crossover_rate_ = Cr
        self.F_ = F
        self.gen_ = 0
        
        
    def generatePopulation(self, pop = [], pop_size=Np, pop_dim=D):
        """ Generate the initial population: either random or the one given """
        if pop == []:
            self.population_ = np.random.rand(pop_size, D)
            print("Population generated randomly")
        else:
            self.population_ = np.array(pop)
            print("Population generated from given array")
    
    def DE(self, it):
        """ Given the initial population, perform Differential Evolution on 
            it generations """
        pop_size = self.population_.shape[0]
        pop_dim = self.population_.shape[1]
        for n_iter in range(it):
            # generate a trial population
            u = self.population_.copy()
            for i in range(pop_size):
                r0,r1,r2=i,i,i
                while r0==i:
                    r0 = np.random.randint(pop_size)
                while r1==r0 or r1==i:
                    r1 = np.random.randint(pop_size)
                while r2==r1 or r2==r0 or r2==i:
                    r2 = np.random.randint(pop_size)
                jrand = np.random.randint(pop_dim)
    
                for j in range(pop_dim):
                    b = np.random.rand()
                    if b<=self.crossover_rate_ or j==jrand:
                        u[i,j] = self.population_[r0,j]+self.F_*(x[r1,j]-x[r2,j])
                    else:
                        u[i,j] = self.population_[i,j]
    
            # select the next generation
            for i in range(pop_size):
                if self.objective_(u[i,:])<self.objective_(self.population_[i,:]):
                    self.population_[i,:]=u[i,:].copy()
                    
        self.gen_ +=it
        print("DE done for ", it, " iterations")
        return self.population_
    
    def representPopulation(self, ax):
        """ Represents population, computes and returns the best individual """
        ax.scatter(self.population_.T[0], self.population_.T[1], color = 'blue')
        ax.set_title("Population at generation %i" %self.gen_)
        best = self.population_[np.argmin(np.array([self.objective_(individual) for individual in self.population_]))]
        ax.plot([best[0]], [best[1]], marker = 'x', color = 'red')
        plt.show()
        return best
         
#%% Applying DE to Ackley
difEv = Differential_Evolution(f_Ack, Cr, F)
difEv.generatePopulation()
pop_init = difEv.population_

fig, ax = plt.subplots(nrows=1, ncols=3, sharex = True, sharey = True)
# After 10 generations
pop_g1 = difEv.DE(10)
best_g1 = difEv.representPopulation(ax[0])


# After 100 generations
pop_g2 = difEv.DE(90)
best_g2 = difEv.representPopulation(ax[1])

# After 1000 generations 
pop_g3 = difEv.DE(900)
best_g3 = difEv.representPopulation(ax[2])

plt.suptitle("Differential Evolution with population of size %i" %Np)
