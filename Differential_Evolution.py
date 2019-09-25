import numpy as np

# set random seed 
np.random.seed(0)

# Constants
Np = 10 # population number
D = 2 # number of dimensions
Cr = 0.5
F = 1

# generate initial population
x = np.random.rand(Np,D)

# Test functions

# function given by PhD : f(\text{x$\_$})\text{:=}\sum _{i=1}^{\text{Dim}} \left(20 \sin \left(\frac{1}{2} \pi  (x[[i]]-2 \pi )\right)+(x[[2]]-2 \pi )^2\right)
f = lambda x: np.sum(20*np.sin(0.5*np.pi*(x-2*np.pi))+x[2]-2*np.pi)
#Ackley function (as found on wikipedia)
f_Ack = lambda x: -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2)))-np.exp(0.5*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])))+20


def diff_ev(x,f=f_Ack):
    for n_iter in range(100):
        # generate a trial population
        u = x.copy()
        for i in range(Np):
            r0,r1,r2=i,i,i
            while r0==i:
                r0 = np.random.randint(Np)
            while r1==r0 or r1==i:
                r1 = np.random.randint(Np)
            while r2==r1 or r2==r0 or r2==i:
                r2 = np.random.randint(Np)
            jrand = np.random.randint(D)

            for j in range(D):
                b = np.random.rand()
                if b<=Cr or j==jrand:
                    u[i,j] = x[r0,j]+F*(x[r1,j]-x[r2,j])
                else:
                    u[i,j] = x[i,j]

        # select the next generation
        for i in range(Np):
            if f(u[i,:])<f(x[i,:]):
                x[i,:]=u[i,:].copy()

diff_ev(x)
print(x)