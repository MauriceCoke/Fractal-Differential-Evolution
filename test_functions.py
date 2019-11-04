import numpy as np

#rastrigin and rosenbrock functions in 2D

def rastrigin(X,Y):
    return (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
  
def rosenbrock(X,Y):
    return (1 - X)**2 + ((y - x**2)**2)*100

#same functions in n (>=2) dimensions

def rastrigin_n(X):
    res=0
    for xj in X:
        res+= xj**2 - 10 * np.cos(2 * np.pi * xj)+10
    return res
    

def rosenbrock_n(X):
    n=len(X)
    res=0
    for i in range(n-1):
        res+= (1 - X[i])**2 + ((X[i+1] - X[i]**2)**2)*100
    return res 
