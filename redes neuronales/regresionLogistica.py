import numpy as np
import matplotlib.pyplot as plt

#regresion logistica
#clasificacion lineal
def f(x):
    return x.dot(w) + b

def activate(z):
    return 1/(1+np.exp(-z))

def log_loss(y,p):
    return -y * np.log(p)-(1-y)*np.log(1-p)

w=np.array([0.1,0.1])
b=0.1

X = np.array([(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709)])
pred= f(X)
print(pred)
act=activate(pred)
print(act)

targets = np.array([0,0,1,1])#dos primeros a etiqueta uno y los dos ultimos a etiqueta 1
print(act)
print(targets)

for a , t in zip(act,targets):
    print(f'targets:{t}, act:{a}') #el error entre lo que se quiere y contra lo que se predice

print(log_loss(1,0.95))
print(log_loss(0,0.05))

print(log_loss(1,0.95))
print(log_loss(0,0.95))

print(log_loss(targets,act))

