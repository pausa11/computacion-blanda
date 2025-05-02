#Algoritmo Regresión lineal múltiple:

import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return x.dot(w) + b

def activate(z):
  return 1/(1+np.exp(-z))

def log_loss(y,p):
  return -y*np.log(p)-(1-y)*np.log(1-p)


#training data
X = np.array([(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709), (0.3600, 0.4702), 
              (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026), (0.6000, 0.6358), (0.6200, 0.3212), 
              (0.6600, 0.7185), (0.7000, 0.7351), (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), 
              (1.0000, 1.0000)])

targets =np.concatenate([np.zeros(12,dtype=int),np.ones(4,dtype=int)])
print(targets)

plt.scatter(X[:,0],X[:,1],c=targets)
plt.grid()
plt.show()

w=np.array([0.1,0.1])

b=1

lr=0.01 #tasa de aprendizaje

iterations=10000
for iter in range(iterations):
    pred = f(X)
    act = activate(pred)   
    #   print(act)
    #log_cost
    cost = log_loss(targets,act)
    
    
    mse=np.mean(cost)
    print(f'iter:{iter},cost:{mse:.3f}')


    z_d = act - targets #derivada parcial respecto a z
    # Gradiente con respecto a w
    # w_d= -2*(X.T).dot(targets-f(X))
    w_d = (X.T).dot(z_d)
    avg_w_d = w_d / np.size(targets)

    # gradiente respecto a b
    b_d=np.ones([1,np.size(targets)]).dot(z_d)
    avg_b_d=b_d/np.size(targets)
    #print(avg_b_d)
    # Actualización de w y b
    w-=lr*avg_w_d
    b-=lr*avg_b_d


#-------------------------------------------------------------------------

#test data
XT = np.array([(0.16,0.13),(0.9,0.9),(0.2,0.6)])
TT = np.array([0,1,0])

pred= f(XT)
act = activate(pred)
for a , t in zip(act, TT):
  print(f'targer:{t},act:{a:.3f}')


plt.scatter(X[:,0],X[:,1],c = targets)
plt.scatter(0.2,0.6, c = 'red')
plt.grid()
plt.show()



#------------------------------


