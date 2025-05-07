import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return x.dot(ws.T) + bs

def softMax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def activate(z):
  return 1/(1+np.exp(-z))

def log_loss(y,p):
  return -y*np.log(p)-(1-y)*np.log(1-p)

#datos de entrenamiento
inputs = np.array([(0.0000, 0.0000), (0.2778, 0.2500), (0.2778, 0.9375), (0.9167, 0.6563),
(0.4167, 0.2500), (0.3611, 0.3438), (0.3333, 0.4063), (0.9722, 0.3750),
(0.0833, 0.3438), (0.6389, 0.3438), (0.4167, 0.6875), (0.7500, 0.6875),
(0.0833, 0.1875), (0.9167, 0.5313), (0.1389, 0.2500), (0.8333, 0.6250),
(0.8056, 0.6250), (0.1944, 1.0000), (0.8333, 0.5625), (0.4167, 1.0000),
(1.0000, 0.6875), (0.4722, 0.6563), (0.3611, 0.5625), (0.4722, 0.8438),
(0.1667, 0.3125), (0.4167, 0.9375), (0.3611, 0.9688), (0.9167, 0.3438),
(0.0833, 0.0313), (0.3333, 0.8750)])
X = inputs.copy()

red=      np.array([1,0,0])
green=    np.array([0,1,0])
blue=     np.array([0,0,1])
#aprendizaje supervisado
targets=  np.array([red, red, blue, green, red, red, red, green, red, green,
blue, green, red,
green, red, green, green, blue, green, blue, green, blue, blue, blue,
red, blue, blue, green, red, blue])

# plt.scatter(X[:,0],X[:,1],c = targets)
# plt.grid()
# plt.show()

m = 3 #categorias
inp = 2 #entradas / l-w
bs = np.ones([1,m])

ws = np.ones([m,inp])*0.1
# print(ws)

#primera activacion
# pred= f(np.array([0.1,0.3]))

#enviar nuestros valores
# pred= f(X)

# act = softMax(pred)


lr=0.01 #tasa de aprendizaje

iterations=10000
for iter in range(iterations):
    pred = f(X)
    act = softMax(pred)   
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



