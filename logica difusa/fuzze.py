import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return ((5 <= x) & (x <= 12 ))* 1

def baja(x):
    return (x <= 10)*1 + ((x > 10) & (x <= 20))*((20 - x)/10) + (x > 20)*0

def media(x):
    return (x < 10)*0 + (((x >= 10) & (x <= 25))*((x - 10)/15)) + ((x > 25) & (x <= 30))*((30 - x)/5) + (x > 30)*0

def alta(x):
    return (x <= 25)*0 + ((x > 25) & (x <= 35))*((x - 25)/10) + (x > 35)*1

def bajaHumedad(x):
    return ((x <= 60) & (x >= 20))*(60-x)/60 + ((x > 60) )*0

def mediaHumedad(x):
    return (x < 20)*0 + ((x >= 20) & (x <= 60))*((x - 20)/40) + ((x > 60))*((80 - x)/20)

def altaHumedad(x):
    return (x <= 60)*0 + ((x > 60) & (x <= 80))*((x - 60)/20) + (x > 80)*1

def bajaVentilador(x):
    return (x <= 10)*1 + ((x > 10) & (x <= 20))*((20 - x)/10) + (x > 20)*0

def mediaVentilador(x):
    return (x < 10)*0 + (((x >= 10) & (x <= 25))*((x - 10)/15)) + ((x > 25) & (x <= 30))*((30 - x)/5) + (x > 30)*0

def altaVentilador(x):
    return (x <= 25)*0 + ((x > 25) & (x <= 35))*((x - 25)/10) + (x > 35)*1

def subconjuntos(subconjunto, subconjunto2):
    return np.all(subconjunto <= subconjunto2, axis=0)

#operadores difusos
def AND(A,B):
    return np.minimum(A,B)

def OR(A,B):
    return np.maximum(A,B)

def NOT(A):
    return 1-A

def implication(A,B):
    return OR(NOT(A),B)

x = np.linspace(1,50,100)

plt.plot(x,f(x))
# plt.plot(x,baja(x))
# plt.plot(x,media(x))
# plt.plot(x,alta(x))
plt.plot(x,bajaHumedad(x))
plt.plot(x,mediaHumedad(x))
plt.plot(x,altaHumedad(x))

a = 7
b = 2 
plt.axvline(a, color = 'r', linestyle=':')
plt.axvline(b, color = 'r', linestyle=':')

Temperaturas = np.arange(10,36,5)
for temp in Temperaturas:
    plt.axvline(temp, color = 'g', linestyle=':')


plt.grid()
plt.show()

print("Subconjuntos",subconjuntos(baja(np.array([11, 20, 30])), baja(np.array([10, 20, 30]))))
print(baja(Temperaturas))
print(media(Temperaturas))
print(alta(Temperaturas))



x = 0.6
y = 0.5
print(AND(x,y))

x = np.array([10,15,20,25,30,35])
bx = baja(x)
mx = media(x)
print('---------------------------AND-----------------------------------')
print( AND(bx,mx))

print("-----------------------------or---------------------------")
print(OR(bx,mx))

print("---------------------------not------------------------------")
print(media(x))
print(NOT(media(x)))

print('---------------------------implication------------------------')
print(implication(0.6,0.8))

print('-------------------------MPP---------------------------')
print(AND((implication(0.8,0.7)),0.8))

print('------------------------mtt---------------------------------')
print(AND(implication(0.8,0.7),NOT(0.7)))
