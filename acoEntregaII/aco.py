import numpy as np
import matplotlib.pyplot as plt

# --- Generar ciudades ---
n = 100
np.random.seed(0)
cities = np.random.uniform(0, 10, [n, 2])  # Generar ciudades en un espacio 2D

# --- Calcular matriz de distancias ---
def distances(cities):
    n = len(cities)
    d = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            d[i, j] = np.linalg.norm(cities[i] - cities[j])
    np.fill_diagonal(d, np.inf)
    return d

d = distances(cities)
nij = 1 / d  # Atractividad
To = np.ones([n, n])  # Feromonas iniciales

# --- Parámetros ACO ---
ro = 0.5  # Tasa de evaporación
delta = ro  # Refuerzo
maxIter = 100
n_ants = 10
alpha = 1
beta = 1

best_path = []
best_path_length = np.inf

# --- Algoritmo ACO ---
for iter in range(maxIter):
    paths = []
    paths_length = []

    for ant in range(n_ants):
        S = np.zeros(n, dtype=bool)
        current_city = np.random.randint(n)
        S[current_city] = True
        path = [current_city]
        path_length = 0

        while not np.all(S):
            unvisited = np.where(~S)[0]
            pij = (To[current_city, unvisited] ** alpha) * (nij[current_city, unvisited] ** beta)
            pij /= np.sum(pij)
            next_city = np.random.choice(unvisited, p=pij)
            path.append(next_city)
            path_length += d[current_city, next_city]
            current_city = next_city
            S[current_city] = True

        # Cerrar el ciclo
        path_length += d[current_city, path[0]]
        paths.append(path)
        paths_length.append(path_length)

        # Actualizar mejor camino
        if path_length < best_path_length:
            best_path = path.copy()
            best_path_length = path_length

    # --- Actualización de feromonas ---
    To *= (1 - ro)
    for path, length in zip(paths, paths_length):
        for i in range(n - 1):
            To[path[i], path[i + 1]] += delta / length
        To[path[-1], path[0]] += delta / length  # Cierre del ciclo

# --- Graficar resultado en 2D ---
def plot_path(cities, path):
    for i in range(len(path) - 1):
        inicio = cities[path[i]]
        fin = cities[path[i + 1]]
        plt.plot([inicio[0], fin[0]], [inicio[1], fin[1]], 'b')
    # Cerrar el ciclo
    plt.plot([cities[path[-1]][0], cities[path[0]][0]],
             [cities[path[-1]][1], cities[path[0]][1]], 'b')
    plt.scatter(cities[:, 0], cities[:, 1], c='r')
    plt.title("Mejor ruta encontrada (ACO)")
    plt.grid()
    plt.show()

print(f'Mejor longitud encontrada: {best_path_length:.4f}')
plot_path(cities, best_path)
