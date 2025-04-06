import numpy as np
import matplotlib.pyplot as plt
import tsplib95
import os
import random

# --- Configuración ---
TSP_FOLDER = './tsplib-master/'
RO = 0.5
MAX_ITER = 1000
N_ANTS = 10
ALPHA = 1
BETA = 1

# --- Funciones auxiliares ---
def is_graphable(problem):
    return problem.node_coords is not None

def get_random_problems(file_list, count):
    selected = []
    available = file_list.copy()
    while len(selected) < count and available:
        filename = random.choice(available)
        filepath = os.path.join(TSP_FOLDER, filename)
        problem = tsplib95.load(filepath)
        if is_graphable(problem):
            selected.append((filename, problem))
        available.remove(filename)
    return selected

def distances(cities):
    n = len(cities)
    d = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            d[i, j] = np.linalg.norm(cities[i] - cities[j])
    np.fill_diagonal(d, np.inf)
    return d

def plot_path(cities, path, title=""):
    for i in range(len(path) - 1):
        inicio = cities[path[i]]
        fin = cities[path[i + 1]]
        plt.plot([inicio[0], fin[0]], [inicio[1], fin[1]], 'b')
    plt.plot([cities[path[-1]][0], cities[path[0]][0]], [cities[path[-1]][1], cities[path[0]][1]], 'b')
    plt.scatter(cities[:, 0], cities[:, 1], c='r')
    plt.title(title)
    plt.grid()
    plt.show()

def solve_aco(cities):
    n = len(cities)
    d = distances(cities)
    nij = 1 / d
    To = np.ones([n, n])
    delta = RO
    best_path = []
    best_path_length = np.inf

    for _ in range(MAX_ITER):
        paths = []
        paths_length = []

        for _ in range(N_ANTS):
            S = np.zeros(n, dtype=bool)
            current_city = np.random.randint(n)
            S[current_city] = True
            path = [current_city]
            path_length = 0

            while not np.all(S):
                unvisited = np.where(~S)[0]
                pij = (To[current_city, unvisited] ** ALPHA) * (nij[current_city, unvisited] ** BETA)
                pij /= np.sum(pij)
                next_city = np.random.choice(unvisited, p=pij)
                path.append(next_city)
                path_length += d[current_city, next_city]
                current_city = next_city
                S[current_city] = True

            path_length += d[current_city, path[0]]
            paths.append(path)
            paths_length.append(path_length)

            if path_length < best_path_length:
                best_path = path.copy()
                best_path_length = path_length

        To *= (1 - RO)
        for path, length in zip(paths, paths_length):
            for i in range(n - 1):
                To[path[i], path[i + 1]] += delta / length
            To[path[-1], path[0]] += delta / length

    return best_path, best_path_length

# --- Clasificación de instancias ---
small_problems = []
medium_problems = []
large_problems = []

for filename in os.listdir(TSP_FOLDER):
    if filename.endswith('.tsp'):
        filepath = os.path.join(TSP_FOLDER, filename)
        try:
            problem = tsplib95.load(filepath)
            if not is_graphable(problem):
                continue
            dimension = problem.dimension
            if 10 <= dimension <= 20:
                small_problems.append(filename)
            elif 50 <= dimension <= 100:
                medium_problems.append(filename)
            elif dimension > 101:
                large_problems.append(filename)
        except:
            print(f"Error cargando {filename}, se omite.")

# --- Selección aleatoria de problemas ---
problems = []
problems += get_random_problems(small_problems, 1)
problems += get_random_problems(medium_problems, 0)
problems += get_random_problems(large_problems, 0)

# --- Ejecutar ACO sobre cada problema ---
print("Problemas seleccionados:")
for fname, problem in problems:
    print(f"\nArchivo: {fname} ({problem.dimension} ciudades)")
    coords = [problem.node_coords[i] for i in problem.node_coords]
    cities = np.array(coords)
    best_path, best_length = solve_aco(cities)
    print(f"→ Longitud ACO: {best_length:.2f}")
    plot_path(cities, best_path, title=f"{fname} - ACO")
