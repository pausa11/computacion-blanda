import numpy as np
import matplotlib.pyplot as plt
import tsplib95
import os
import random
import math

# --- Configuración ---
TSP_FOLDER = './tsplib-master/'
RO = 0.3
MAX_ITER = 1000
N_ANTS = 40
ALPHA = 1.5
BETA = 5

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

def deg_to_rad(coord):
    deg = int(coord)
    min = coord - deg
    return math.pi * (deg + 5.0 * min / 3.0) / 180.0

def geo_distance(coord1, coord2):
    RRR = 6378.388  # radio terrestre en km (TSPLIB)
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1 = deg_to_rad(lat1)
    lon1 = deg_to_rad(lon1)
    lat2 = deg_to_rad(lat2)
    lon2 = deg_to_rad(lon2)

    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)

    return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1)

def compute_distance_matrix(cities, edge_type="EUC_2D"):
    n = len(cities)
    d = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == j:
                d[i, j] = np.inf
            else:
                if edge_type == "GEO":
                    d[i, j] = geo_distance(cities[i], cities[j])
                else:
                    d[i, j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
    return d

def plot_path(cities, path, title=""):
    cities = np.array(cities)
    for i in range(len(path) - 1):
        inicio = cities[path[i]]
        fin = cities[path[i + 1]]
        plt.plot([inicio[0], fin[0]], [inicio[1], fin[1]], 'b')
    plt.plot([cities[path[-1]][0], cities[path[0]][0]],
             [cities[path[-1]][1], cities[path[0]][1]], 'b')
    plt.scatter(cities[:, 0], cities[:, 1], c='r')
    plt.title(title)
    plt.grid()
    plt.show()

def solve_aco(cities, edge_type):
    n = len(cities)
    d = compute_distance_matrix(cities, edge_type)
    nij = 1 / d
    To = np.ones([n, n])
    delta = 1.0 
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
                suma_pij = np.sum(pij)

                # ⚠️ Evitar división por 0 o NaN
                if suma_pij == 0 or np.isnan(suma_pij):
                    pij = np.ones(len(unvisited)) / len(unvisited)
                else:
                    pij /= suma_pij

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

        # Evaporación + protección contra 0s
        To = np.maximum(To * (1 - RO), 1e-10)
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
problems += get_random_problems(small_problems, 0)
problems += get_random_problems(medium_problems, 1)
problems += get_random_problems(large_problems, 0)

# --- Ejecutar ACO sobre cada problema ---
print("Problemas seleccionados:")
for fname, problem in problems:
    print(f"\nArchivo: {fname} ({problem.dimension} ciudades)")
    coords = [problem.node_coords[i] for i in problem.node_coords]
    cities = list(coords)
    edge_type = problem.edge_weight_type if problem.edge_weight_type else "EUC_2D"
    best_path, best_length = solve_aco(cities, edge_type)
    print(f"→ Tipo: {edge_type} | Longitud ACO: {best_length:.2f}")
    plot_path(cities, best_path, title=f"{fname} - ACO")
