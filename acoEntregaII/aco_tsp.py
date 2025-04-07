import numpy as np
import matplotlib.pyplot as plt
import tsplib95
import os
import math
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

TSP_FOLDER = './tsplib-master/'

known_solutions = {
    "a280.tsp": 2579,
    "att48.tsp": 10628,
    "bayg29.tsp": 1610,
    "burma14.tsp": 3323,
    "berlin52.tsp": 7542,
    "pr76.tsp": 108159,
}

DEFAULT_MAX_ITER = 1000
DEFAULT_RO = 0.3
DEFAULT_ALPHA = 1.5
DEFAULT_BETA = 5
DEFAULT_N_ANTS = 40

def is_graphable(problem):
    """Verifica si hay coordenadas para graficar/usar en matriz de distancias."""
    return problem.node_coords is not None

def deg_to_rad(coord):
    deg = int(coord)
    min_ = coord - deg
    return math.pi * (deg + 5.0 * min_ / 3.0) / 180.0

def geo_distance(coord1, coord2):
    """Distancia según EDGE_WEIGHT_TYPE=GEO (TSPLIB)."""
    RRR = 6378.388
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1 = deg_to_rad(lat1)
    lon1 = deg_to_rad(lon1)
    lat2 = deg_to_rad(lat2)
    lon2 = deg_to_rad(lon2)

    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)

    return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2
                                      - (1.0 - q1) * q3)) + 1)

def compute_distance_matrix(cities, edge_type="EUC_2D"):
    """Genera la matriz de distancias para el conjunto de ciudades."""
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
                    # Distancia euclidiana
                    d[i, j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
    return d

def solve_aco(cities, edge_type="EUC_2D", max_iter=DEFAULT_MAX_ITER, ro=DEFAULT_RO, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, n_ants=DEFAULT_N_ANTS):
    """
    Ejecuta el ACO con los parámetros dados sobre un conjunto de cities.
    Retorna (best_path, best_path_length).
    """
    n = len(cities)
    d = compute_distance_matrix(cities, edge_type)
    nij = 1 / d  # Atractividad
    To = np.ones([n, n])  # Feromonas
    delta = 1.0  # refuerzo base
    best_path = []
    best_path_length = np.inf

    for _ in range(max_iter):
        paths = []
        paths_length = []

        for _ant in range(n_ants):
            S = np.zeros(n, dtype=bool)
            current_city = np.random.randint(n)
            S[current_city] = True
            path = [current_city]
            path_length = 0

            while not np.all(S):
                unvisited = np.where(~S)[0]
                pij = (To[current_city, unvisited] ** alpha) * (nij[current_city, unvisited] ** beta)
                suma_pij = np.sum(pij)

                # Evitar división por cero o NaN
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

            # Actualizar mejor camino
            if path_length < best_path_length:
                best_path = path.copy()
                best_path_length = path_length

        # Evaporación y protección
        To = np.maximum(To * (1 - ro), 1e-10)
        for path, length in zip(paths, paths_length):
            for i in range(n - 1):
                To[path[i], path[i + 1]] += delta / length
            # Cierre
            To[path[-1], path[0]] += delta / length

    return best_path, best_path_length

def plot_path(cities, path, title=""):
    """Grafica la ruta resultante en 2D."""
    cities = np.array(cities)
    for i in range(len(path) - 1):
        inicio = cities[path[i]]
        fin = cities[path[i + 1]]
        plt.plot([inicio[0], fin[0]], [inicio[1], fin[1]], 'b')
    # Cerrar ciclo
    plt.plot([cities[path[-1]][0], cities[path[0]][0]],
             [cities[path[-1]][1], cities[path[0]][1]], 'b')
    plt.scatter(cities[:, 0], cities[:, 1], c='r')
    plt.title(title)
    plt.grid()
    plt.show()

# 1) Listas de valores a experimentar (ajusta a tu gusto)
ANTS_VALUES = [10, 20, 30]         # n_ants
ALPHA_VALUES = [0.5,1.5]     # α
BETA_VALUES = [1.0, 3.0, 5.0]      # β
RHO_VALUES = [0.1, 0.3, 0.5]       # evaporación
MAX_ITER_VALUES = [1000]      # iteraciones

def experiment_task(params, cities, edge_type, known):
    ants, alpha, beta, rho, iters = params
    path, length = solve_aco( cities, edge_type=edge_type, max_iter=iters, ro=rho, alpha=alpha, beta=beta, n_ants=ants )
    gap = None
    if known:
        gap = (length - known) / known * 100
    return {
        'ants': ants,
        'alpha': alpha,
        'beta': beta,
        'rho': rho,
        'max_iter': iters,
        'found_length': length,
        'gap_%': gap,
        'path': path
    }

def run_experiment_parallel(fname):
    filepath = os.path.join(TSP_FOLDER, fname)
    problem = tsplib95.load(filepath)
    edge_type = problem.edge_weight_type if problem.edge_weight_type else "EUC_2D"

    coords = [problem.node_coords[i] for i in problem.node_coords]
    cities = list(coords)
    n = len(cities)

    best_known = known_solutions.get(fname)

    print(f"\n🧪 Ejecutando experimento paralelo con {fname} ({n} ciudades)")

    all_combinations = list(product(ANTS_VALUES, ALPHA_VALUES, BETA_VALUES, RHO_VALUES, MAX_ITER_VALUES))

    results = []
    total = len(all_combinations)
    completadas = 0
    siguiente_avance = 10  # Siguiente porcentaje a reportar

    with ProcessPoolExecutor() as executor:
        futures = []
        for combo in all_combinations:
            futures.append(executor.submit(experiment_task, combo, cities, edge_type, best_known))

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completadas += 1
            porcentaje = (completadas / total) * 100

            if porcentaje >= siguiente_avance:
                print(f"🔄 Progreso: {int(porcentaje)}% ({completadas}/{total}) combinaciones completadas")
                siguiente_avance += 10


    results.sort(key=lambda x: x['found_length'])

    # Guardar CSV
    csv_filename = f"resultados_{fname.replace('.tsp', '')}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['ants', 'alpha', 'beta', 'rho', 'max_iter', 'found_length', 'gap_%']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"\n📁 Resultados guardados en: {csv_filename}")

    # Graficar mejor
    plot_path(cities, results[0]['path'], f"Mejor resultado - {fname}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_experiment_parallel('burma14.tsp')


