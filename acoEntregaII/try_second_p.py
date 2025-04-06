import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_tsp(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    coords = []
    with open(filename, 'r') as f:
        in_section = False
        for line in f:
            line = line.strip()
            if line.startswith('NODE_COORD_SECTION'):
                in_section = True
                continue
            if line.startswith('EOF'):
                break
            if in_section:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
    if not coords:
        raise ValueError(f"No coordinates found in file: {filename}")
    return np.array(coords)


def distances(cities):
    n = len(cities)
    d = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            d[i, j] = np.linalg.norm(cities[i] - cities[j])
    # Evitar división por cero asignando un valor muy alto (o infinito) en la diagonal
    np.fill_diagonal(d, np.inf)
    return d


def aco_tsp(cities, n_ants, max_iter, alpha, beta, ro, delta):    
    n = len(cities)
    d = distances(cities)
    nij = 1.0 / d
    np.fill_diagonal(nij, 0.0)
    # feromonas iniciales
    tau = np.ones((n, n))
    
    best_path = None
    best_length = np.inf

    for it in range(max_iter):
        all_paths = []
        all_lengths = []

        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            current = np.random.randint(n)
            visited[current] = True
            path = [current]
            length = 0.0

            # construir la solución
            while not visited.all():
                unvisited = np.where(~visited)[0]
                tau_k = tau[current, unvisited] ** alpha
                eta_k = nij[current, unvisited] ** beta
                probs = tau_k * eta_k
                probs /= probs.sum()
                nxt = np.random.choice(unvisited, p=probs)

                path.append(nxt)
                length += d[current, nxt]
                current = nxt
                visited[current] = True

            # cerrar el ciclo
            length += d[current, path[0]]
            all_paths.append(path)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_path = path.copy()

        # evaporación
        tau *= (1 - ro)
        # refuerzo
        for path, L in zip(all_paths, all_lengths):
            for i in range(len(path) - 1):
                tau[path[i], path[i+1]] += delta / L
            tau[path[-1], path[0]] += delta / L

    return best_path, best_length


def plot_path(cities, path):
    for i in range(len(path)-1):
        inicio = cities[path[i]]
        fin = cities[path[i+1]]
        plt.plot([inicio[0], fin[0]], [inicio[1], fin[1]], 'b')
    plt.plot([cities[path[-1]][0], cities[path[0]][0]], [cities[path[-1]][1], cities[path[0]][1]], 'b')
    plt.scatter(cities[:,0], cities[:,1], c='r')
    plt.grid()
    plt.show()

def solution():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsp_file', type=str, default=None, help='Path to TSPLIB .tsp file')
    args = parser.parse_args()

    if args.tsp_file:
        cities = load_tsp(args.tsp_file)
        print(f"Loaded {len(cities)} cities from {args.tsp_file}")
    else:
        n = 10
        np.random.seed(0)
        cities = np.random.uniform(0, 10, (n, 2))
        print(f"No .tsp file provided")

    best_path, best_length = aco_tsp(cities,
                                        n_ants=10,
                                        max_iter=100,
                                        alpha=1.0,
                                        beta=3.0,
                                        ro=0.5,
                                        delta=0.5)

    print(f"Best path: {best_path}")
    print(f"Best path length: {best_length:.4f}")
    plot_path(cities, best_path)

solution()
