import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo
n = 100  # Número de ciudades
n_ants = 10  # Número de hormigas
maxIter = 100  # Número máximo de iteraciones
alpha = 1  # Importancia de las feromonas
beta = 2  # Importancia de la visibilidad (inversa de distancia)
ro = 0.5  # Tasa de evaporación
Q = 1  # Constante para actualización de feromonas

# Generar ciudades aleatorias (usando solo 2 dimensiones para visualización)
np.random.seed(0)
cities = np.random.uniform(0, 10, [n, 2])

# Calcular matriz de distancias
def distances(cities):
    n = len(cities)
    d = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            d[i, j] = np.linalg.norm(cities[i] - cities[j])
    # Evitar división por cero en la diagonal
    np.fill_diagonal(d, np.inf)
    return d

d = distances(cities)

# Calcular matriz de visibilidad (inversa de la distancia)
visibility = 1.0 / d

# Inicializar matriz de feromonas
pheromones = np.ones([n, n])

# Función para visualizar el recorrido
def plot_path(cities, path):
    plt.figure(figsize=(10, 8))
    # Dibujar el camino entre ciudades
    for i in range(len(path)-1):
        start = cities[path[i]]
        end = cities[path[i+1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-')
    # Conectar la última ciudad con la primera para completar el ciclo
    plt.plot([cities[path[-1]][0], cities[path[0]][0]], 
             [cities[path[-1]][1], cities[path[0]][1]], 'b-')
    # Graficar las ciudades
    plt.scatter(cities[:,0], cities[:,1], c='r', s=50)
    # Etiquetar las ciudades
    for i, city in enumerate(cities):
        plt.text(city[0]+0.1, city[1]+0.1, str(i), fontsize=8)
    plt.grid(True)
    plt.title(f'Mejor recorrido encontrado (longitud: {best_path_length:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Función para calcular la longitud de un recorrido
def calculate_path_length(path, distance_matrix):
    total_length = 0
    for i in range(len(path)-1):
        total_length += distance_matrix[path[i], path[i+1]]
    # Añadir la distancia de regreso al punto de inicio
    total_length += distance_matrix[path[-1], path[0]]
    return total_length

# Variables para almacenar el mejor recorrido
best_path = None
best_path_length = np.inf
best_iteration = 0

# Historial para graficar la convergencia
iteration_history = []
best_length_history = []

# Algoritmo ACO
for iter in range(maxIter):
    all_paths = []
    all_path_lengths = []
    
    # Cada hormiga construye un recorrido
    for ant in range(n_ants):
        # Seleccionar ciudad inicial aleatoria
        current_city = np.random.randint(n)
        path = [current_city]
        visited = np.zeros(n, dtype=bool)
        visited[current_city] = True
        
        # Construir el recorrido
        while False in visited:
            unvisited = np.where(visited == False)[0]
            
            # Calcular probabilidades de transición
            probabilities = np.zeros(len(unvisited))
            for i, city in enumerate(unvisited):
                probabilities[i] = (pheromones[current_city, city]**alpha) * \
                                   (visibility[current_city, city]**beta)
            
            # Normalizar probabilidades
            if np.sum(probabilities) > 0:
                probabilities = probabilities / np.sum(probabilities)
            else:
                # Si todas las probabilidades son cero, elegir uniformemente
                probabilities = np.ones(len(unvisited)) / len(unvisited)
            
            # Seleccionar la siguiente ciudad según las probabilidades
            next_city = np.random.choice(unvisited, p=probabilities)
            path.append(next_city)
            visited[next_city] = True
            current_city = next_city
        
        # Calcular longitud del recorrido
        path_length = calculate_path_length(path, d)
        all_paths.append(path)
        all_path_lengths.append(path_length)
        
        # Actualizar mejor recorrido global
        if path_length < best_path_length:
            best_path = path.copy()
            best_path_length = path_length
            best_iteration = iter
    
    # Evaporación de feromonas
    pheromones = (1-ro) * pheromones
    
    # Depositar nuevas feromonas
    for path, path_length in zip(all_paths, all_path_lengths):
        for i in range(len(path)-1):
            pheromones[path[i], path[i+1]] += Q / path_length
            pheromones[path[i+1], path[i]] += Q / path_length  # Simetría
        # Cerrar el ciclo (última ciudad a primera)
        pheromones[path[-1], path[0]] += Q / path_length
        pheromones[path[0], path[-1]] += Q / path_length  # Simetría
    
    # Guardar historial para graficar convergencia
    iteration_history.append(iter)
    best_length_history.append(best_path_length)
    
    # Mostrar progreso cada 10 iteraciones
    if (iter+1) % 10 == 0:
        print(f"Iteración {iter+1}/{maxIter}, Mejor longitud: {best_path_length:.2f}")

# Mostrar resultados finales
print(f"\nMejor recorrido encontrado en la iteración {best_iteration+1}")
print(f"Longitud del mejor recorrido: {best_path_length:.2f}")

# Graficar la convergencia
plt.figure(figsize=(10, 6))
plt.plot(iteration_history, best_length_history)
plt.grid(True)
plt.title('Convergencia del algoritmo ACO')
plt.xlabel('Iteración')
plt.ylabel('Longitud del mejor recorrido')
plt.show()

# Visualizar el mejor recorrido
plot_path(cities, best_path)