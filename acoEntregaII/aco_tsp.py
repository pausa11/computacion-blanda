import tsplib95
import os
import random

# Ruta a tu carpeta con archivos .tsp
TSP_FOLDER = './tsplib-master/'

# Diccionarios para clasificar por tamaño y verificar si se pueden graficar
small_problems = []
medium_problems = []
large_problems = []

def is_graphable(problem):
    return problem.node_coords is not None

# Clasificar problemas válidos
for filename in os.listdir(TSP_FOLDER):
    if filename.endswith('.tsp'):
        filepath = os.path.join(TSP_FOLDER, filename)
        try:
            problem = tsplib95.load(filepath)
            if not is_graphable(problem):
                continue  # Ignorar problemas sin coordenadas
            dimension = problem.dimension

            if 10 <= dimension <= 20:
                small_problems.append(filename)
            elif 50 <= dimension <= 100:
                medium_problems.append(filename)
            elif dimension > 101:
                large_problems.append(filename)
        except:
            print(f"Error cargando {filename}, se omite.")

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

# Seleccionar 3 problemas válidos por categoría
problems = []
problems += get_random_problems(small_problems, 3)
problems += get_random_problems(medium_problems, 3)
problems += get_random_problems(large_problems, 3)

# Mostrar los problemas seleccionados
print("Problemas seleccionados:")
for fname, problem in problems:
    print(f"- {fname} ({problem.dimension} ciudades)")

