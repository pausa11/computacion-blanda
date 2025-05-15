import numpy as np
import matplotlib.pyplot as plt

# ——— Funciones de activación ———
def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

def softmax(z):
    # softmax estable
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ——— Pérdida y derivada ———
def log_loss(y_true, y_pred):
    eps = 1e-15
    p = np.clip(y_pred, eps, 1-eps)
    return -np.sum(y_true * np.log(p), axis=1)

# ——— Datos de entrenamiento y prueba ———
inputs = np.array([(0.0000, 0.3929), (0.5484, 0.7500), (0.0645, 0.5714), (0.5806, 0.5714),
(0.2258, 0.8929), (0.4839, 0.2500), (0.3226, 0.2143), (0.7742, 0.8214),
(0.4516, 0.5000), (0.4194, 0.0357), (0.4839, 0.2500), (0.3226, 0.7143),
(0.5806, 0.5000), (0.5484, 0.1071), (0.6129, 0.6429), (0.6774, 0.1786),
(0.2258, 0.8214), (0.7419, 0.1429), (0.6452, 1.0000), (0.8387, 0.2500),
(0.9677, 0.3214), (0.3226, 0.4643), (0.3871, 0.5357), (0.3548, 0.1429),
(0.3548, 0.6429), (0.1935, 0.4643), (0.4516, 0.3929), (0.4839, 0.6071),
(0.6129, 0.6786), (0.2258, 0.6071), (0.5161, 0.3214), (0.5484, 0.6786),
(0.3871, 0.8571), (0.6452, 0.6071), (0.1935, 0.3929), (0.6452, 0.3929),
(0.6774, 0.4643), (0.3226, 0.2857), (0.7419, 0.7143), (0.7419, 0.3214),
(1.0000, 0.3929), (0.8065, 0.3929), (0.1935, 0.5000), (0.1613, 0.8214),
(0.2903, 0.9286), (0.3548, 0.0000), (0.2903, 0.6786), (0.5484, 0.9643),
(0.4194, 0.1786), (0.2581, 0.2500), (0.3226, 0.7143), (0.5161, 0.3929),
(0.2903, 0.6429), (0.5484, 0.9286), (0.2581, 0.3214), (0.0968, 0.5000),
(0.6129, 0.7857), (0.0968, 0.3214), (0.6452, 0.9286), (0.8065, 0.7500)])

purple = np.array([1,0,0])
orange = np.array([0,1,0])
green  = np.array([0,0,1])
targets = [purple, orange, purple, orange, green, purple, purple, green,
            orange, purple, purple, green, orange, purple, orange, purple,
              green, purple, green, purple, purple, orange, orange, purple, 
              orange, purple, orange, orange, orange, green, orange, orange, 
              green, orange, purple, orange, orange, purple, orange, orange, 
              purple, orange, green, green, green, purple, green, green, 
              purple, purple, green, orange, green, green, purple, purple,
              green, purple, green, green]

test_inputs = np.array([(0.0000, 0.3929), (0.0645, 0.5714), (0.0968, 0.3214),
(0.0968, 0.5000), (0.2581, 0.3214), (0.1935, 0.4643), (0.2581, 0.2500),
(0.1935, 0.3929), (0.3226, 0.2143), (0.4839, 0.2500), (0.3226, 0.4643),
(0.3871, 0.5357), (0.3548, 0.6429), (0.4516, 0.5000), (0.4516, 0.3929),
(0.5161, 0.3929), (0.5484, 0.7500), (0.6129, 0.6786), (0.5161, 0.3214),
(0.5484, 0.6786), (0.1935, 0.5000), (0.2258, 0.6071), (0.3226, 0.7143),
(0.2903, 0.6786), (0.3226, 0.7143), (0.2258, 0.8214), (0.2903, 0.6429),
(0.6129, 0.7857), (0.7742, 0.8214), (0.8065, 0.7500)])

test_targets = np.array([purple, purple, purple, purple, purple, purple, purple, purple,
purple, purple, orange, orange, orange, orange, orange, orange,
orange, orange, orange, orange, green, green, green, green, green, green, green,
green, green, green])

# ——— Hiperparámetros mejorados ———
n_input = 2
n_hidden = 4  
n_output = 3
lr_initial = 0.15 
lr_decay = 0.9999 
iterations = 10000
momentum = 0.9  

# ——— Inicialización de pesos y sesgos mejorada (Xavier/Glorot) ———
rng = np.random.default_rng(42)
# Xavier initialization para mejor convergencia
W1 = rng.normal(0, np.sqrt(2/(n_input + n_hidden)), size=(n_hidden, n_input))
b1 = np.zeros((1, n_hidden))
W2 = rng.normal(0, np.sqrt(2/(n_hidden + n_output)), size=(n_output, n_hidden))
b2 = np.zeros((1, n_output))

# Para momentum
vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)

# ——— Entrenamiento ———
loss_history = []
N = inputs.shape[0]
lr = lr_initial

# Convertir lista de targets a matriz numpy para cálculos más eficientes
targets = np.array(targets)

# Normalizamos los datos de entrada para mejor rendimiento
mean = np.mean(inputs, axis=0)
std = np.std(inputs, axis=0)
inputs_normalized = (inputs - mean) / (std + 1e-8)
test_inputs_normalized = (test_inputs - mean) / (std + 1e-8)

for iter in range(iterations):
    # Forward
    Z1 = inputs_normalized.dot(W1.T) + b1  # (N, n_hidden)
    A1 = relu(Z1)                          # (N, n_hidden)
    Z2 = A1.dot(W2.T) + b2                 # (N, n_output)
    A2 = softmax(Z2)                       # (N, n_output)

    # Pérdida
    loss = np.mean(log_loss(targets, A2))
    loss_history.append(loss)

    # Backprop con gradiente más preciso
    dZ2 = (A2 - targets) / N                   # (N, n_output)
    dW2 = dZ2.T.dot(A1)                        # (n_output, n_hidden)
    db2 = np.sum(dZ2, axis=0, keepdims=True)   # (1, n_output)

    dA1 = dZ2.dot(W2)                          # (N, n_hidden)
    dZ1 = dA1 * drelu(Z1)                      # (N, n_hidden)
    dW1 = dZ1.T.dot(inputs_normalized)         # (n_hidden, n_input)
    db1 = np.sum(dZ1, axis=0, keepdims=True)   # (1, n_hidden)

    # Actualización con momentum y regularización
    vW2 = momentum * vW2 - lr * dW2
    vb2 = momentum * vb2 - lr * db2
    vW1 = momentum * vW1 - lr * dW1
    vb1 = momentum * vb1 - lr * db1
    
    W2 += vW2
    b2 += vb2
    W1 += vW1
    b1 += vb1
    
    # Learning rate decay
    lr *= lr_decay

    # (Opcional) imprimir cada 1000 epochs
    if iter % 1000 == 0:
        print(f"Epoch {iter}, loss={loss:.4f}, lr={lr:.6f}")

# ——— Evaluación en test ———
Z1_test = test_inputs_normalized.dot(W1.T) + b1
A1_test = relu(Z1_test)
Z2_test = A1_test.dot(W2.T) + b2
A2_test = softmax(Z2_test)

pred_labels = np.argmax(A2_test, axis=1)
true_labels = np.argmax(test_targets, axis=1)
accuracy = np.mean(pred_labels == true_labels)
print(f"Precisión en test: {accuracy*100:.2f}%")

# Colores usados
color_map = {
    0: 'purple',
    1: 'orange',
    2: 'green'
}

# Etiquetas para leyenda
label_map = {
    0: "Purple (clase 0)",
    1: "Orange (clase 1)",
    2: "Green (clase 2)"
}

# Gráfica de clasificación en test:
plt.figure(figsize=(10,8))
for cls in [0,1,2]:
    # máscara para verdaderos
    mask_real = np.argmax(test_targets, axis=1) == cls
    # máscara para predichos
    mask_pred = pred_labels == cls

    # puntos reales (huecos, color gris)
    plt.scatter(test_inputs[mask_real,0],
                test_inputs[mask_real,1],
                facecolors='none',
                edgecolors='gray',
                s=100,
                label=f'Real {label_map[cls]}')

    # puntos predichos (color de la clase, marcador lleno)
    plt.scatter(test_inputs[mask_pred,0],
                test_inputs[mask_pred,1],
                c=color_map[cls],
                s=60,
                marker='o',
                label=f'Pred {label_map[cls]}')

plt.title("Clasificación en datos de prueba", fontsize=14)
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Gráfica de la pérdida:
plt.figure(figsize=(8,6))
plt.plot(loss_history)
plt.title("Loss durante el entrenamiento", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True)
plt.show()

# Visualización de las fronteras de decisión
plt.figure(figsize=(10, 8))

# Crear una malla de puntos
h = 0.01
x_min, x_max = -0.1, 1.1
y_min, y_max = -0.1, 1.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Aplanar la malla en un arreglo 2D de puntos
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Normalizar puntos de la malla usando la misma normalización de datos
grid_points_normalized = (grid_points - mean) / (std + 1e-8)

# Predicción de la clase para cada punto de la malla
Z1_grid = grid_points_normalized.dot(W1.T) + b1
A1_grid = relu(Z1_grid)
Z2_grid = A1_grid.dot(W2.T) + b2
A2_grid = softmax(Z2_grid)
grid_preds = np.argmax(A2_grid, axis=1)

# Dar forma a las predicciones para mostrar el mapa de fronteras
grid_preds = grid_preds.reshape(xx.shape)

# Crear un mapa de colores personalizado para las regiones
custom_cmap = plt.cm.colors.ListedColormap(['#AA20FF', '#FF7500', '#20BF50'])

# Graficar regiones de decisión
plt.contourf(xx, yy, grid_preds, alpha=0.3, cmap=custom_cmap)

# Graficar puntos de test
for cls in [0,1,2]:
    mask_real = np.argmax(test_targets, axis=1) == cls
    plt.scatter(test_inputs[mask_real,0],
                test_inputs[mask_real,1],
                c=color_map[cls],
                edgecolors='k',
                marker='o',
                s=80,
                label=f'{label_map[cls]}')

plt.title('Fronteras de Decisión y Puntos de Prueba', fontsize=14)
plt.xlabel('X1', fontsize=12)
plt.ylabel('X2', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()