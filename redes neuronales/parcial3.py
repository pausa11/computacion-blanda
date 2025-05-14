import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
def cross_entropy(y_true, y_pred):
    eps = 1e-15
    p = np.clip(y_pred, eps, 1-eps)
    return -np.sum(y_true * np.log(p), axis=1)

# ——— Datos de entrenamiento y prueba ———
inputs = np.array([(0.0000, 0.3929), (0.5484, 0.7500), (0.0645, 0.5714), (0.5806,
0.5714),
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

test_targets = np.array([purple, purple, purple, purple, purple, purple, purple,
purple,
purple, purple, orange, orange, orange, orange, orange, orange,
orange, orange,
orange, orange, green, green, green, green, green, green, green,
green, green, green])

# ——— Hiperparámetros ———
n_input    = 2
n_hidden   = 40        
n_output   = 3
lr         = 0.1
epochs     = 10000

# ——— Inicialización de pesos y sesgos ———
rng = np.random.default_rng(42)
W1 = rng.normal(scale=0.1, size=(n_hidden, n_input))
b1 = np.zeros((1, n_hidden))
W2 = rng.normal(scale=0.1, size=(n_output, n_hidden))
b2 = np.zeros((1, n_output))

# ——— Entrenamiento ———
loss_history = []
N = inputs.shape[0]

for epoch in range(epochs):
    # Forward
    Z1 = inputs.dot(W1.T) + b1     # (N, n_hidden)
    A1 = relu(Z1)                  # (N, n_hidden)
    Z2 = A1.dot(W2.T) + b2         # (N, n_output)
    A2 = softmax(Z2)               # (N, n_output)

    # Pérdida
    loss = np.mean(cross_entropy(targets, A2))
    loss_history.append(loss)

    # Backprop
    dZ2 = (A2 - targets) / N                   # (N, n_output)
    dW2 = dZ2.T.dot(A1)                        # (n_output, n_hidden)
    db2 = np.sum(dZ2, axis=0, keepdims=True)   # (1, n_output)

    dA1 = dZ2.dot(W2)                          # (N, n_hidden)
    dZ1 = dA1 * drelu(Z1)                      # (N, n_hidden)
    dW1 = dZ1.T.dot(inputs)                    # (n_hidden, n_input)
    db1 = np.sum(dZ1, axis=0, keepdims=True)   # (1, n_hidden)

    # Actualización
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # (Opcional) imprimir cada 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, loss={loss:.4f}")

# ——— Evaluación en test ———
Z1_test = test_inputs.dot(W1.T) + b1
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

# Obtener predicciones como índices
pred_labels = np.argmax(A2_test, axis=1)

# Crear colores para cada punto según la clase predicha
colors = [color_map[label] for label in pred_labels]

# Graficar
plt.figure(figsize=(7,6))
plt.scatter(test_inputs[:,0], test_inputs[:,1], c=colors, s=60, edgecolors='k')
plt.title("Clasificación en datos de prueba")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)

# Agregar leyenda
legend_elements = [mpatches.Patch(color=clr, label=label_map[idx]) for idx, clr in color_map.items()]
plt.legend(handles=legend_elements)
plt.show()


# ——— Gráfica de la pérdida ———
plt.plot(loss_history)
plt.title("Loss durante el entrenamiento")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()