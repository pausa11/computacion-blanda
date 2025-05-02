import requests
from bs4 import BeautifulSoup
import csv
import json

# URL del sitio
url = "https://quindicolor.com/Productos/index"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Lista para guardar todos los productos
todos_los_productos = []

# Iteramos sobre los IDs de categorías (1 a 5)
for i in range(1, 6):
    categoria_id = f"productos{i}"
    categoria_div = soup.find("div", {"id": categoria_id})

    if categoria_div:
        items = categoria_div.find_all("div", class_="col-lg-3 col-6")

        for item in items:
            nombre = item.find("div", class_="producto_texto").get_text(strip=True)
            imagen = item.find("img")["src"]
            imagen = imagen if imagen.startswith("http") else f"https://quindicolor.com{imagen}"
            producto = {
                "categoria": f"categoria_{i}",
                "nombre": nombre,
                "imagen": imagen
            }
            todos_los_productos.append(producto)

# Guardar en CSV
with open("productos_quindicolor.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["categoria", "nombre", "imagen"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for producto in todos_los_productos:
        writer.writerow(producto)

# Guardar en JSON
with open("productos_quindicolor.json", "w", encoding="utf-8") as jsonfile:
    json.dump(todos_los_productos, jsonfile, indent=4, ensure_ascii=False)

print("✅ Archivos guardados como productos_quindicolor.csv y productos_quindicolor.json")
