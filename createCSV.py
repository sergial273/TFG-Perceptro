import csv

# Abre el archivo CSV en modo de escritura y crea un objeto escritor
with open("PosicionsEvaluacions.csv", "w", newline="") as archivo:
    escritor = csv.writer(archivo)

    # Escribe una fila de encabezado
    escritor.writerow(["FEN", "eval"])