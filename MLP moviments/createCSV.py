import csv
import os

# Abre el archivo CSV en modo de escritura y crea un objeto escritor
with open(os.getcwd()+'\MLP moviments\PosicionsEvaluacions2.csv', "w", newline="") as archivo:
    escritor = csv.writer(archivo)

    # Escribe una fila de encabezado
    escritor.writerow(["FEN1","FEN","eval","Piece","Mate","textMate"])