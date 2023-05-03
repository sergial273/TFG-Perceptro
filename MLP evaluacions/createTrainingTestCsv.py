import ast
import csv
import sys
from getFiles import *

numEvaluacions = 2000000
numTests = 100000


# Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
with open(os.getcwd()+'\MLP evaluacions\PosicionsEvaluacions.csv', 'r') as file:
    reader = csv.reader(file)
    
    #Obtener los valores para el entrenamiento
    first=True
    i = 0  
    o = 0

    trainingfen = []
    # Iterar sobre cada fila del archivo
    for row in reader:
        
        if not first and i < numEvaluacions:
            # Obtener la cadena de caracteres y el campo 'value' como objeto literal de Python
            value_literal = ast.literal_eval(row[1])
            # Si el tipo es 'cp', imprimir la cadena de caracteres y el valor
            if value_literal['type'] == 'cp':
                trainingfen.append(row[0])
                i += 1

        elif not first and o < numTests:
            if row[0] not in trainingfen:
                # Obtener la cadena de caracteres y el campo 'value' como objeto literal de Python
                value_literal = ast.literal_eval(row[1])
                # Si el tipo es 'cp', imprimir la cadena de caracteres y el valor
                if value_literal['type'] == 'cp':
                    
                    # Abre el archivo CSV en modo de escritura y crea un objeto escritor
                    with open("PosicionsTest.csv", "a", newline="") as archivo:
                        escritor = csv.writer(archivo)

                        # Escribe una fila de encabezado
                        escritor.writerow([row[0], value_literal['value']])
                    o += 1

        first = False