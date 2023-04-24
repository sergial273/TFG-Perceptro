import ast
import csv
from getFiles import *

# Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
with open('PosicionsEvaluacions.csv', 'r') as file:
    reader = csv.reader(file)
    
    first=True
    numEvaluacions = 2
    TrainingTuples = []
    i = 0

    # Iterar sobre cada fila del archivo
    for row in reader:
        if not first and i < numEvaluacions:
            # Obtener la cadena de caracteres y el campo 'value' como objeto literal de Python
            value_literal = ast.literal_eval(row[1])
            # Si el tipo es 'cp', imprimir la cadena de caracteres y el valor
            if value_literal['type'] == 'cp':
                TrainingTuples.append((row[0], value_literal['value']))
                i += 1

        first = False

print(TrainingTuples)