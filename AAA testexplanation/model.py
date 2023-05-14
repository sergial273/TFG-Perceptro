# Importar las librerías necesarias
from keras.models import load_model
from getFiles import *
import numpy as np


def ConvertToInput(fen1, fen2):
    inputs = []
    g = getFiles()
    # convertir la cadena de 448 bits en una lista de 64 elements de 7 bits
    binary = g.fenToBinaryAllInSquares(fen1)
    
    # Split the string into 64 groups of 7 digits
    groups = [binary[i:i+7] for i in range(0, len(binary), 7)]

    binary_numbers = [int(group, 2) for group in groups]

    arr = np.array(binary_numbers, dtype=int)

    #fer el mateix amb el segon fen
    binary = g.fenToBinaryAllInSquares(fen2)
    
    # Split the string into 64 groups of 7 digits
    groups = [binary[i:i+7] for i in range(0, len(binary), 7)]

    binary_numbers = [int(group, 2) for group in groups]

    arr1 = np.array(binary_numbers, dtype=int)

    arr = np.concatenate((arr, arr1))
    inputs.append(arr)
    inputs = np.array(inputs)
    
    return inputs

# Cargar el modelo
model = load_model('./model.h5')

# Hacer la predicción
input = ConvertToInput("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
input = input.astype('float32') / 127
print(input)
prediction = model.predict(input)

# Imprimir el resultado
print(prediction)