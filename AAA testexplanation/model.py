# Importar las librerías necesarias
import csv
from keras.models import load_model
from getFiles import *
import numpy as np

def getTuples(numEvaluacions):
    # Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
    with open(os.getcwd()+'\MLP moviments\PosicionsEvaluacions2.csv', 'r') as file:
        reader = csv.reader(file)
        
        #Obtener los valores para el entrenamiento
        first=True
        TrainingTuples = []
        i = 0
        
        # Iterar sobre cada fila del archivo
        for row in reader:
            if not first and i < numEvaluacions:     
                TrainingTuples.append((row[0], row[1]))
                i += 1
            first = False

    
    return TrainingTuples

def convertTuple(Tuples):
        
    g = getFiles()
    inputs = []
    outputs = []
    for line in Tuples:
        # convertir la cadena de 448 bits en una lista de 64 elements de 7 bits
        binary = g.fenToBinaryAllInSquares(line[0])
        
        # Split the string into 64 groups of 7 digits
        groups = [binary[i:i+7] for i in range(0, len(binary), 7)]

        binary_numbers = [int(group, 2) for group in groups]

        arr = np.array(binary_numbers, dtype=int)

        #fer el mateix amb el segon fen
        binary = g.fenToBinaryAllInSquares(line[1])
        
        # Split the string into 64 groups of 7 digits
        groups = [binary[i:i+7] for i in range(0, len(binary), 7)]

        binary_numbers = [int(group, 2) for group in groups]

        arr1 = np.array(binary_numbers, dtype=int)

        arr = np.concatenate((arr, arr1))
        inputs.append(arr)

    inputs = np.array(inputs)
    
    return inputs


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
#input = ConvertToInput("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
TrainingTuples = getTuples(1)
input = convertTuple(TrainingTuples)
input = input.astype('float32') / 127


prediction = model.predict(input)
array2 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

for predict in prediction:
    # Imprimir el resultado
    #contar el nombre de bits que es pot tenir com a maxim a 1, si es més petit rescalcular-ho tot i fer un array nou
    if np.array_equal(predict, array2):
        print("iguals")
    else: print("a")