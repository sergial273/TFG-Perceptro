import ast
import csv
from getFiles import *
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

def getTuples(numEvaluacions,numTests):
    # Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
    with open('PosicionsEvaluacions.csv', 'r') as file:
        reader = csv.reader(file)
        
        #Obtener los valores para el entrenamiento
        first=True
        TrainingTuples = []
        i = 0
        
        TestTuples = []
        o = 0

        trainingfen = []
        # Iterar sobre cada fila del archivo
        for row in reader:
            if not first and i < numEvaluacions:
                # Obtener la cadena de caracteres y el campo 'value' como objeto literal de Python
                value_literal = ast.literal_eval(row[1])
                # Si el tipo es 'cp', imprimir la cadena de caracteres y el valor
                if value_literal['type'] == 'cp':
                    TrainingTuples.append((row[0], value_literal['value']))
                    trainingfen.append(row[0])
                    i += 1

            elif not first and o < numTests:
                if row[0] not in trainingfen:
                    # Obtener la cadena de caracteres y el campo 'value' como objeto literal de Python
                    value_literal = ast.literal_eval(row[1])
                    # Si el tipo es 'cp', imprimir la cadena de caracteres y el valor
                    if value_literal['type'] == 'cp':
                        TestTuples.append((row[0], value_literal['value']))
                        o += 1

            first = False
    
    return TrainingTuples, TestTuples


def convertTuple(Tuples):

    # convertir cada línea en una entrada numérica de 64 x 7 i una sortida de 1 elemetn
    g = getFiles()
    inputs = []
    outputs = []
    for line in Tuples:
        # convertir la cadena de 448 bits en una lista de 64 elementos de 7 bits
        binary = g.fenToBinaryAllInSquares(line[0])
        
        # Split the string into 64 groups of 7 digits
        groups = [binary[i:i+7] for i in range(0, len(binary), 7)]

        binary_numbers = [int(group, 2) for group in groups]

        arr = np.array(binary_numbers, dtype=int)

        inputs.append(arr)

        
        #Generate the output
        eval = int(line[1])/1000

        signe = 0 if eval >= 0 else 1

        eval = abs(eval)
        output_bin = []
        interv1 = 1 if eval < 0.5 else 0
        interv2 = 1 if (eval >= 0.5  and eval < 1.5) else 0
        interv3 = 1 if (eval >= 1.5  and eval < 2.5) else 0
        interv4 = 1 if (eval >= 2.5  and eval < 3.5) else 0
        interv5 = 1 if (eval >= 3.5  and eval < 4.5) else 0
        interv6 = 1 if (eval >= 4.5) else 0

        output_bin.append(signe)
        output_bin.append(interv1)
        output_bin.append(interv2)
        output_bin.append(interv3)
        output_bin.append(interv4)
        output_bin.append(interv5)
        output_bin.append(interv6)

        arr1 = np.array(output_bin, dtype=float)
        
        outputs.append(arr1)

    outputs = np.array(outputs)
    inputs = np.array(inputs)
    
    return inputs,outputs



TrainingTuples,TestTuples = getTuples(numEvaluacions=100000,numTests=10000)

inputsTraining,outputsTraining = convertTuple(TrainingTuples)
inputsTest,outputsTest = convertTuple(TestTuples)

#normalitzar la info
inputsTraining = inputsTraining.astype('float32') / 127
inputsTest = inputsTest.astype('float32') / 127


MLP = Sequential()
MLP.add(InputLayer(input_shape=(64, ))) # input layer
#MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
#MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
#MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
#MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
MLP.add(Dense(7, activation='sigmoid')) # output layer

# summary
MLP.summary()

# optimization
MLP.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# train (fit)
MLP.fit(inputsTraining, outputsTraining, 
        epochs=1, batch_size=128) #was 20 epochs

