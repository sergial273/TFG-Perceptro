import ast
import csv
from getFiles import *
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

# Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
with open('PosicionsEvaluacions.csv', 'r') as file:
    reader = csv.reader(file)
    
    first=True
    numEvaluacions = 2000000
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


# convertir cada línea en una entrada numérica de 64 x 7 i una sortida de 1 elemetn
g = getFiles()
inputs = []
outputs = []
for line in TrainingTuples:
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

#normalitzar la info
inputs = inputs.astype('float32') / 127


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
MLP.fit(inputs, outputs, 
        epochs=1, batch_size=128) #was 20 epochs




#abrir el archivo i encontrar 100000 partidas que no esten en el set de entrenamiento
with open('PosicionsEvaluacions.csv', 'r') as file:
    reader = csv.reader(file)
    
    first=True
    numTests = 100000
    TestTuples = []
    i = 0

    # Iterar sobre cada fila del archivo
    for row in reader:
        if not first and i < numTests:
            for trainPosition in TrainingTuples:
                if row[0] != trainPosition[0]:
                    # Obtener la cadena de caracteres y el campo 'value' como objeto literal de Python
                    value_literal = ast.literal_eval(row[1])
                    # Si el tipo es 'cp', imprimir la cadena de caracteres y el valor
                    if value_literal['type'] == 'cp':
                        TestTuples.append((row[0], value_literal['value']))
                        i += 1

        first = False


# convertir cada línea en una entrada numérica de 64 x 7 i una sortida de 1 elemet
g = getFiles()
inputsTest = []
outputsTest = []
for line in TrainingTuples:
    
    # convertir la cadena de 448 bits en una lista de 64 elementos de 7 bits
    binary = g.fenToBinaryAllInSquares(line[0])
    
    # Split the string into 64 groups of 7 digits
    groups = [binary[i:i+7] for i in range(0, len(binary), 7)]

    binary_numbers = [int(group, 2) for group in groups]

    arrTest = np.array(binary_numbers, dtype=int)

    inputsTest.append(arrTest)

    
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

    arr1Test = np.array(output_bin, dtype=float)
    
    outputsTest.append(arr1Test)

outputsTest = np.array(outputsTest)
inputsTest = np.array(inputsTest)

#normalitzar la info
inputsTest = inputsTest.astype('float32') / 127


# evaluate performance
test_loss, test_acc = MLP.evaluate(inputsTest, outputsTest,
                                   batch_size=128,
                                   verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


print(len(TestTuples))
""""
digit = np.reshape(digit, (-1, 784))
digit = digit.astype('float32') / 255

MLP.predict(digit, verbose=0)

# Guardar el modelo en un archivo
model.save('red_neuronal.h5')

# Guardar los pesos entrenados en un archivo separado
model.save_weights('pesos.h5')


# Cargar el modelo desde el archivo
from keras.models import load_model
modelo = load_model('red_neuronal.h5')

# Cargar los pesos entrenados desde el archivo
modelo.load_weights('pesos.h5')
"""