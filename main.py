import ast
import csv
import numpy as np
from getFiles import *
from keras.models import Sequential
from keras.layers import Dense

# Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
with open('PosicionsEvaluacions.csv', 'r') as file:
    reader = csv.reader(file)
    
    first=True
    numEvaluacions = 1500000
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

"""
#Guardar los 'numEvaluacions' valores leidos en un csv para saber con que partidas hemos entrenado lar red
# Open the file in write mode
with open('TrainingPositions.csv', mode='w', newline='') as file:
    
    # Create a writer object
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['FEN', 'Evaluation'])
    
    # Write the data rows
    for item in TrainingTuples:
        writer.writerow([item[0], item[1]])

        
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
    eval = int(line[1])/float(1000)
    if eval > 1:
        eval = 1
    elif eval < -1:
        eval = -1

    outputs.append(eval)

outputs = np.array(outputs,dtype=float)
inputs = np.array(inputs)

print("Started training:")

# definir la red neuronal
model = Sequential()
model.add(Dense(64, input_shape=(64,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# compilar la red neuronal
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# entrenar la red neuronal
model.fit(inputs, outputs, epochs=500, batch_size=32, verbose=1)

# Guardar el modelo en un archivo
model.save('red_neuronal.h5')

# Guardar los pesos entrenados en un archivo separado
model.save_weights('pesos.h5')


print("Started evaluation:")
#EVALUAR
with open('PosicionsEvaluacions.csv', 'r') as file:
    reader = csv.reader(file)
    
    first=True
    numEvaluacions = 200000
    EvaluatingTuples = []
    i = 0

    # Iterar sobre cada fila del archivo
    for row in reader:
        if not first and i < numEvaluacions:
            for line in TrainingTuples:
                #if line[0] != row[0]:
                    # Obtener la cadena de caracteres y el campo 'value' como objeto literal de Python
                    value_literal = ast.literal_eval(row[1])
                    # Si el tipo es 'cp', imprimir la cadena de caracteres y el valor
                    if value_literal['type'] == 'cp':
                        EvaluatingTuples.append((row[0], value_literal['value']))
                        i += 1

        first = False

# convertir cada línea en una entrada numérica de 64 x 7 i una sortida de 1 elemetn
g = getFiles()
EvaluatingInputs = []
EvaluatingOutputs = []
for line in EvaluatingTuples:
    # convertir la cadena de 448 bits en una lista de 64 elementos de 7 bits
    binary = g.fenToBinaryAllInSquares(line[0])
    
    # Split the string into 64 groups of 7 digits
    groups = [binary[i:i+7] for i in range(0, len(binary), 7)]

    binary_numbers = [int(group, 2) for group in groups]

    array = np.array(binary_numbers, dtype=int)

    EvaluatingInputs.append(array)

    #Generate the output
    eval = int(line[1])/float(1000)
    if eval > 1:
        eval = 1
    elif eval < -1:
        eval = -1
    
    EvaluatingOutputs.append(eval)

EvaluatingOutputs = np.array(EvaluatingOutputs,dtype=float)
EvaluatingInputs = np.array(EvaluatingInputs)

# Evaluar la red neuronal
loss, accuracy = model.evaluate(EvaluatingInputs, EvaluatingOutputs)
print('Accuracy: %.2f' % (accuracy*100))

"""

"""
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