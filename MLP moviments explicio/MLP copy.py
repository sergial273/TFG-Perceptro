import ast
import csv
import time
from getFiles import *
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout
from concurrent.futures import ProcessPoolExecutor


def eval6(eval,mate):
    output_bin = []

    #calcculating evaluation ranges
    interv1pos = 1 if eval < 0.25 else 0
    interv2pos = 1 if (eval >= 0.25  and eval < 0.5) else 0
    interv3pos = 1 if (eval >= 0.5  and eval < 0.75) else 0
    interv4pos = 1 if (eval >= 0.75  and eval < 1) else 0
    interv5pos = 1 if (eval >= 1  and eval < 1.25) else 0
    interv6pos = 1 if (eval >= 1.25  and eval < 1.5) else 0
    interv7pos = 1 if (eval >= 1.5  and eval < 2.5) else 0
    interv8pos = 1 if (eval >= 2.5  and eval < 3.5) else 0
    interv9pos = 1 if (eval >= 3.5  and eval < 4.5) else 0
    interv10pos = 1 if (eval >= 4.5) else 0
    interv1neg = 1 if (eval > -0.25 and eval<0) else 0
    interv2neg = 1 if (eval <= -0.25  and eval > -0.5) else 0
    interv3neg = 1 if (eval <= -0.5  and eval > -0.75) else 0
    interv4neg = 1 if (eval <= -0.75  and eval > -1) else 0
    interv5neg = 1 if (eval <= -1  and eval > -1.25) else 0
    interv6neg = 1 if (eval <= -1.25  and eval > -1.5) else 0
    interv7neg = 1 if (eval <= -1.5  and eval > -2.5) else 0
    interv8neg = 1 if (eval <= -2.5  and eval > -3.5) else 0
    interv9neg = 1 if (eval <= -3.5  and eval > -4.5) else 0
    interv10neg = 1 if (eval <= -4.5) else 0

    output_bin.append(interv1pos)
    output_bin.append(interv2pos)
    output_bin.append(interv3pos)
    output_bin.append(interv4pos)
    output_bin.append(interv5pos)
    output_bin.append(interv6pos)
    output_bin.append(interv7pos)
    output_bin.append(interv8pos)
    output_bin.append(interv9pos)
    output_bin.append(interv10pos)
    output_bin.append(interv1neg)
    output_bin.append(interv2neg)
    output_bin.append(interv3neg)
    output_bin.append(interv4neg)
    output_bin.append(interv5neg)
    output_bin.append(interv6neg)
    output_bin.append(interv7neg)
    output_bin.append(interv8neg)
    output_bin.append(interv9neg)
    output_bin.append(interv10neg)

    #adding the mate bit
    output_bin.append(mate)

    #checking if piece attacks new ones
    arr1 = np.array(output_bin, dtype=float)
    
    return arr1

def xarxa1():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(128, ))) # input layer
    #MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    #MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa2():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(128, ))) # input layer
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    #MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa3():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(128, ))) # input layer
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa4():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(128, ))) # input layer
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa5():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(128, ))) # input layer
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa6():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(128, ))) # input layer
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa7():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(128, ))) # input layer
    MLP.add(Dense(512, activation='sigmoid')) # hidden layer 1
    #MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa2Dropout():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(128, ))) # input layer
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    MLP.add(Dropout(0.45)) # hidden layer 1
    MLP.add(Dense(64, activation='sigmoid')) # hidden layer 1
    MLP.add(Dropout(0.45)) # hidden layer 1
    #MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def getTuples(numEvaluacions,numTests):
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
                TrainingTuples.append((row[0], row[1], row[2], row[3], row[4]))
                i += 1
            first = False

    # Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
    with open(os.getcwd()+'\MLP moviments\PosicionsTest2.csv', 'r') as file:
        reader = csv.reader(file)

        TestTuples = []
        o = 0
    
        for row in reader:    
            if o < numTests:
                TestTuples.append((row[0], row[1], row[2], row[3], row[4]))
                o += 1
    
    return TrainingTuples, TestTuples

def convertTuple(Tuples, func):
        
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

        #Generate the output

        eval = int(line[2])/1000

        arr1 = func(eval,line[4])
        
        outputs.append(arr1)

    outputs = np.array(outputs)
    inputs = np.array(inputs)
    
    res =[inputs, outputs]
    return res

def process_files_concurrently(Tuples, eval):
    AllTuples = tuple(Tuples[i:i + int(numEvaluacions/4)] for i in range(0, len(Tuples), int(numEvaluacions/4)))

    # Crea un ProcessPoolExecutor con 4 procesos
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Crea una lista de tareas a ejecutar con executor.submit()
        tasks = [executor.submit(convertTuple, tuple, eval) for tuple in AllTuples]
        # Espera a que todas las tareas se completen y devuelve los resultados
        inputs = []
        outputs = []
        for task in tasks:
            a = task.result()
            inputs.append(a[0])
            outputs.append(a[1])
            
        return inputs,outputs

start = time.time()
if __name__ == '__main__':
    evalutionFunctions = [(eval6,21)]
    differentNetworks = [xarxa2] #[xarxa1,xarxa2,xarxa3,xarxa4,xarxa5,xarxa6,xarxa7,xarxa2Dropout]
    listOptimizers = ['Adam'] #['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam','Ftrl']
    numEvaluacions = 2000000
    numTests = 100000

    TrainingTuples,TestTuples = getTuples(numEvaluacions,numTests)


    inputsTraining,outputsTraining = process_files_concurrently(TrainingTuples, eval6)

    inputsTest,outputsTest = process_files_concurrently(TestTuples, eval6)

    #normalitzar la info
    inputsTraining = inputsTraining.astype('float32') / 127
    inputsTest = inputsTest.astype('float32') / 127


    for func in evalutionFunctions:
        for xarxa in differentNetworks:
            for optimizer in listOptimizers:
        
                MLP = xarxa()

                # summary
                MLP.summary()

                # optimization
                MLP.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])

                # train (fit)
                history = MLP.fit(inputsTraining, outputsTraining, 
                        epochs=20, batch_size=128) #was 20 epochs and 128 batch_size

                train_accuracy = history.history['accuracy'][-1]
                train_loss = history.history['loss'][-1]
                
                # evaluate performance
                test_loss, test_acc = MLP.evaluate(inputsTest, outputsTest,
                                                batch_size=128,
                                                verbose=0)

                with open(os.getcwd()+'\MLP moviments\ValorsTests.txt', mode='a') as archivo:
                    archivo.write('Xarxa, Funcio eval, Optimitzador: '+str(xarxa)+', '+str(func)+', '+str(optimizer)+'\n')
                    archivo.write('Train acc '+str(train_accuracy)+'\n')
                    archivo.write('Test acc '+str(test_acc)+'\n')
                    archivo.write("-" * 50+'\n')

    end = time.time()

    print(end-start)


 