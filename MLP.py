import ast
import csv
from getFiles import *
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout

def eval1(eval):
    output_bin = []

    interv1pos = 1 if eval < 0.25 else 0
    interv2pos = 1 if (eval >= 0.25  and eval < 0.5) else 0
    interv3pos = 1 if (eval >= 0.5  and eval < 1) else 0
    interv4pos = 1 if (eval >= 1  and eval < 1.5) else 0
    interv5pos = 1 if (eval >= 1.5  and eval < 2.5) else 0
    interv6pos = 1 if (eval >= 2.5  and eval < 3.5) else 0
    interv7pos = 1 if (eval >= 3.5  and eval < 4.5) else 0
    interv8pos = 1 if (eval >= 4.5) else 0
    interv1neg = 1 if (eval > -0.25 and eval<0) else 0
    interv2neg = 1 if (eval <= -0.25  and eval > -0.5) else 0
    interv3neg = 1 if (eval <= -0.5  and eval > -1) else 0
    interv4neg = 1 if (eval <= -1  and eval > -1.5) else 0
    interv5neg = 1 if (eval <= -1.5  and eval > -2.5) else 0
    interv6neg = 1 if (eval <= -2.5  and eval > -3.5) else 0
    interv7neg = 1 if (eval <= -3.5  and eval > -4.5) else 0
    interv8neg = 1 if (eval <= -4.5) else 0

    output_bin.append(interv1pos)
    output_bin.append(interv2pos)
    output_bin.append(interv3pos)
    output_bin.append(interv4pos)
    output_bin.append(interv5pos)
    output_bin.append(interv6pos)
    output_bin.append(interv7pos)
    output_bin.append(interv8pos)
    output_bin.append(interv1neg)
    output_bin.append(interv2neg)
    output_bin.append(interv3neg)
    output_bin.append(interv4neg)
    output_bin.append(interv5neg)
    output_bin.append(interv6neg)
    output_bin.append(interv7neg)
    output_bin.append(interv8neg)

    arr1 = np.array(output_bin, dtype=float)
    
    return arr1

def eval2(eval):
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
    
    return arr1

def eval3(eval):
    output_bin = []

    zero = 1 if eval == 0 else 0
    interv1pos = 1 if (eval < 0.25 and eval>0) else 0
    interv2pos = 1 if (eval >= 0.25  and eval < 0.5) else 0
    interv3pos = 1 if (eval >= 0.5  and eval < 1) else 0
    interv4pos = 1 if (eval >= 1  and eval < 1.5) else 0
    interv5pos = 1 if (eval >= 1.5  and eval < 2.5) else 0
    interv6pos = 1 if (eval >= 2.5  and eval < 3.5) else 0
    interv7pos = 1 if (eval >= 3.5  and eval < 4.5) else 0
    interv8pos = 1 if (eval >= 4.5) else 0
    interv1neg = 1 if (eval > -0.25 and eval<0) else 0
    interv2neg = 1 if (eval <= -0.25  and eval > -0.5) else 0
    interv3neg = 1 if (eval <= -0.5  and eval > -1) else 0
    interv4neg = 1 if (eval <= -1  and eval > -1.5) else 0
    interv5neg = 1 if (eval <= -1.5  and eval > -2.5) else 0
    interv6neg = 1 if (eval <= -2.5  and eval > -3.5) else 0
    interv7neg = 1 if (eval <= -3.5  and eval > -4.5) else 0
    interv8neg = 1 if (eval <= -4.5) else 0

    output_bin.append(zero)
    output_bin.append(interv1pos)
    output_bin.append(interv2pos)
    output_bin.append(interv3pos)
    output_bin.append(interv4pos)
    output_bin.append(interv5pos)
    output_bin.append(interv6pos)
    output_bin.append(interv7pos)
    output_bin.append(interv8pos)
    output_bin.append(interv1neg)
    output_bin.append(interv2neg)
    output_bin.append(interv3neg)
    output_bin.append(interv4neg)
    output_bin.append(interv5neg)
    output_bin.append(interv6neg)
    output_bin.append(interv7neg)
    output_bin.append(interv8neg)

    arr1 = np.array(output_bin, dtype=float)
    
    return arr1

def eval4(eval):
    output_bin = []

    signe = 0 if eval >= 0 else 1
    zero = 1 if eval == 0 else 0
    interv1pos = 1 if (eval < 0.25 and eval>0) else 0
    interv2pos = 1 if (eval >= 0.25  and eval < 0.5) else 0
    interv3pos = 1 if (eval >= 0.5  and eval < 1) else 0
    interv4pos = 1 if (eval >= 1  and eval < 1.5) else 0
    interv5pos = 1 if (eval >= 1.5  and eval < 2.5) else 0
    interv6pos = 1 if (eval >= 2.5  and eval < 3.5) else 0
    interv7pos = 1 if (eval >= 3.5  and eval < 4.5) else 0
    interv8pos = 1 if (eval >= 4.5) else 0
    interv1neg = 1 if (eval > -0.25 and eval<0) else 0
    interv2neg = 1 if (eval <= -0.25  and eval > -0.5) else 0
    interv3neg = 1 if (eval <= -0.5  and eval > -1) else 0
    interv4neg = 1 if (eval <= -1  and eval > -1.5) else 0
    interv5neg = 1 if (eval <= -1.5  and eval > -2.5) else 0
    interv6neg = 1 if (eval <= -2.5  and eval > -3.5) else 0
    interv7neg = 1 if (eval <= -3.5  and eval > -4.5) else 0
    interv8neg = 1 if (eval <= -4.5) else 0

    output_bin.append(signe)
    output_bin.append(zero)
    output_bin.append(interv1pos)
    output_bin.append(interv2pos)
    output_bin.append(interv3pos)
    output_bin.append(interv4pos)
    output_bin.append(interv5pos)
    output_bin.append(interv6pos)
    output_bin.append(interv7pos)
    output_bin.append(interv8pos)
    output_bin.append(interv1neg)
    output_bin.append(interv2neg)
    output_bin.append(interv3neg)
    output_bin.append(interv4neg)
    output_bin.append(interv5neg)
    output_bin.append(interv6neg)
    output_bin.append(interv7neg)
    output_bin.append(interv8neg)

    arr1 = np.array(output_bin, dtype=float)
    
    return arr1

def eval5(eval):
    output_bin = []

    zero = 1 if eval == 0 else 0
    interv1pos = 1 if eval < 0.25 else 0
    interv2pos = 1 if (eval >= 0.25  and eval < 0.5) else 0
    interv3pos = 1 if (eval >= 0.5  and eval < 1) else 0
    interv4pos = 1 if (eval >= 1  and eval < 1.5) else 0
    interv5pos = 1 if (eval >= 1.5  and eval < 2.5) else 0
    interv6pos = 1 if (eval >= 2.5  and eval < 3.5) else 0
    interv7pos = 1 if (eval >= 3.5  and eval < 4.5) else 0
    interv8pos = 1 if (eval >= 4.5) else 0
    interv1neg = 1 if (eval > -0.25 and eval<0) else 0
    interv2neg = 1 if (eval <= -0.25  and eval > -0.5) else 0
    interv3neg = 1 if (eval <= -0.5  and eval > -1) else 0
    interv4neg = 1 if (eval <= -1  and eval > -1.5) else 0
    interv5neg = 1 if (eval <= -1.5  and eval > -2.5) else 0
    interv6neg = 1 if (eval <= -2.5  and eval > -3.5) else 0
    interv7neg = 1 if (eval <= -3.5  and eval > -4.5) else 0
    interv8neg = 1 if (eval <= -4.5) else 0

    output_bin.append(zero)
    output_bin.append(interv1pos)
    output_bin.append(interv2pos)
    output_bin.append(interv3pos)
    output_bin.append(interv4pos)
    output_bin.append(interv5pos)
    output_bin.append(interv6pos)
    output_bin.append(interv7pos)
    output_bin.append(interv8pos)
    output_bin.append(interv1neg)
    output_bin.append(interv2neg)
    output_bin.append(interv3neg)
    output_bin.append(interv4neg)
    output_bin.append(interv5neg)
    output_bin.append(interv6neg)
    output_bin.append(interv7neg)
    output_bin.append(interv8neg)

    arr1 = np.array(output_bin, dtype=float)
    
    return arr1

def eval6(eval):
    output_bin = []

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

    arr1 = np.array(output_bin, dtype=float)
    
    return arr1

def eval7(eval):
    output_bin = []

    zero = 1 if eval == 0 else 0
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

    output_bin.append(zero)
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

    arr1 = np.array(output_bin, dtype=float)
    
    return arr1

def xarxa1():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(64, ))) # input layer
    #MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    #MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa2():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(64, ))) # input layer
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    #MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa3():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(64, ))) # input layer
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa4():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(64, ))) # input layer
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa5():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(64, ))) # input layer
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa6():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(64, ))) # input layer
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
    MLP.add(InputLayer(input_shape=(64, ))) # input layer
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    #MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def xarxa2Dropout():
    MLP = Sequential()
    MLP.add(InputLayer(input_shape=(64, ))) # input layer
    MLP.add(Dense(128, activation='sigmoid')) # hidden layer 1
    MLP.add(Dropout(0.45)) # hidden layer 1
    MLP.add(Dense(256, activation='sigmoid')) # hidden layer 1
    MLP.add(Dropout(0.45)) # hidden layer 1
    #MLP.add(Dense(64, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(32, activation='sigmoid')) # hidden layer 2
    #MLP.add(Dense(16, activation='sigmoid')) # hidden layer 2
    MLP.add(Dense(func[1], activation='sigmoid')) # output layer
    return MLP

def getTuples(numEvaluacions,numTests):
    # Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
    with open('PosicionsEvaluacions.csv', 'r') as file:
        reader = csv.reader(file)
        
        #Obtener los valores para el entrenamiento
        first=True
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

    # Abrir el archivo csv y leer los primeros 'numEvaluacions' valores
    with open('PosicionsTest.csv', 'r') as file:
        reader = csv.reader(file)

        first=True
        TestTuples = []
        o = 0
        
        for row in reader:    
            if not first and o < numTests:
                TestTuples.append((row[0], row[1]))
                o += 1
            first = False
    
    return TrainingTuples, TestTuples


def convertTuple(Tuples, func):

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

        arr1 = func(eval)
        
        outputs.append(arr1)

    outputs = np.array(outputs)
    inputs = np.array(inputs)
    
    return inputs,outputs

evalutionFunctions = [(eval6,20)]
differentNetworks = [xarxa2] #[xarxa1,xarxa2,xarxa3,xarxa4,xarxa5,xarxa6,xarxa7]
listOptimizers = ['Adam'] #['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam','Ftrl']

TrainingTuples,TestTuples = getTuples(numEvaluacions=2000000,numTests=100000)

inputsTraining,outputsTraining = convertTuple(TrainingTuples, eval6)

inputsTest,outputsTest = convertTuple(TestTuples, eval6)

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
                    epochs=10, batch_size=256) #was 20 epochs and 128 batch_size

            train_accuracy = history.history['accuracy'][-1]
            train_loss = history.history['loss'][-1]
            
            # evaluate performance
            test_loss, test_acc = MLP.evaluate(inputsTest, outputsTest,
                                            batch_size=128,
                                            verbose=0)

            with open('ValorsTestDropout.txt', mode='a') as archivo:
                archivo.write('Xarxa (batchsize diferent): '+str(xarxa2Dropout)+'\n')
                archivo.write('Train acc '+str(train_accuracy)+'\n')
                archivo.write('Train loss '+str(train_loss)+'\n')
                archivo.write('Test acc '+str(test_acc)+'\n')
                archivo.write('Test loss '+str(test_loss)+'\n')


