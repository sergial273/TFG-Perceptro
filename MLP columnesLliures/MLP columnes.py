import csv
from getFiles import *
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

def aCasellaLliure(fen1, fen2, moved):
    PiecesTonNum = {
    "p": "1",
    "n": "2",
    "b": "3",
    "r": "4",
    "q": "5",
    "k": "6"
    }
    
    board2 = chess.Board(fen2)

    fen1 = fen1.split()
    fen2 = fen2.split()

    a = b = 0
    codifsencera1 = codifsencera2 = ""
    for elem in fen1[0]:
        if not elem.isdigit() and elem != "/":
            a += 1
            codifsencera1 += elem
        elif elem == "/":
            codifsencera1 += elem
        else:
            codifsencera1 += ("1" * int(elem))
        

    for elem in fen2[0]:
        if not elem.isdigit() and elem != "/":
            b += 1
            codifsencera2 += elem
        elif elem == "/":
            codifsencera2 += elem
        else:
            codifsencera2 += ("1" * int(elem))
    col = 0
    fil = 0
    count = 0   
    position2 = 0
    #checking which piece moved
    for elem in codifsencera1:
        if elem.isdigit() and elem != codifsencera2[count] and codifsencera2[count].isascii():
            if moved == PiecesTonNum[codifsencera2[count].lower()]:   
                piece = codifsencera2[count]
                position2 = (7-fil)*8 + col
            
        if elem == "/" and position2 == 0:
            fil += 1
            col = -1

        col += 1
        count += 1
        
    freeCol = []
    for col in range(8):
        for fil in range(8):
            ind = col + 8*fil
            if board2.piece_at(ind) != None:
                if board2.piece_at(ind).piece_type == chess.PAWN:
                    freeCol.append(0)
                    break
            if ind > 55: freeCol.append(1)
    
    if freeCol[position2 % 8] == 1:
        return 1
    
    return 0

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

        arr1 = func(line[0],line[1],line[3])
        
        outputs.append(arr1)

    outputs = np.array(outputs)
    inputs = np.array(inputs)
    
    return inputs,outputs

evalutionFunctions = [(aCasellaLliure,1)]
differentNetworks = [xarxa2] #[xarxa1,xarxa2,xarxa3,xarxa4,xarxa5,xarxa6,xarxa7,xarxa2Dropout]
listOptimizers = ['Adam'] #['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam','Ftrl']

TrainingTuples,TestTuples = getTuples(numEvaluacions=2000000,numTests=100000)


inputsTraining,outputsTraining = convertTuple(TrainingTuples, aCasellaLliure)


inputsTest,outputsTest = convertTuple(TestTuples, aCasellaLliure)

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
            MLP.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])

            # train (fit)
            history = MLP.fit(inputsTraining, outputsTraining, 
                    epochs=10, batch_size=128) #was 20 epochs and 128 batch_size

            train_accuracy = history.history['accuracy'][-1]
            train_loss = history.history['loss'][-1]
            
            # evaluate performance
            test_loss, test_acc = MLP.evaluate(inputsTest, outputsTest,
                                            batch_size=128,
                                            verbose=0)

            """with open(os.getcwd()+'\MLP columnesLliures\\test.txt', mode='a') as archivo:
                archivo.write('Xarxa, Funcio eval, Optimitzador: '+str(xarxa)+', '+str(func)+', '+str(optimizer)+'\n')
                archivo.write('Train acc '+str(train_accuracy)+'\n')
                archivo.write('Test acc '+str(test_acc)+'\n')
                archivo.write("-" * 50+'\n')"""
            

            TrainingTuples,TestTuples = getTuples(numEvaluacions=5,numTests=5)


            inputsTraining,outputsTraining = convertTuple(TrainingTuples, aCasellaLliure)


            inputsTest,outputsTest = convertTuple(TestTuples, aCasellaLliure)

            print("AAAAAAAAA")
            print(MLP.predict(inputsTest))

            

