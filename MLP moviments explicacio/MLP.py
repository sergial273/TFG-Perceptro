import csv
from getFiles import *
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

def definirmoviment(fen1, fen2, moved):
    PiecesTonNum = {
    "p": "1",
    "n": "2",
    "b": "3",
    "r": "4",
    "q": "5",
    "k": "6"
    }

    center = [27, 28, 35, 36]
    outside = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 57, 58, 59, 60, 61, 62, 63]

    board1 = chess.Board(fen1)
    board2 = chess.Board(fen2)

    fen1 = fen1.split()
    fen2 = fen2.split()

    #check if there has been a capture
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
    position1 = 0
    #checking which piece moved
    for elem in codifsencera2:

        if elem.isdigit() and elem != codifsencera1[count] and codifsencera1[count].isascii():
            if moved == PiecesTonNum[codifsencera1[count].lower()]:   
                piece = codifsencera1[count]
                position1 = (7-fil)*8 + col
            
        if elem == "/" and position1 == 0:
            fil += 1
            col = -1
    
        col += 1
        count += 1

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

    centre = 0
    mig = 0
    fora = 0
    if position2 in center:
        centre = 1
    elif position2 in outside:
        fora = 1
    else:
        mig = 1


    control = 0
    if len(board2.attacks(position2)) - len(board1.attacks(position1)) < 0:
        #perd d'algunes caselles
        control = 1

    captura = a-b #És captura
    check = 1 if board1.is_check() else 0 #Està en jaque

    return captura, check, centre, mig, fora, control

def eval6(eval,mate, fen1, fen2, moved):
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

    captura, escac, centre, mig, exterior, control = definirmoviment(fen1, fen2, moved)
    output_bin.append(captura)
    output_bin.append(escac)
    output_bin.append(centre)
    output_bin.append(mig)
    output_bin.append(exterior)
    output_bin.append(control)

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

        arr1 = func(eval,line[4],line[0],line[1],line[3])
        
        outputs.append(arr1)

    outputs = np.array(outputs)
    inputs = np.array(inputs)
    
    return inputs,outputs

evalutionFunctions = [(eval6,27)]
differentNetworks = [xarxa2] #[xarxa1,xarxa2,xarxa3,xarxa4,xarxa5,xarxa6,xarxa7,xarxa2Dropout]
listOptimizers = ['Adam'] #['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam','Ftrl']

TrainingTuples,TestTuples = getTuples(numEvaluacions=2000000,numTests=100000)


inputsTraining,outputsTraining = convertTuple(TrainingTuples, eval6)


inputsTest,outputsTest = convertTuple(TestTuples, eval6)

#normalitzar la info
inputsTraining = inputsTraining.astype('float32') / 127
inputsTest = inputsTest.astype('float32') / 127

"""with open(os.getcwd()+'\MLP moviments explicacio\\inputsTraining.txt', mode='w') as archivo:
    for line in inputsTraining:
        archivo.write(str(line)+'\n')"""


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
                    epochs=10, batch_size=128) #was 20 epochs and 128 batch_size

            train_accuracy = history.history['accuracy'][-1]
            train_loss = history.history['loss'][-1]
            
            # evaluate performance
            test_loss, test_acc = MLP.evaluate(inputsTest, outputsTest,
                                            batch_size=128,
                                            verbose=0)

            """with open(os.getcwd()+'\MLP moviments explicacio\\test.txt', mode='a') as archivo:
                archivo.write('Xarxa, Funcio eval, Optimitzador: '+str(xarxa)+', '+str(func)+', '+str(optimizer)+'\n')
                archivo.write('Train acc '+str(train_accuracy)+'\n')
                archivo.write('Test acc '+str(test_acc)+'\n')
                archivo.write("-" * 50+'\n')"""

            TrainingTuples,TestTuples = getTuples(numEvaluacions=2,numTests=2)

            inputsTraining,outputsTraining = convertTuple(TrainingTuples, eval6)

            pred = MLP.predict(inputsTraining)

            print("AAAAAAA")
            print(pred)
            print("AAAAAAA")
            
            # Guardar los pesos de la red en un archivo HDF5
            MLP.save("model.h5")
            
