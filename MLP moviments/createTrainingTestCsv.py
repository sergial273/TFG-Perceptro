import ast
import csv
import sys
from getFiles import *

numEvaluacions = 2000000
numTests = 100000


# Open the csv file and read the first 'numEvaluacions' values
with open(os.getcwd()+'\MLP moviments\PosicionsEvaluacions2.csv', 'r') as file:
    reader = csv.reader(file)
    
    #Get the values ​​for training
    first=True
    i = 0  
    o = 0

    trainingfen1 = []
    trainingfen2 = []

    # Iterate over each row in the file
    for row in reader:
        
        if not first and i < numEvaluacions:

            trainingfen1.append(row[0])
            trainingfen2.append(row[1])
            i += 1

        elif not first and o < numTests:
            if (row[0] not in trainingfen1) and (row[1] not in trainingfen2):
                
                
                with open(os.getcwd()+'\MLP moviments\PosicionsTest2.csv', "a", newline="") as archivo:
                    escritor = csv.writer(archivo)

                    
                    escritor.writerow([row[0], row[1], row[2], row[3], row[4]])
                o += 1

        first = False