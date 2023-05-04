import csv
import os

# Open the CSV file in write mode and create a writting object
with open(os.getcwd()+'\MLP moviments\PosicionsEvaluacions2.csv', "w", newline="") as archivo:
    escritor = csv.writer(archivo)

    # Writes the header row
    escritor.writerow(["FEN1","FEN","eval","Piece","Mate"])