import os
from stockfish import Stockfish
import chess.pgn
import io
import csv
from concurrent.futures import ProcessPoolExecutor

    
def readNext(self):
    self.directory = os.getcwd()+'\\PGNs'
    self.stockfish = Stockfish(path="./stockfish/stockfish-windows-2022-x86-64-avx2")
    for filename in os.listdir(self.directory):
        totalGames = 0
        totalBinaries = 0
        primer = True
        with open(os.path.join(self.directory, filename), 'r') as f: # open in readonly mode
        
            contenido = f.read()

            bloques = contenido.split("[Event")
            # Open the CSV file in learn mode and create a writer object
            with open(os.getcwd()+'\MLP moviments\PosicionsEvaluacions2.csv', "a", newline="") as archivo:
                # Print each block of text without the separator
                for bloque in bloques:

                    if primer:
                        primer = False
                    else:
                        game = io.StringIO("[Event"+bloque)
                        game = chess.pgn.read_game(game)
                        board = game.board()

                        fen1 = chess.STARTING_FEN
                        self.stockfish.set_fen_position(fen1)
                        evaluation1 = self.stockfish.get_evaluation()
                        evaluation1 = evaluation1['value']
                        mate1 = 0

                        for move in game.mainline_moves():
                            mate2 = 0

                            piece = (board.piece_at(move.from_square)).piece_type

                            totalBinaries += 1

                            # Make the move on the board
                            board.push(move)
                            
                            # generate FEN board in binary
                            fen2 = board.fen()

                            #evaluate FEN with stockfish and save pair
                            self.stockfish.set_fen_position(fen2)
                            evaluation2 = self.stockfish.get_evaluation()

                            if evaluation2['type'] == 'cp':
                                evaluation2 = evaluation2['value']

                            else:
                                mate2 = 1
                                evaluation2 = evaluation2['value']   
                            
                            #If it is mate, the evaluation will be that of the mate
                        
                            if (mate1 == 1 and mate2 == 1):
                                evaluation = int(evaluation2) - int(evaluation1)
                                mate = 1
                                textmate = "Dos son mate"
                            
                            elif (mate1 == 0 and mate2 == 0):
                                evaluation = int(evaluation2) - int(evaluation1)
                                mate = 0
                                textmate = "normal"

                            elif mate1 == 1 and mate2 == 0:
                                evaluation = int(int(evaluation2) - round((1000/int(evaluation1)),0))
                                mate = 0
                                textmate = "Mate pos1"

                            else:
                                evaluation = int(round((1000/int(evaluation2)),0) - int(evaluation1))
                                mate = 1
                                textmate = "Mate pos2"
                            
                            escritor = csv.writer(archivo)                      

                            # We write it back to the file
                            escritor.writerow([fen1,fen2,evaluation,piece,mate,textmate])

                            

                            if fen2 != fen1:
                                fen1 = fen2
                                evaluation1 = evaluation2
                                mate1 = mate2

                        totalGames += 1
            
            print(filename)
            print(totalGames, " Games")
            print(totalBinaries, " Binaries")
            print("-" * 50) 


# Define la función que se ejecutará en cada proceso
def process_file(file_path):
    primer = True
    stockfish = Stockfish(path="./stockfish/stockfish-windows-2022-x86-64-avx2")
    with open(file_path, 'r') as file:
        contenido = file.read()
    bloques = contenido.split("[Event")
    result = []
    # Print each block of text without the separator
    for bloque in bloques:

        if primer:
            primer = False
        else:
            game = io.StringIO("[Event"+bloque)
            game = chess.pgn.read_game(game)
            board = game.board()

            fen1 = chess.STARTING_FEN
            stockfish.set_fen_position(fen1)
            evaluation1 = stockfish.get_evaluation()
            evaluation1 = evaluation1['value']
            mate1 = 0

            for move in game.mainline_moves():
                mate2 = 0

                piece = (board.piece_at(move.from_square)).piece_type

                

                # Make the move on the board
                board.push(move)
                
                # generate FEN board in binary
                fen2 = board.fen()

                #evaluate FEN with stockfish and save pair
                stockfish.set_fen_position(fen2)
                evaluation2 = stockfish.get_evaluation()

                if evaluation2['type'] == 'cp':
                    evaluation2 = evaluation2['value']

                else:
                    mate2 = 1
                    evaluation2 = evaluation2['value']   
                
                #If it is mate, the evaluation will be that of the mate
            
                if (mate1 == 1 and mate2 == 1):
                    evaluation = int(evaluation2) - int(evaluation1)
                    mate = 1
                    textmate = "Dos son mate"
                
                elif (mate1 == 0 and mate2 == 0):
                    evaluation = int(evaluation2) - int(evaluation1)
                    mate = 0
                    textmate = "normal"

                elif mate1 == 1 and mate2 == 0:
                    evaluation = int(int(evaluation2) - round((1000/int(evaluation1)),0))
                    mate = 0
                    textmate = "Mate pos1"

                else:
                    evaluation = int(round((1000/int(evaluation2)),0) - int(evaluation1))
                    mate = 1
                    textmate = "Mate pos2"

                result.append([fen1,fen2,evaluation,piece,mate,textmate])

                if fen2 != fen1:
                            fen1 = fen2
                            evaluation1 = evaluation2
                            mate1 = mate2

    # Devuelve el resultado del procesamiento
    print(len(result))
    return len(result)

# Define la función que ejecuta los procesos
def process_files_concurrently(file_paths):
    # Crea un ProcessPoolExecutor con 4 procesos
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Crea una lista de tareas a ejecutar con executor.submit()
        tasks = [executor.submit(process_file, file_path) for file_path in file_paths]
        # Espera a que todas las tareas se completen y devuelve los resultados
        return [task.result() for task in tasks]

# Ejemplo de uso
if __name__ == '__main__':
    # Lista los archivos a procesar
    
    file_paths = [os.path.join(os.getcwd()+'\\PGNsTEST', file_name) for file_name in os.listdir('./PGNsTEST') if file_name.endswith('.pgn')]
    
    # Procesa los archivos concurrentemente
    results = process_files_concurrently(file_paths)
    # Escribe los resultados en un archivo CSV
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Archivo', 'Resultado'])
        for file_path, result in zip(file_paths, results):
            writer.writerow([file_path, result])

