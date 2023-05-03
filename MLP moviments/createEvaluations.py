import os
from stockfish import Stockfish
import chess.pgn
import io
import csv

class getFiles:
    
    def __init__(self):
        self.directory = os.getcwd()+'\\PGNs'
        self.stockfish = Stockfish(path="./stockfish/stockfish-windows-2022-x86-64-avx2")
    
    def readNext(self):
        
        for filename in os.listdir(self.directory):
            totalGames = 0
            totalBinaries = 0
            primer = True
            with open(os.path.join(self.directory, filename), 'r') as f: # open in readonly mode
            # do your stuff 
                contenido = f.read()

                bloques = contenido.split("[Event")
                # Abre el archivo CSV en modo de apendizaje y crea un objeto escritor
                with open(os.getcwd()+'\MLP moviments\PosicionsEvaluacions2.csv', "a", newline="") as archivo:
                    # Imprimir cada bloque de texto sin el separador
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

                                # Realizar el movimiento en el tablero
                                board.push(move)
                                
                                # generar el tablero FEN en binario
                                fen2 = board.fen()

                                #evaluar FEN amb stockfish i guardar parell  
                                self.stockfish.set_fen_position(fen2)
                                evaluation2 = self.stockfish.get_evaluation()

                                if evaluation2['type'] == 'cp':
                                    evaluation2 = evaluation2['value']

                                else:
                                    mate2 = 1
                                    evaluation2 = evaluation2['value']   
                                
                                #si es mate la evaluació serà la del mate
                            
                                if (mate1 == 1 and mate2 == 1) or (mate1 == 0 and mate2 == 0):
                                    evaluation = int(evaluation2) - int(evaluation1)
                                    mate = mate1
                                elif mate1 == 1 and mate2 == 0:
                                    evaluation = int(int(evaluation2) - int(evaluation1)*(100/int(evaluation1)))
                                    mate = 0
                                else:
                                    evaluation = int(int(evaluation2)*(100/int(evaluation2))) - int(evaluation1)
                                    mate = 1
                                
                                escritor = csv.writer(archivo)                      

                                # Agrega algunas filas de datos al final del archivo
                                escritor.writerow([fen1,fen2,evaluation,piece,mate])

                                if fen2 != fen1:
                                    fen1 = fen2
                                    evaluation1 = evaluation2
                                    mate1 = mate2

                            totalGames += 1
                
                print(filename)
                print(totalGames, " Games")
                print(totalBinaries, " Binaries")
                print("-" * 50)  # Imprimir línea separador



print("starting")
g = getFiles()
g.readNext()
print("end")