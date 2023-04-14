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
                with open("PosicionsEvaluacions.csv", "a", newline="") as archivo:
                    # Imprimir cada bloque de texto sin el separador
                    for bloque in bloques:

                        if primer:
                            primer = False
                        else:
                            game = io.StringIO("[Event"+bloque)
                            game = chess.pgn.read_game(game)
                            board = game.board()

                            for move in game.mainline_moves():
                                totalBinaries += 1
                                # Realizar el movimiento en el tablero
                                board.push(move)
                                
                                # generar el tablero FEN en binario
                                fen = board.fen()

                                #evaluar FEN amb stockfish i guardar parell  
                                self.stockfish.set_fen_position(fen)
                                evaluation = self.stockfish.get_evaluation()

                                
                                escritor = csv.writer(archivo)

                                # Agrega algunas filas de datos al final del archivo
                                escritor.writerow([fen, evaluation])

                            totalGames += 1
                
                # Abrir el archivo en modo lectura
                with open('totalgames.txt', 'r') as archivo:
                    # Leer todas las líneas del archivo
                    lineas = archivo.readlines()
                    
                    # Analizar cada línea en busca de los números
                    for linea in lineas:
                        if 'Partides:' in linea:
                            partides = int(linea.split(':')[1])
                        elif 'PosicionsEvaluades:' in linea:
                            posicions_evaluades = int(linea.split(':')[1])

                # Sumar otro número a los valores extraídos
                partides += totalGames
                posicions_evaluades += totalBinaries

                # Abrir el archivo en modo escritura
                with open('totalgames.txt', 'w') as archivo:
                    # Escribir los valores actualizados en el archivo
                    archivo.write('Partides: {}\n'.format(partides))
                    archivo.write('PosicionsEvaluades: {}'.format(posicions_evaluades))
                                
                
                print(filename)
                print(totalGames, " Games")
                print(totalBinaries, " Binaries")
                print("-" * 50)  # Imprimir línea separador

                os.remove(os.path.join(self.directory, filename))



print("starting")
g = getFiles()
g.readNext()
print("end")