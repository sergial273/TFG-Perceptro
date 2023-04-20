import os
from stockfish import Stockfish
import chess.pgn
import io

class getFiles:
    
    def __init__(self):
        self.directory = os.getcwd()+'\\PGNs'
        self.stockfish = Stockfish(path="./stockfish/stockfish-windows-2022-x86-64-avx2")
    
    def readNext(self):
        #filename = "test.pgn" and put if instead of for below
        for filename in os.listdir(self.directory):
            total = 0
            totalBinaries = 0
            primer = True
            with open(os.path.join(self.directory, filename), 'r') as f: # open in readonly mode
            # do your stuff 
                contenido = f.read()

                bloques = contenido.split("[Event")

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
                            
                            # generar el tablero FEN en binari
                            binary = self.fenToBinaryAllInSquares(board.fen())  

                            #evaluar FEN amb stockfish i guardar parell  
                            self.stockfish.set_fen_position(board.fen())
                            evaluation = self.stockfish.get_evaluation()

                            #del binary tornar al fen per sabre si funciona correctament
                            fen = self.binaryToFenAllInSquares(binary)

                            fn =""
                            blanks = 0
                            for ele in board.fen():
                                if ele == " ":
                                    blanks += 1
                                    if blanks == 4:
                                        break
                                fn +=ele

                            if fen != fn:
                                print(board.fen())
                                print(fen)
                                break

                        total += 1


                print(filename)
                print(total, " Games")
                print(totalBinaries, " tests")
                print("-" * 50)  # Imprimir línea separador

        
    def fenToBinarySeparatedCamps(self, fen):
        sections = fen.split(' ')

        binary = ""

        for symbol in sections[0]:
            if symbol == '/':
                pass
            else:
                if symbol.isdigit():

                    #si es un espai en blanc
                    for val in range(int(symbol)):

                        binary += "0000"
                else:
                    pieceColour = "0" if symbol.isupper() else "1" #0 si es majuscula = blanc o 1 si es minuscula = negre
                    pieceTypeFromSymbol = {
                        'k': "110",
                        'p': "001",
                        'n': "010",
                        'b': "011",
                        'r': "100",
                        'q': "101"
                    }
                    pieceType = pieceTypeFromSymbol[symbol.lower()]
                    
                    binary += pieceColour+pieceType
                    
        #seguent jugador
        next_player = "0" if (sections[1] == "w") else "1"

        binary += next_player

        #drets d'enrocar
        castlingRights = sections[2] if len(sections) > 2 else "KQkq"

        castlingRightsBinary = int('0000',2)

        if "K" in castlingRights:
            castlingRightsBinary = castlingRightsBinary | int('1000',2)
        if "Q" in castlingRights:
            castlingRightsBinary = castlingRightsBinary | int('100',2)
        if "k" in castlingRights:
            castlingRightsBinary = castlingRightsBinary | int('10',2)
        if "q" in castlingRights:
            castlingRightsBinary = castlingRightsBinary | int('1',2)              
                
        binary += str(format(castlingRightsBinary,'04b'))

        #peo de capturar en passada
        enPassant = sections[3] if len(sections) > 3 else "-"
        enPassantBinary = ""

        if enPassant == "-":
            enPassantBinary = "0000001"
        else:
            #convertir la lletra
            col = bin(ord(enPassant[0])-97) #retorna el valor en ASCII de 97 si es a, restem per tenir rang de 0-7
            enPassantBinary += str(col[2:])
            
            #convertir el número
            fil = bin(int(enPassant[1]))
            enPassantBinary += str(fil[2:])

            enPassantBinary += "0"

        binary += enPassantBinary

        return binary

    def binaryToFenSeparatedCamps(self, binary):
        binary = str(binary)
        total = 1 #numero de caracters llegits

        bits = ""
        emptySpace = 0
        result = ""

        pieceTypeFromSymbol = {
                        '110': "k",
                        '001': "p",
                        '010': "n",
                        '011': "b",
                        '100': "r",
                        '101': "q"
                    }
        
        for num in binary[0:256]:

            bits += num

            if total%4 == 0: #agafem el grup de 4
                if bits == "0000":
                    emptySpace += 1

                else:
                    if emptySpace != 0:
                        result += str(emptySpace)
                        emptySpace = 0

                    letter = bits[1:4]

                    letter = pieceTypeFromSymbol[letter]
                    
                    if bits[0] == "0": #blanques
                        letter = letter.upper()

                    result += letter

                bits = ""

            if total%32 == 0:#es una linia en fen #
                if emptySpace != 0:
                    result += str(emptySpace)
                    emptySpace = 0
                
                if total!=256:
                    result += "/"
                

            total += 1       
           
        
        result += " "
        #primer cop que entra agafa el color de torn
        color = "w " if binary[256] == '0' else "b "
        
        result += color

        #ara els drets d'enroc
        totes = 4

        if binary[257] == '1': #valor de K
            result += "K"
        else:
            totes -= 1
            
        if binary[258] == '1': #valor de Q
            result += "Q"
        else:
            totes -= 1

        if binary[259] == '1': #valor de k
            result += "k"
        else:
            totes -= 1

        if binary[260] == '1': #valor de q
            result += "q"
        else:
            totes -= 1

        if totes == 0:
            result += "-"

        result += " "

        #ara la captura al pas
        if binary[267] == "0":
            pieceColumn = {
                            '000': "a",
                            '001': "b",
                            '010': "c",
                            '011': "d",
                            '100': "e",
                            '101': "f",
                            '110': "g",
                            '111': "h",
                        }
            letter = pieceColumn[binary[261:264]]

            result += letter

            row = str(int(binary[264:267],2) + 1)

            result += row
        else:
            result += "-"

        return result


    def fenToBinaryAllInSquares(self,fen):
        sections = fen.split(' ')

        binary = ""
        allbinary = ""
        fil, col = 1, 1

        #seguent jugador
        turn = "0" if (sections[1] == "w") else "1"

        for symbol in sections[0]:
            binary=""
            if symbol == '/':
                fil += 1
                col = 1
                pass
            else:
                if symbol.isdigit():
                    #si es un espai en blanc
                    for val in range(int(symbol)):

                        binary += turn+"000000"
                        allbinary +=binary
                        binary=""
                        col += 1
                        
                else:
                    pieceColour = "0" if symbol.isupper() else "1" #0 si es majuscula = blanc o 1 si es minuscula = negre
                    pieceTypeFromSymbol = {
                        'k': "110",
                        'p': "001",
                        'n': "010",
                        'b': "011",
                        'r': "100",
                        'q': "101"
                    }
                    pieceType = pieceTypeFromSymbol[symbol.lower()]
                    if symbol.lower() == "p": #e peó
                        #mirar si es al pas
                        
                        enPassant = sections[3] if len(sections) > 3 else "-"
                        enPassantBinary = ""

                        if enPassant == "-":
                            enPassantBinary = "0"
                            
                        else:
                            #convertir la lletra
                            column = int(ord(enPassant[0])-96) #retorna el valor en ASCII de 97 si es a, restem per tenir rang de 0-7
                            
                            #convertir el número
                            row = int(enPassant[1])
                            val = -2 if turn == "0" else +2
                            row += val

                            if column==col and row == fil:
                                enPassantBinary = "1"
                            else:
                                enPassantBinary = "0"

                        allbinary += turn+pieceColour+"0"+enPassantBinary+pieceType

                    elif symbol.lower() == "r": #es torre
                        #mirar drets enroc
                        
                        #drets d'enrocar
                        castlingRights = sections[2] if len(sections) > 2 else "KQkq"
                        
                        castling="0"
                        
                        if "K" in castlingRights and col==8 and fil == 8:
                            castling = "1"

                        if "Q" in castlingRights and col==1 and fil == 8:
                            castling = "1"

                        if "k" in castlingRights and col==8 and fil == 1:
                            castling = "1"

                        if "q" in castlingRights and col==1 and fil == 1:
                            castling = "1"
                        
                        allbinary += turn+pieceColour+castling+"0"+pieceType

                    elif symbol.lower() == "k": #es rei
                        castlingRights = sections[2] if len(sections) > 2 else "KQkq"
                        castling = "0"
                        if symbol.isupper() and ("K" in castlingRights or "Q" in castlingRights):
                            castling = "1"

                        if symbol.islower() and ("k" in castlingRights or "q" in castlingRights):
                            castling = "1"

                        allbinary += turn+pieceColour+castling+"0"+pieceType

                    
                    else: #es qualsevol altra peça

                        allbinary += turn+pieceColour+"0"+"0"+pieceType
                    
                    col += 1
               

        return allbinary

    def binaryToFenAllInSquares(self, binary):
        
        pieceTypeFromSymbol = {
                        '110': "k",
                        '001': "p",
                        '010': "n",
                        '011': "b",
                        '100': "r",
                        '101': "q"
                    }
        
        empty = 0
        fen = ""
        jump = 1
        fil, col = 1,1
        finalEnPassant="-"
        finalCastle = ""

        for i in range(0, len(binary), 7):

             

            square = binary[i:i+7]
            turn = square[0]
            color = square[1]
            castle = square[2]
            enpassant = square[3]
            piece = square[-3:]
            
            if piece in pieceTypeFromSymbol: #es peça
                
                if empty != 0:
                    fen += str(empty)

                empty = 0

                letter = pieceTypeFromSymbol[piece]

                letter = letter.upper() if color=="0" else letter.lower()

                if letter.lower() == "r":

                    if col==8 and fil==8 and letter.isupper() and int(castle):
                        finalCastle += "K"
                    if col==1 and fil==8 and letter.isupper() and int(castle):
                        finalCastle += "Q"
                    if col==8 and fil==1 and letter.islower() and int(castle):
                        finalCastle += "k"
                    if col==1 and fil==1 and letter.islower() and int(castle):
                        finalCastle += "q"
                    
                
                elif letter.lower() == "p":
                    if enpassant == "1":
                        val = +2 if turn == "0" else -2
                        finalEnPassant = str(chr(col+96))+str(fil+val)

                fen += letter

                
            else: #espai en blanc
                empty += 1

            if jump == 8:

                if empty != 0:
                    fen += str(empty)
                empty = 0
                if i < 441:
                    fen+="/"
                jump = 0

                fil += 1
                col = 0
            
            col += 1
            jump += 1


        if finalCastle == "":
                    finalCastle = "-"
        else:
            order = []
            for letter in finalCastle:
                order.append(letter)
            order.sort()
            finalCastle = ''.join(order)

        turn = "w" if turn == "0" else "b"
        fen += " " + turn + " " + finalCastle + " " + finalEnPassant


                

        return fen





        

print("starting")
g = getFiles()
g.readNext()
#fen = g.fenToBinaryAllInSquares("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
#print(g.binaryToFenAllInSquares(fen))
print("end")