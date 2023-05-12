import chess

fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


"""board = chess.Board(fen1)

print (board)"""


fen1 = fen1.split()
fen2 = fen2.split()

#check if there has been a capture
a = 0
b = 0
for elem in fen1[0]:
    if not elem.isdigit() and elem != "/":
        a += 1

for elem in fen2[0]:
    if not elem.isdigit() and elem != "/":
        b += 1
    
capture = a-b #----------------------------------------------------------------------------

print (capture)

