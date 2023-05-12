import chess

fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


"""board = chess.Board(fen1)

print (board)"""


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
    
capture = a-b #----------------------------------------------------------------------------

print (capture)
print (codifsencera1)
print (codifsencera2)

