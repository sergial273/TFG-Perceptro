import chess

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

moved = "2"

fen1 = "rnbqk1nr/pp1p1pbp/4p1p1/2p5/7P/1P3P2/PBPPPNP1/R2QKBNR w KQkq - 0 1"#"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen2 = "rnbqk1nr/pp1p1pbp/4p1p1/2p5/4N2P/1P3P2/PBPPP1P1/R2QKBNR w KQkq - 0 1"#"rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR w KQkq - 0 1"

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


capture = a-b #---------------------------------------------------------------------------- És captura
check = 1 if board1.is_check() else 0 #---------------------------------------------------- Està en jaque

print (capture)
print (check)
print (codifsencera1)
print (codifsencera2)

print(position1)
print(position2)
print(piece)


print(len(board2.attacks(position2)))

print(len(board1.attacks(position1)))

control = 0
if len(board2.attacks(position2)) - len(board1.attacks(position1)) < 0:
    #perd d'algunes caselles
    control = 1