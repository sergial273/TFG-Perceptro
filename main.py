from explanations import Explanations

g = Explanations("./stockfish/stockfish-windows-2022-x86-64-avx2")
exp = g.explanations("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
print(exp[2])
print(exp[0])
print(exp[1])