from Layout import *
from square import *

class Board(object):


    def __init__(self, sudokuarray, length, w, h):
        self.sudokuarray    = sudokuarray
        self.board          = [[0for y in range(0,length)] for x in range(0,length)]
        self.box            = [0 for x in range(0,length)]
        self.row            = [0 for x in range(0,length)]
        self.column         = [0 for x in range(0,length)]
        self.length         = length
        self.countHor       = 1
        self.countVert      = 1
        self.countBox       = 1
        self.boxW           = w
        self.boxH           = h
        self.solution       = [[0for y in range(0,length)] for x in range(0,length)]


    def addtosolution(self,board):
        for x in range(0,self.length):
            for y in range(0,self.length):
                self.solution[x][y] = board[x][y].number
        #print "TEEEST", board[0][0].number

    def solveBoard(self):
        #boolean
        return self.board[0][0].FillBoard()
    def createBoard(self):
        for x in range(0, self.length):

            newBox          = Box(int(self.length))
            newRow          = Row(int(self.length))
            newColumn       = Column(int(self.length))
            self.box[x]     = newBox
            self.row[x]     = newRow
            self.column[x]  = newColumn

        self.squareNumber   = 0
        self.newSquare      = None
        self.lastSquare     = None
        for x in range(0,self.length):
            for y in range(0,self.length):

                if self.sudokuarray[x][y] == 0:
                    self.squareNumber += 1
                    self.newSquare = EmptySquare(self.row[x], self.column[y], self.box[self.countBox-1], self.length, 0,self)
                else:
                    self.newSquare = PreExistingSquare(self.row[x], self.column[y], self.box[self.countBox-1], self.sudokuarray[x][y], self.length,self)
                if x == 0 and y == 0:
                    self.lastSquare = self.newSquare

                self.box[self.countBox-1].addsquare(self.newSquare)
                self.row[x].addsquare(self.newSquare)
                self.column[y].addsquare(self.newSquare)
                #print self.column[y].squares[x].number
                self.board[x][y]    = self.newSquare
                self.countHor       += 1

                if self.countHor == self.boxW+1:

                    self.countHor = 1
                    self.countBox += 1

                if self.countVert == self.boxH and y == (self.length -1):
                    self.countHor   = 1
                    self.countVert  = 0
                elif y == (self.length -1) and self.countVert != self.boxH:
                    self.countBox -= self.boxH

                self.lastSquare.addNext(self.newSquare)
                self.lastSquare = self.newSquare

            self.countVert += 1



array = np.array([[7,0,0,8,0,0,0,0,0], [9,0,0,0,0,0,0,6,0], [0,6,0,2,0,1,5,0,0],
                  [0,0,0,0,4,0,8,0,0], [0,0,7,0,0,0,3,0,0], [4,0,0,0,2,5,0,0,7],
                  [0,4,6,0,0,0,0,8,3], [0,0,3,5,0,0,0,0,0], [0,0,0,0,9,0,0,2,0]])
x = Board(array, 9,3,3)
x.createBoard()


x.solveBoard()


