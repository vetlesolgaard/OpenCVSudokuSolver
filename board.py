import numpy as np
from Layout import *
from square import *

class Board:

    def __init__(self, sudokuarray, length, w, h):
        self.sudokuarray    = sudokuarray
        self.board          = np.full((length, length), None)
        self.box            = np.full((length,1), None)
        self.row            = np.full((length,1), None)
        self.column         = np.full((length,1), None)
        self.length         = length
        self.countHor       = 1
        self.countVert      = 1
        self.countBox       = 1
        self.boxW           = w
        self.boxH           = h


    def solveBoard(self):
        #boolean
        return self.sudokuarray.FillBoard()
    def createBoard(self):
        for x in range(0, self.length):

            newBox          = Box(int(self.length))
            newRow          = Row(int(self.length))
            newColumn       = Column(int(self.length))
            self.box[x]     = newBox
            self.row[x]     = newRow
            self.column     = newColumn

        self.squareNumber   = 0
        self.newSquare      = None
        self.lastSquare     = None

        for x in range(0,self.length):
            for y in range(0,self.length):
                if self.sudokuarray[x][y] == 0:
                    self.squareNumber += 1
                    print self.squareNumber
                    self.newSquare = EmptySquare(self.row[x], self.column[y], self.box[self.countBox-1], self.length, self.squareNumber)
                else:
                    self.newSquare = PreExistingSquare(self.row[x], self.column[y], self.box[self.countBox-1], self.sudokuarray[x][y])
                if x == 0 and y == 0:
                    self.lastSquare = self.newSquare
                self.box[self.countBox-1].addsquare(self.newSquare)
                self.row[x].addsquare(self.newSquare)
                self.column[y].addsquare(self.newSquare)
                self.board[x][y]    = self.newSquare
                self.countHor       += 1

                if self.countHor == self.boxW+1:
                    self.countHor = 1
                    self.countBox += 1

                if self.countVert == self.boxH and y == (self.length -1):
                    self.countHor   = 1
                    self.countVert  = 0
                elif y == (length -1) and self.countVert != self.boxH:
                    self.countBox -= self.boxH

                self.lastSquare.addNext(self.newSquare)
                self.lastSquare = self.newSquare

            self.countVert += 1

array = np.array([[7,0,0,8,0,0,0,0,0], [9,0,0,0,0,0,0,6,0], [0,6,0,2,0,1,5,0,0],
                  [0,0,0,0,4,0,8,0,0], [0,0,7,0,0,0,3,0,0], [4,0,0,0,2,5,0,0,7],
                  [0,4,6,0,0,0,0,8,3], [0,0,3,5,0,0,0,0,0], [0,0,0,0,9,0,0,2,0]])
x = Board(array, 9,3,3)
x.createBoard()