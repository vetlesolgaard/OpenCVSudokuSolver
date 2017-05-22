from Layout import *
import numpy as np
class Square(object):

    def __init__(self, row, column, box, board, number):
        self.number     = number
        self.box        = box
        self.column     = column
        self.row        = row
        self.nextsquare = None
        self.board      = board

    def addnumber(self, number):
        self.number = number

    def addNext(self, nextsquare):
        self.nextsquare = nextsquare

    def getnumber(self):
        return self.number

    def search(self, number):
        if not self.column.exists(number) and not self.row.exists(number) and not self.box.exists(number):
            return True
        return False
    def FillBoard(self):
        nextSquare = False
        if ifisinstance(PreExistingSquare(), Square):
            if nextSquare is not None:
                print "is not None"
                self.nextsquare.FillBoard
            else:
                print "is None"
                #add to solution
                return True
        else:
            for i in range(self.box.getlength()):
                if self.search(i):
                    self.number = i
                    if nextSquare is None:
                        #add to solution
                        #return True? maybe th
                        return True
                    self.nextsquare = self.nextsquare.FillBoard()
                self.number = 0
        return False





class PreExistingSquare(Square):

    def __init__(self, row, column, box, board, number):
        super(PreExistingSquare, self).__init__(row,column,box,board,number)

class EmptySquare(Square):
    
    def __init__(self, row, column, box, board):
        super(EmptySquare, self).__init__(row,column,box,board, 0)

