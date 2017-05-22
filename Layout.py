from square import *
import numpy as np

class Layout(object):
    def __init__(self, length):
        self.squares = np.full((length,1), None)
    def addsquare(self, square):
        for x in range(0, len(self.squares)):
            if self.squares[x] is None:
                self.squares[x] = square
                return True

        return False

    def exists(self, number):
        for x in range(0, len(self.squares)):
            if self.squares[x].number == number:
                return True
        return False
    def getlength(self):
        return len(self.squares)

class Box(Layout):

    def __init__(self, length):
        super(Box,self).__init__(length)

class Column(Layout):

    def __init__(self,length):
        super(Column,self).__init__(length)

class Row(Layout):

    def __init__(self,length):
        super(Row,self).__init__(length)