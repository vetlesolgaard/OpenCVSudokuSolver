from square import *

class Layout(object):
    def __init__(self, length):
        self.squares = [0 for x in range(0,length)]
    def addsquare(self, square):
        for x in range(0, len(self.squares)):
            if self.squares[x] is 0:
                self.squares[x] = square
                #print square.number
                return True

        return False

    def exists(self, number):
        for x in range(0, len(self.squares)):

            '''if isinstance(self, Column):
                print "kolonne"
            elif isinstance(self, Row):
                print "row"
            elif isinstance(self, Box):
                print "box"
            print "exists number " , number
            print "self squares ",self.squares[x].number
            print " x ", x'''
            if self.squares[x].number == number:
                #print "found"
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