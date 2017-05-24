from Layout import *
class Square(object):

    def __init__(self, row, column, box, length, board):
        self.number     = 0
        self.box        = box
        self.column     = column
        self.row        = row
        self.nextsquare = 0
        self.length     = length
        self.board      = board
        self.test       = 0



    def printboard(self):
        for x in range(0,self.length):
            for y in range(0,self.length):
                #print self.board[x][y].number
                test = 0
        #person = input('stoop')

    def addnumber(self, number):
        self.number = number

    def addNext(self, nextsquare):
        self.nextsquare = nextsquare

    def getnumber(self):
        return self.number

    def search(self, number):
        #print "search number: ", number
        #print "kolonne 0", self.column.squares[0].number
        #print " rad 0", self.row.squares[0].number
        #print " box 0 ", self.box.squares[0].number
        if not self.column.exists(number) and not self.row.exists(number) and not self.box.exists(number):
            #print "True"
            return True
        #print "False"
        return False
    def FillBoard(self):
        next = False

        if isinstance(self,PreExistingSquare):
            #print "pree"
            if self.nextsquare != 0:
                #print "number : ", self.number
                #person = input('preexisting')
                self.nextsquare.FillBoard()
            else:
                #print "else som den ikke skal inn i"
                self.nextsquare = self.nextsquare
                return True
        else:
            for i in range(1,self.box.getlength()+1):
               #print self.test, "testing"
                #print "number",self.row.squares[0].number
                #print "i", i
                #person = input('empty')
               #if this test is true, it did not find anything.
                #print "nextsquare", self.nextsquare
                if self.search(i):
                    #print "not found"
                    self.number = i
                    #print self.number

                    if self.nextsquare == 0:
                        #print "nextsquare =0"
                        self.board.addtosolution(self.board.board)
                        return True
                    else:
                        next = self.nextsquare.FillBoard()
                        #print "next" , next
            self.number = 0
        #print "return false"
        return False





class PreExistingSquare(Square):

    def __init__(self, row, column, box, number,length, board):
        super(PreExistingSquare, self).__init__(row,column,box,length,board)
        self.addnumber(number)
class EmptySquare(Square):
    
    def __init__(self, row, column, box,  length, number, board):
        super(EmptySquare, self).__init__(row,column,box,length, board)
        self.number = number


