class Actions:
    UNKNOWN = -2
    BLACK = -1
    WHITE = 0

    @classmethod
    def size(cls):
        return 3
    
    @classmethod
    def get_actions(cls):
        return [cls.UNKNOWN, cls.BLACK, cls.WHITE]
    
    @classmethod
    def flip(cls, action): 
        return -((abs(action) + 1) % cls.size())

NUM_CELL_RWD = -10000