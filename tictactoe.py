import mcts
import random
from typing import Iterable, Self, Optional

class TicTacToeState(mcts.MCTSState):
    def __init__(self):
        self.board = [0 for _ in range(9)]
        self.turn = 1
        self.turn_count = 0

    def pl(self, player):
        if player == 1:
            return "O"
        elif player == -1:
            return "X"
        elif player == 0:
            return "*"

    def __str__(self):
        b = self.board
        result = f"PLAYER: {self.pl(self.turn)} \n"

        for i in range(3):
            result += " ".join([self.pl(p) for p in b[i*3 : i*3 + 3]]) + "\n"
                
        terminal = self.terminal_value()
        if terminal == 0.5:
            result += "DRAW \n"
        elif terminal != None:
            result += f"WINNER: {"O" if terminal == 1 else "X"} \n"

        return result

    def succesor(self, action: int):
        next_state = TicTacToeState()
        next_state.board = self.board.copy()
        next_state.turn = -1 if self.turn == 1 else 1
        next_state.board[action] = self.turn
        next_state.turn_count = self.turn_count + 1
        return next_state

    def all_next_states(self) -> Iterable[Self]:
        for i in range(9):
            if self.board[i]:
                continue
            yield self.succesor(i)
    
    def terminal_value(self) -> Optional[float]:
        if self.turn_count < 5:
            return None
        b = self.board

        for i in range(3): #check rows
            sum = 0
            for j in range(3):
                sum += b[i*3 + j]
            if sum == 3:
                return 1
            if sum == -3:
                return 0
        
        for i in range(3): #check cols
            sum = 0
            for j in range(3):
                sum += b[i + j*3]
            if sum == 3:
                return 1
            if sum == -3:
                return 0
        
        sum = 0
        for i in range(3): #check left-to-right diagonal
            sum += b[i*3 + i]
        if sum == 3:
            return 1
        if sum == -3:
            return 0
        
        sum = 0
        for i in range(3): #check right-to-left diagonal
            sum += b[i*3 + 2 - i]
        if sum == 3:
            return 1
        if sum == -3:
            return 0
        
        if self.turn_count == 9:
            return 0.5
        
        return None