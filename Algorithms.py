import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict



class BFSGAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError
        


class GreedyAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self):
        raise NotImplementedError
        
    def h_focal(self, state: int) -> float: # heuristic for focal list (you don't have to use it)
        pass

    def search(self, env: HaifaEnv, epsilon: float = None): 
        raise NotImplementedError  



class AStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

