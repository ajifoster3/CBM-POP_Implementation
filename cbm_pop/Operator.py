from enum import Enum

class Operator(Enum):
    TWO_SWAP = 1
    ONE_MOVE = 2
    BEST_COST_ROUTE_CROSSOVER = 3
    INTRA_DEPOT_REMOVAL = 4
    INTRA_DEPOT_SWAPPING = 5
    #INTER_DEPOT_SWAPPING = 6 Not applicable due to lack of depots
    SINGLE_ACTION_REROUTING = 6