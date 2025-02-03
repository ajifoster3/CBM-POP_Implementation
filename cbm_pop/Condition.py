from collections import Counter
from cbm_pop.Operator_LLM import Operator
from enum import Enum

class Condition(Enum):
    C_0 = 0 # Starting a DI-cycle
    C_1 = 1 # A diversification operator was previous applied
    C_2 = 2 # The first intensification operator was applied
    C_3 = 3 # The second intensification operator was applied
    C_4 = 4 # All intensification operators have been applied this DI-cycle and fitness hasn't improved


class ConditionFunctions:
    @staticmethod
    def perceive_condition(previous_experience):
        """
        Calculates the condition based on the experience memory of which operators where used previously.
        :param previous_experience: Experience memory
        :return: The current condition
        """
        if not previous_experience:
            return Condition.C_0
        if previous_experience[-1][1] in {Operator.BEST_COST_ROUTE_CROSSOVER,
                                          Operator.INTRA_DEPOT_REMOVAL,
                                          Operator.INTRA_DEPOT_SWAPPING,
                                          #'Operator.INTER_DEPOT_SWAPPING,
                                          Operator.SINGLE_ACTION_REROUTING}:
            return Condition.C_1
            # New condition: check if both TWO_SWAP and ONE_MOVE have been used once without improvement
        last_two_operators = [entry[1] for entry in previous_experience[-2:]]
        if Counter(last_two_operators) == Counter([Operator.TWO_SWAP, Operator.ONE_MOVE]) and \
                all(entry[2] == 0 for entry in previous_experience[-2:]):  # Check if gain is zero for both entries
            return Condition.C_4
        if previous_experience[-1][1] == Operator.TWO_SWAP:
            return Condition.C_2
        if previous_experience[-1][1] == Operator.ONE_MOVE:
            return Condition.C_3
