from collections import Counter
from cbm_pop.Operator import Operator

class ConditionFunctions:
    @staticmethod
    def perceive_condition_row(previous_experience, intensifiers, diversifiers):
        """
        Returns the row index:
          0 = DI start (no previous experience)
          1 = last operator was a diversifier
          2..(2+n_int-1) = last operator was intensifier j -> 2 + j
        """
        if not previous_experience:
            return 0

        op_order = list(intensifiers) + list(diversifiers)
        n_int = len(intensifiers)

        # previous_experience items are [condition_unused, op_col_idx, gain]
        last_op = previous_experience[-1][1]

        # If last_op is already an int, use it; if it's an Operator enum, map to column index.
        if isinstance(last_op, int):
            op_col = last_op
        else:
            op_col = op_order.index(last_op)  # raises if unknown, which is fine to surface

        # Intensifier columns are 0..n_int-1 since intensifiers come first in op_order
        if 0 <= op_col < n_int:
            return 2 + op_col  # intensifier rows start at 2
        else:
            return 1  # diversifier