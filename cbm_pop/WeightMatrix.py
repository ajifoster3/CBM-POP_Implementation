
class WeightMatrix:
    def __init__(self, num_intensifiers, num_diversifiers, ours=False):
        self.num_diversifiers = num_diversifiers
        self.num_intensifiers = num_intensifiers
        if ours:
            self.weights = self.init_our_weight_matrix()
        else:
            self.weights = self.init_weight_matrix()


    def init_weight_matrix(self):
        """
        Generates a weight matrix mapping conditions onto operations.
            o1o2o3o4o5o6o7
        c0: 0 0 1 1 1 1 1 <- Initial diversification operator
        c1: 1 1 0 0 0 0 0 <- Subsequent intensification operator
        c2: 0 1 0 0 0 0 0 <- Which operator after intensification operator 1
        c3: 1 0 0 0 0 0 0 <- Which operator after intensification operator 2 (Move rows if there's more intensifiers)
        c4: 0 0 1 1 1 1 1 <- If all intensification operators haven't improved the solution, which diversificator to use
        :return: A weight matrix mapping conditions to operations
        """
        # Initialize and return a weight matrix (for operator selection, if needed)
        weight_matrix = []
        initial_diversifier_condition_row = [0] * self.num_intensifiers + [1] * self.num_diversifiers
        weight_matrix.append(initial_diversifier_condition_row)
        initial_intensifier_condition_row = [1] * self.num_intensifiers + [0] * self.num_diversifiers
        weight_matrix.append(initial_intensifier_condition_row)
        for i in range(self.num_intensifiers):
            intensifier_condition_row = [1] * self.num_intensifiers + [0] * self.num_diversifiers
            intensifier_condition_row[i] = 0
            weight_matrix.append(intensifier_condition_row)
        final_diversifier_condition_row = [0] * self.num_intensifiers + [1] * self.num_diversifiers
        weight_matrix.append(final_diversifier_condition_row)
        return weight_matrix

    def init_our_weight_matrix(self):
        """
        Generates a weight matrix mapping conditions onto operations.
            o1o2o3o4o5o6o7
        c0: 0 0 1 1 1 1 1 <- Initial diversification operator
        c1: 1 1 0 0 0 0 0 <- Subsequent intensification operator
        c2: 0 1 0 0 0 0 0 <- Which operator after intensification operator 1
        c3: 1 0 0 0 0 0 0 <- Which operator after intensification operator 2 (Move rows if there's more intensifiers)
        c4: 0 0 1 1 1 1 1 <- If all intensification operators haven't improved the solution, which diversificator to use
        :return: A weight matrix mapping conditions to operations
        """
        # Initialize and return a weight matrix (for operator selection, if needed)
        weight_matrix = []
        initial_diversifier_condition_row = [1] * (self.num_diversifiers + self.num_intensifiers)
        weight_matrix.append(initial_diversifier_condition_row)
        initial_intensifier_condition_row = [1] * (self.num_diversifiers + self.num_intensifiers)
        weight_matrix.append(initial_intensifier_condition_row)
        for i in range(self.num_intensifiers):
            intensifier_condition_row = [1] * (self.num_diversifiers + self.num_intensifiers)
            weight_matrix.append(intensifier_condition_row)
        final_diversifier_condition_row = [1] * (self.num_diversifiers + self.num_intensifiers)
        weight_matrix.append(final_diversifier_condition_row)
        return weight_matrix

    def pack_weights(self, id):
        """
        Packs a 2D weight matrix (as a list of lists) into a Weights.msg-compatible format.
        """
        rows = len(self.weights)  # Number of rows
        cols = len(self.weights[0]) if rows > 0 else 0  # Number of columns

        # Flatten and ensure all elements are floats
        flattened_weights = [float(value) for row in self.weights for value in row]

        weights_msg = {
            "id": id,
            "rows": rows,
            "cols": cols,
            "weights": flattened_weights,
        }
        return weights_msg

    def unpack_weights(self, weights_msg, agent_id):
        """
        Unpacks a Weights.msg-compatible format into a 2D weight matrix (list of lists).
        """
        if weights_msg.id != agent_id:
            rows = weights_msg.rows
            cols = weights_msg.cols
            weights_flat = weights_msg.weights
            # Recreate the 2D list
            weight_matrix = [weights_flat[i * cols:(i + 1) * cols] for i in range(rows)]
            return weight_matrix
        else:
            return None
