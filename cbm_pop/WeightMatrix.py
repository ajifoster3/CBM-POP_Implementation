
class WeightMatrix:
    def __init__(self, num_intensifiers, num_diversifiers, free_weight_matrix):
        self.num_diversifiers = num_diversifiers
        self.num_intensifiers = num_intensifiers
        if free_weight_matrix:
            self.weights = self.init_free_weight_matrix()
        else:
            self.weights = self.init_weight_matrix()


    def init_weight_matrix(self):
        """
        Generates a weight matrix mapping conditions onto operations.
        """
        print("Initialising Classical Weight Matrix")

        # Initialize and return a weight matrix (for operator selection, if needed)
        weight_matrix = []
        initial_diversifier_condition_row = [0.0] * self.num_intensifiers + [1.0] * self.num_diversifiers
        weight_matrix.append(initial_diversifier_condition_row)
        initial_intensifier_condition_row = [1.0] * self.num_intensifiers + [0.0] * self.num_diversifiers
        weight_matrix.append(initial_intensifier_condition_row)
        for i in range(self.num_intensifiers):
            intensifier_condition_row = [1.0] * self.num_intensifiers + [0.0] * self.num_diversifiers
            intensifier_condition_row[i] = 0.0
            weight_matrix.append(intensifier_condition_row)
        print(f"Classical weights: {weight_matrix}")
        return weight_matrix

    def apply_classical_mask(self):
        """
        Applies the classical initial weight mask to the current weights.
        Any entry that is 0 in the classical initialisation will be set to 0,
        but already-learned values in allowed positions are preserved.
        """
        classical = self.init_weight_matrix()  # get the classical 0/1 mask
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                if classical[i][j] == 0:
                    self.weights[i][j] = 0.0

        print(f"{self.weights}")

    def init_free_weight_matrix(self):
        """
        Generates a weight matrix mapping conditions onto operations.
        """
        print("Initialising Classical Weight Matrix")

        # Initialize and return a weight matrix (for operator selection, if needed)
        weight_matrix = []
        initial_diversifier_condition_row = [1.0] * self.num_intensifiers + [1.0] * self.num_diversifiers
        weight_matrix.append(initial_diversifier_condition_row)
        initial_intensifier_condition_row = [1.0] * self.num_intensifiers + [1.0] * self.num_diversifiers
        weight_matrix.append(initial_intensifier_condition_row)
        for i in range(self.num_intensifiers):
            intensifier_condition_row = [1.0] * self.num_intensifiers + [1.0] * self.num_diversifiers
            intensifier_condition_row[i] = 1.0
            weight_matrix.append(intensifier_condition_row)
        print(f"Classical weights: {weight_matrix}")
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
