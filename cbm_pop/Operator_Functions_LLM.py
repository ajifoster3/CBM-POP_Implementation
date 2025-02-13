import random
import sys
import traceback
from copy import deepcopy
from enum import Enum
from typing import Optional, Dict, List
import json
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator import Operator
class OperatorFunctionsLLM:

    def __init__(self, operator_functions_code: dict = None):

        if operator_functions_code is None:
            operator_functions_code = self.load_operator_function_codes("/home/ajifoster3/Documents/Software/ros2playground/src/CBM-POP/cbm_pop/GUI/function_codes.json")

        # Load the functions dynamically
        self.operator_function_map, self.initial_failed_functions, self.operator_function_code = self._load_operator_functions(
            operator_functions_code["operator_function_code"])
        #self.log_function_codes()

    def _load_operator_functions(self, operator_functions_code):
        """
        Dynamically loads Python functions from a dictionary of code strings.
        Returns:
            - operator_function_map: Dictionary mapping operators to lists of loaded functions.
            - failed_function_map: Dictionary mapping operators to lists of failed function indexes.
        """
        operator_function_code_map = {}
        operator_function_map = {}
        failed_function_map = {}

        for operator, function_codes in operator_functions_code.items():

            operator_function_map[operator] = []
            operator_function_code_map[operator] = []
            failed_function_map[operator] = []  # Initialize failed indexes list
            i = 0
            for code in function_codes:
                try:
                    operator_function_code_map[operator].append(code)
                    code = code.replace("''' python\n", "").strip("'''")
                    # Create a local namespace to store the function
                    local_namespace = {}
                    # Execute the code to define the function
                    exec(code, globals(), local_namespace)
                    # Retrieve the function from the local namespace
                    function_name = operator.name.lower() + "_o" + str(i)
                    function = local_namespace[function_name]
                    # Add the function to the operator's list
                    operator_function_map[operator].append(function)
                except Exception as e:
                    print(f"Failed to load function for operator {operator}, index {i}: {e}")
                    failed_function_map[operator].append((i, e))  # Store the failed index
                i += 1  # Increment index regardless of success or failure

        return operator_function_map, failed_function_map, operator_function_code_map

    def load_new_function(self, operator, code, index):
        try:
            self.operator_function_code[operator][index] = code
            code = code.replace("''' python\n", "").strip("'''")

            # Define function name
            function_name = operator.name.lower() + "_o" + str(index)
            # Remove old function from global namespace if it exists
            if function_name in globals():
                del globals()[function_name]

            # Create a local namespace to store the function
            local_namespace = {}
            # Execute the code to define the function
            exec(code, globals(), local_namespace)

            # Retrieve the function from the local namespace
            function = local_namespace[function_name]

            # Add the function to the operator's list
            if operator not in self.operator_function_map:
                self.operator_function_map[operator] = []
            self.operator_function_map[operator][index] = function
            return True
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

            print(f"An error occurred:\n{traceback_details}", file=sys.stderr)

    def choose_operator(self, weights, condition):
        """
        Stochastically choose an operator for a condition using the weights.
        :param weights: Weights
        :param condition: Current condition
        :return: The chosen operator
        """
        # Choose an operator (e.g., mutation, crossover) based on weight matrix W and current state
        # For simplicity, we only apply a mutation operator in this example
        row = weights[condition.value]
        operators = list(Operator)

        # Randomly select an operator based on weights in `row`
        chosen_operator = random.choices(operators, weights=row, k=1)[0]
        return chosen_operator

    def apply_op(
            self,
            operator: Operator,
            current_solution,
            population,
            cost_matrix,
            failed_operator_dict,
            cycle_operator_index
    ) -> tuple:
        """
        Apply the operator to the current solution and return the newly generated child solution.
        :param cost_matrix: Cost matrix to calculate the cost
        :param operator: The operator to be applied
        :param current_solution: The current solution
        :param population: The population
        :param failed_operator_dict: Dictionary tracking failed function indexes per operator
        :param cycle_operator_index
        :return: A tuple containing the child solution and the index of the selected function
        """
        if operator not in self.operator_function_map:
            raise ValueError(f"Operator {operator} not found in operator_function_map")

        functions = self.operator_function_map[operator]
        if not functions:
            raise ValueError(f"No functions registered for operator {operator}")
        failed_indexes = [tpl[0] for tpl in failed_operator_dict[operator]]
        if cycle_operator_index == -1 or failed_indexes.__contains__(cycle_operator_index):

            # Get valid indexes (exclude previously failed ones)
            available_indexes = [i for i in range(len(functions)) if i not in failed_indexes]

            # Fallback to all indexes if none are available (all have failed)
            if not available_indexes:
                print("No available indexes, falling back to all available indexes.")
                available_indexes = list(range(len(functions)))

            # Randomly select an index from available candidates
            cycle_operator_index = random.choice(available_indexes)
        if functions[cycle_operator_index]:
            selected_function = functions[cycle_operator_index]
        else:
            print("************ Maybe something funny ************")
            print(functions[cycle_operator_index])
        try:
            # Execute the selected function with appropriate arguments
            if operator == Operator.BEST_COST_ROUTE_CROSSOVER:
                child_solution = selected_function(current_solution, population, cost_matrix)
            elif operator in {Operator.SINGLE_ACTION_REROUTING, Operator.TWO_SWAP, Operator.ONE_MOVE}:
                child_solution = selected_function(current_solution, cost_matrix)
            else:
                child_solution = selected_function(current_solution)

            return child_solution, cycle_operator_index

        except Exception as e:
            # Record failed index if tracking is enabled
            if not any(existing[0] == cycle_operator_index for existing in failed_operator_dict[operator]):
                failed_operator_dict[operator].append((cycle_operator_index, e))

            return None, cycle_operator_index, e

    @staticmethod
    def find_best_task_position(cost_matrix, new_solution_task_counts, new_solution_task_order, task):
        best_fitness = float('inf')
        best_position = 0
        best_agent = 0
        # Try inserting the task at each position in the task order
        for agent_index, count in enumerate(new_solution_task_counts):
            # Calculate the insertion range for this agent
            agent_start_index = sum(new_solution_task_counts[:agent_index])
            agent_end_index = agent_start_index + count

            # Try inserting within this agent's range
            for pos in range(agent_start_index, agent_end_index + 1):
                temp_order = new_solution_task_order[:]
                temp_order.insert(pos, task)

                # Update task counts temporarily for fitness calculation
                temp_counts = new_solution_task_counts[:]
                temp_counts[agent_index] += 1

                # Calculate fitness with this temporary insertion
                temp_fitness = Fitness.fitness_function((temp_order, temp_counts), cost_matrix)

                # If the new fitness is better, update the best fitness, position, and agent
                if temp_fitness < best_fitness:
                    best_fitness = temp_fitness
                    best_position = pos
                    best_agent = agent_index
        # Insert the task at the best position found and update the task count for that agent
        new_solution_task_order.insert(best_position, task)
        new_solution_task_counts[best_agent] += 1

    def log_function_codes(self):
        json_file_path = 'function_codes.json'
        data_to_log = {"operator_function_code": self.operator_function_code}

        # Convert all keys in `operator_function_code` dictionaries to strings
        data_to_log = self.convert_keys_to_strings(data_to_log)

        with open(json_file_path, 'w') as f:
            json.dump(data_to_log, f, indent=4)

        print(f"Data has been logged to {json_file_path}")

    def convert_keys_to_strings(self, obj):
        """
        Recursively convert all dictionary keys that aren't JSON-valid
        (string, int, float, bool, None) into strings.
        """
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # If the key isn't a valid JSON key type, convert to string.
                if not isinstance(k, (str, int, float, bool, type(None))):
                    new_k = str(k)
                else:
                    new_k = k
                new_dict[new_k] = self.convert_keys_to_strings(v)
            return new_dict
        elif isinstance(obj, list):
            return [self.convert_keys_to_strings(item) for item in obj]
        else:
            return obj

    def load_operator_function_codes(self, json_file_path: str) -> dict:
        """
        Loads the JSON from file and returns a dictionary where any keys
        that start with 'Operator.' are converted to actual Operator enum members.

        Example JSON structure:
        {
            "operator_function_code": {
                "Operator.TWO_SWAP": ["some", "values"],
                "Operator.THREE_SWAP": ["other", "values"]
            }
        }
        """
        # 1. Read the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # 2. Recursively convert any 'Operator.SOMETHING' string keys
        converted_data = self._convert_operator_keys(data)

        return converted_data

    def _convert_operator_keys(self, obj):
        """
        Recursively convert dictionary keys from string (e.g., 'Operator.TWO_SWAP')
        into the corresponding Operator enum member (Operator.TWO_SWAP).
        """
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                # Convert the key if it matches the pattern 'Operator.<enum_member>'
                new_key = self._parse_operator_key(key)
                # Recursively handle the value
                new_dict[new_key] = self._convert_operator_keys(value)
            return new_dict
        elif isinstance(obj, list):
            # If it's a list, apply the conversion to each element
            return [self._convert_operator_keys(item) for item in obj]
        else:
            # For values that are not dict or list, just return them as-is
            return obj

    def _parse_operator_key(self, key):
        """
        If 'key' is a string like 'Operator.TWO_SWAP', convert it to
        the enum member Operator.TWO_SWAP. Otherwise, leave it as is.
        """
        if isinstance(key, str) and key.startswith("Operator."):
            # Extract the part after 'Operator.'
            enum_name = key.split('.', 1)[1]
            # Convert to the enum member: e.g. Operator["TWO_SWAP"]
            return Operator[enum_name]
        return key