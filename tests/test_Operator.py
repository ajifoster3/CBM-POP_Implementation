import unittest
from unittest import TestCase

from cbm_pop.Operator import OperatorFunctions
from WeightMatrix import*
from Condition import*

class TestOperatorFunctions(TestCase):
    def test_choose_operator_case_condition_0(self):
        weights = WeightMatrix(2, 4)
        for i in range(30):
            operator = OperatorFunctions.choose_operator(weights=weights.weights, condition=Condition.C_0)
            self.assertTrue(operator.value >= 3)
            self.assertFalse(operator.value < 3)

    @unittest.skip("Not yet implemented")
    def test_apply_op(self):
        self.fail() #TODO: Implement maybe

    @unittest.skip("Not yet implemented")
    def test_best_cost_route_crossover(self):
        self.fail() #TODO: Implement maybe

    def test_find_best_task_position(self):

        cost_matrix = [[0, 1, 8, 9],
                       [1, 0, 9, 8],
                       [8, 9, 0, 0.5],
                       [9, 8, 1, 0]]
        solution_task_counts = [2, 1]
        solution_task_order = [0, 1, 3]
        OperatorFunctions.find_best_task_position(
            task=2,
            new_solution_task_counts=solution_task_counts,
            new_solution_task_order=solution_task_order,
            cost_matrix=cost_matrix)
        self.assertEqual(solution_task_order, [0, 1, 2, 3])
        self.assertEqual(solution_task_counts, [2, 2])

        cost_matrix = [[0, 1, 8, 9],
                       [1, 0, 9, 8],
                       [8, 9, 0, 1],
                       [9, 8, 0.5, 0]]

        solution_task_counts = [2, 1]
        solution_task_order = [0, 1, 3]
        OperatorFunctions.find_best_task_position(
            task=2,
            new_solution_task_counts=solution_task_counts,
            new_solution_task_order=solution_task_order,
            cost_matrix=cost_matrix)
        self.assertEqual(solution_task_order, [0, 1, 3, 2])
        self.assertEqual(solution_task_counts, [2, 2])

        cost_matrix = [[0, 1, 8, 9],
                       [1, 0, 9, 8],
                       [0.5, 9, 0, 9],
                       [9, 8, 9, 0]]

        solution_task_counts = [2, 1]
        solution_task_order = [0, 1, 3]
        OperatorFunctions.find_best_task_position(
            task=2,
            new_solution_task_counts=solution_task_counts,
            new_solution_task_order=solution_task_order,
            cost_matrix=cost_matrix)
        self.assertEqual(solution_task_order, [2, 0, 1, 3])
        self.assertEqual(solution_task_counts, [3, 1])

    @unittest.skip("Not yet implemented")
    def test_intra_depot_removal(self):
        self.fail()

    @unittest.skip("Not yet implemented")
    def test_intra_depot_swapping(self):
        self.fail()

    @unittest.skip("Not yet implemented")
    def test_inter_depot_swapping(self):
        self.fail()

    @unittest.skip("Not yet implemented")
    def test_single_action_rerouting(self):
        self.fail()

    def test_one_move(self):
        solution_task_counts = [2, 2]
        solution_task_order = [0, 1, 3, 2]
        solution = [solution_task_order, solution_task_counts]
        cost_matrix = [[0, 1, 8, 9],
                       [1, 0, 9, 8],
                       [8, 9, 0, 0.5],
                       [9, 8, 1, 0]]
        task_order, task_counts = OperatorFunctions.one_move(solution, cost_matrix)
        self.assertEqual(task_order, [0, 1, 2, 3])
        self.assertEqual(task_counts, [2, 2])

        solution_task_counts = [2, 2]
        solution_task_order = [0, 1, 3, 2]
        solution = [solution_task_order, solution_task_counts]
        cost_matrix = [[0, 1, 8, 9],
                       [1, 0, 9, 8],
                       [0.5, 9, 0, 9],
                       [9, 8, 9, 0]]
        task_order, task_counts = OperatorFunctions.one_move(solution, cost_matrix)
        self.assertEqual(task_order, [2, 0, 1, 3])
        self.assertEqual(task_counts, [3, 1])

    def test_two_swap(self):
        solution_task_counts = [4, 4]
        solution_task_order = [0, 5, 6, 3, 4, 1, 2, 7]
        solution = [solution_task_order, solution_task_counts]
        cost_matrix = [[0, 3, 4, 6, 13, 13, 13, 14],
                       [3, 0, 1, 4, 13, 13, 13, 14],
                       [4, 1, 0, 3, 13, 13, 13, 14],
                       [6, 3, 4, 0, 13, 13, 13, 14],
                       [13, 13, 13, 14, 6, 3, 4, 0],
                       [13, 13, 13, 14, 4, 1, 0, 3],
                       [13, 13, 13, 14, 3, 0, 1, 4],
                       [13, 13, 13, 14, 0, 3, 4, 6]]
        task_order, task_counts = OperatorFunctions.two_swap(solution, cost_matrix)
        self.assertEqual(task_order, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(task_counts, [4, 4])

        solution_task_counts = [4, 4]
        solution_task_order = [4, 1, 2, 7, 0, 5, 6, 3]
        solution = [solution_task_order, solution_task_counts]
        cost_matrix = [[0, 3, 4, 6, 13, 13, 13, 14],
                       [3, 0, 1, 4, 13, 13, 13, 14],
                       [4, 1, 0, 3, 13, 13, 13, 14],
                       [6, 3, 4, 0, 13, 13, 13, 14],
                       [13, 13, 13, 14, 6, 3, 4, 0],
                       [13, 13, 13, 14, 4, 1, 0, 3],
                       [13, 13, 13, 14, 3, 0, 1, 4],
                       [13, 13, 13, 14, 0, 3, 4, 6]]
        task_order, task_counts = OperatorFunctions.two_swap(solution, cost_matrix)
        self.assertEqual(task_order, [4, 5, 6, 7, 0, 1, 2, 3])
        self.assertEqual(task_counts, [4, 4])
