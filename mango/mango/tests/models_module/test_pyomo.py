import os
from copy import deepcopy
from unittest import TestCase

from mango.models.pyomo import (
    var_to_table,
    instance_from_excel,
    var_to_sdict,
    safe_value,
    load_variable,
    is_feasible,
    get_status,
    get_gap,
    variables_to_excel,
    model_data_to_excel,
    instance_from_json,
    variables_to_json,
    solver_result_to_json,
    solver_result_from_json,
)
from mango.processing import load_excel_light
from mango.table import Table
from mango.tests.const import normalize_path
from pyomo.core import AbstractModel, Set, Var
from pyomo.environ import Binary


class TestTools(TestCase):
    def setUp(self):
        def create_model():
            model = AbstractModel()
            model.sSet = Set()
            model.vVariable = Var(model.sSet, domain=Binary, initialize=0)
            return model

        self.model = create_model()
        self.data = {"sSet": {None: ["01", "02", "03"]}}
        self.instance = self.model.create_instance({None: self.data})

    def test_is_feasible_True(self):
        result = is_feasible("optimal")
        self.assertTrue(result, msg="Optimal solution was found")

    def test_is_feasible_False(self):
        result = is_feasible("infeasible")
        self.assertFalse(result, msg="Optimal solution wasn`t found")

    def test_safe_value(self):
        result = safe_value(self.instance.vVariable)
        self.assertEqual(result, 0, msg="When variable has not been used return 0")

    def test_var_to_sdict(self):
        result = var_to_sdict(self.instance.vVariable)
        expected = {"01": 0, "02": 0, "03": 0}
        self.assertEqual(
            result, expected, msg="convert instance variable to dictionary"
        )

    def test_var_to_table(self):
        result = var_to_table(
            self.instance.vVariable,
            ["sSet", "Value"],
        )
        expected = [
            {"sSet": "01", "Value": 0},
            {"sSet": "02", "Value": 0},
            {"sSet": "03", "Value": 0},
        ]
        self.assertEqual(result, expected, msg="convert instance variable to Table")

    def test_get_status(self):
        path = normalize_path("./data/result.json")
        result = solver_result_from_json(path)
        status = get_status(result)
        self.assertEqual(status, "optimal", msg="Resolution status is: optimal")

    def test_get_gap(self):
        path = normalize_path("./data/result.json")
        result = solver_result_from_json(path)
        gap = get_gap(result)
        self.assertEqual(gap, 0, msg="the relative gap of the solution")

    def test_variables_to_excel(self):
        path = normalize_path("./data/")
        instance = instance_from_excel(deepcopy(self.instance), path)
        expected = var_to_table(
            instance.vVariable,
            ["sSet", "Value"],
        )
        variables_to_excel(instance, path, file_name="variables_test.xlsx")

        instance_result = instance_from_excel(
            deepcopy(self.instance), path, file_name="variables_test.xlsx"
        )
        result = var_to_table(
            instance_result.vVariable,
            ["sSet", "Value"],
        )
        self.assertEqual(result, expected, msg="save variables to excel file")
        file = normalize_path("./data/variables_test.xlsx")
        os.remove(file)

    def test_variables_to_json(self):
        path = normalize_path("./data/")
        instance = instance_from_json(deepcopy(self.instance), path)
        variables_to_json(instance, path)
        expected = var_to_table(
            instance.vVariable,
            ["sSet", "Value"],
        )
        instance_result = instance_from_json(deepcopy(self.instance), path)
        result = var_to_table(
            instance_result.vVariable,
            ["sSet", "Value"],
        )
        self.assertEqual(result, expected, msg="save variables to json file")

    def test_instance_from_excel(self):
        path = normalize_path("./data/")
        instance = instance_from_excel(deepcopy(self.instance), path)
        result = var_to_sdict(instance.vVariable)
        expected = {"01": 1, "02": 1, "03": 1}
        self.assertEqual(result, expected, msg="read instance from excel")

    def test_instance_from_json(self):
        path = normalize_path("./data/")
        instance = instance_from_json(deepcopy(self.instance), path)
        result = var_to_sdict(instance.vVariable)
        expected = {"01": 1, "02": 1, "03": 1}
        self.assertEqual(result, expected, msg="read instance from excel")

    def test_load_variable(self):
        instance = deepcopy(self.instance)
        load_variable(instance, "vVariable", {"01": 1, "02": 0, "03": 1})
        result = var_to_sdict(instance.vVariable)
        expected = {"01": 1, "02": 0, "03": 1}
        self.assertEqual(result, expected, msg="change instance variable values")

    def test_model_data_to_excel(self):
        path = normalize_path("./data/")
        model_data_to_excel(self.data, path)

        file = normalize_path("./data/model_data.xlsx")
        data = load_excel_light(file)
        self.assertEqual(data.keys(), self.data.keys(), "write model data to excel")
        os.remove(file)

    def test_solver_result_to_json(self):
        path = normalize_path("./data/result.json")
        expected = solver_result_from_json(path)
        solver_result_to_json(expected, path)
        result = solver_result_from_json(path)
        self.assertEqual(Table(result), Table(expected), "write pyomo result to json")

    def test_solver_result_from_json(self):
        path = normalize_path("./data/result.json")
        result = solver_result_from_json(path)
        self.assertEqual(
            str(result.problem.sense), "maximize", "read pyomo result from json"
        )
