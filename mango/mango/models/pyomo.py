from mango.processing import (
    as_list,
    load_json,
    write_json,
    load_excel_light,
    write_excel_light,
)
from mango.table import Table
from pyomo.core import value, Var
from pyomo.opt import (
    SolverResults,
    UndefinedData,
    SolverStatus,
    TerminationCondition,
    SolutionStatus,
)
from pytups import SuperDict


def is_feasible(status):
    """
    Return True if the status is optimal or maxTimeLimit

    :param status: a status (string or pyomo object)
    :return: True if the status is optimal or maxTimeLimit
    """
    return str(status) == str(TerminationCondition.optimal) or str(status) == str(
        TerminationCondition.maxTimeLimit
    )


def safe_value(x):
    """
    Safely apply pyomo value to a variable.

    pyomo value generate an error if the variable has not been used.
    This function will return 0 instead.

    :param x: a pyomo variable
    :return: value of the variable
    """
    try:
        if x is not None:
            return value(x)
        else:
            return 0
    except:
        return 0


def var_to_sdict(variable, clean=False, rounding=2):
    """
    Transform a pyomo variable into a python dict

    :param variable: a pyomo variable.
    :param clean: if true, only return non-zero values.
    :param rounding: number of decimal to round.
    :return: a SuperDict containing the indices and values of the variable.
    """
    new_dict = SuperDict(
        {key: round(safe_value(variable[key]), rounding) for key in variable.keys()}
    )
    if clean:
        return new_dict.clean()
    else:
        return new_dict


def var_to_table(variable, keys, clean=False, rounding=2):
    """
    Generate a table from a pyomo variable.

    :param variable: a pyomo variable.
    :param keys: the names of the keys of the variable.
    :param clean: if true, only return non-zero values.
    :param rounding: number of decimal to round.
    :return: a table [{Index_1:i1, Index_2:i2, Value:v}...]
    """
    var_tup = var_to_sdict(variable, clean=clean, rounding=rounding).to_tuplist()
    if keys is None:
        if len(var_tup) > 0 and len(var_tup[0]) > 1:
            keys = ["Index_" + str(i) for i in range(1, len(var_tup[0]))] + ["Value"]
        else:
            keys = ["Value"]

    return Table(var_tup.to_dictlist(keys))


def get_status(result):
    """
    Return the status of the solution from the result object

    :param result: a pyomo result object
    :return: the status
    """
    return str(result.solver.termination_condition)


def get_gap(result):
    """
    Return the relative gap of the solution.

    :param result: a pyomo result object
    :return: the gap
    """
    sense = result.Problem.sense
    lb = result.Problem.lower_bound
    ub = result.Problem.upper_bound

    if ub == lb:
        return 0
    if sense == "minimize":
        if ub == 0:
            ub = 10**-6
        gap = round(100 * (ub - lb) / ub, 3)
    else:
        if lb == 0:
            lb = 10**-6
        gap = round(100 * (lb - ub) / lb, 3)
    return abs(gap)


def write_cbc_warmstart_file(filename, instance, opt):
    """
    This function write a file to be passed to cbc solver as a warmstart file.
    This function is necessary because of a bug of cbc that does not allow reading warmstart files on windows
    with absolute path.
    :param filename: path to the file
    :param instance: model instance (created with create_instance)
    :param opt: solver instance (created with solver factory)
    :return:
    """
    opt._presolve(instance)
    opt._write_soln_file(instance, filename)


def variables_to_excel(
    model_solution, path, file_name="variables.xlsx", clean=True, rounding=6
):
    """
    Save all the values of the variables in an Excel file.

    :param model_solution: the pyomo solution.
    :param path: path of the directory where to save the file
    :param clean: discard variables with values of 0.

    :return: nothing
    """
    variables = {
        k: var_to_table(v, keys=None, clean=clean, rounding=rounding)
        for k, v in model_solution.component_map(ctype=Var).items()
    }
    write_excel_light(path + file_name, variables)


def variables_to_json(model_solution, path, file_name="variables.json", clean=True):
    """
    Save all the values of the variables in an json file.

    :param model_solution: the pyomo solution.
    :param path: path of the directory where to save the file
    :param clean: discard variables with values of 0.

    :return: nothing
    """
    variables = {
        k: var_to_table(v, keys=None, clean=clean)
        for k, v in model_solution.component_map(ctype=Var).items()
    }
    write_json(variables, path + file_name)


def instance_from_excel(instance, path, file_name="variables.xlsx"):
    """
    The function reads instance from excel.

    :param instance: model instance
    :param path: the path of the file to be loaded
    :return: model instance
    """
    data = load_excel_light(path + file_name)
    return instance_from_dict(instance, data)


def instance_from_json(instance, path, file_name="variables.json", **kwargs):
    """
    The function reads instance from json.

    :param instance: model instance
    :param path: the path of the file to be loaded
    :return: model instance
    """
    data = load_json(path + file_name, **kwargs)
    return instance_from_dict(instance, data)


def instance_from_dict(instance, data):
    """
    The function loads the instance based on data from dict with tables.

    :param instance: model instance
    :param data: data for instance
    :return: model instance
    """
    model_data = {}
    for k, v in data.items():
        v = Table(v)
        if len(v) > 0:
            indices = [k for k in v[0] if k != "Value"]
            model_data[k] = v.to_param(indices, "Value")
            load_variable(instance, k, model_data[k])
    return instance


def load_variable(instance, var_name, value):
    """
    The function loads the values to indicated variable.

    :param instance: model instance
    :param var_name: name of the variable to which values will be assign
    :param value: values to be assigned to variable
    """
    var = instance.__dict__[var_name]
    for k, v in value.items():
        var[k] = v


def model_data_to_excel(model_data, path, file_name="model_data.xlsx"):
    """
    Save a pyomo instance to excel.

    :param model_data: dict with model data for pyomo model.
    :param path: path of the directory where to save the file
    :param clean: discard variables with values of 0.

    :return: nothing
    """

    def prepare_data(v):
        if None in v:
            v = v[None]
            return [
                {"Index_" + str(i): as_list(k)[i] for i in range(len(as_list(k)))}
                for k in as_list(v)
            ]
        else:
            return [
                {
                    **{
                        "Index_" + str(i): as_list(k)[i] for i in range(len(as_list(k)))
                    },
                    **{"value": v2},
                }
                for k, v2 in v.items()
            ]

    data = model_data
    tables = {k: prepare_data(v) for k, v in data.items()}
    write_excel_light(path + file_name, tables)


def solver_result_to_json(result, path):
    """
    Save the result object in a json file.

    :param result: result object from pyomo
    :param path: path of the file
    :return: nothing
    """

    def get_val(v):
        try:
            if isinstance(v.value, UndefinedData):
                return None
            if isinstance(
                v.value, (SolverStatus, TerminationCondition, SolutionStatus)
            ):
                return str(v.value)
            return v.value
        except AttributeError:
            return None

    data = {
        k: Table([{k1: get_val(v1) for k1, v1 in v.items()}]) for k, v in result.items()
    }
    write_json(data, path)


def solver_result_from_json(path, **kwargs):
    """
    Load a result object from pyomo

    :param path: path of the file
    :param kwargs: kwargs for load_json
    :return: the result object
    """
    data = load_json(path, **kwargs)
    result = SolverResults()
    for k, v in result.items():
        for k1, v1 in v.items():
            if data[k][0][k1] is not None:
                v1.value = data[k][0][k1]
    return result
