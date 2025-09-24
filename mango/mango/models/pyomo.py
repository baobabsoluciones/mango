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
    Check if a solver status indicates a feasible solution.

    Determines whether the given status represents a feasible solution
    by checking if it corresponds to optimal or maximum time limit
    termination conditions.

    :param status: Solver status (string or pyomo object)
    :type status: Union[str, TerminationCondition]
    :return: True if the status indicates a feasible solution
    :rtype: bool

    Example:
        >>> result = solver.solve(model)
        >>> if is_feasible(result.solver.termination_condition):
        ...     print("Solution is feasible")
    """
    return str(status) == str(TerminationCondition.optimal) or str(status) == str(
        TerminationCondition.maxTimeLimit
    )


def safe_value(x):
    """
    Safely extract the value from a Pyomo variable.

    Pyomo's value() function generates an error if the variable has not
    been used in the model. This function provides a safe alternative
    that returns 0 for unused or None variables.

    :param x: Pyomo variable to extract value from
    :type x: Union[Var, None]
    :return: Value of the variable, or 0 if variable is None or unused
    :rtype: float
    :raises Exception: Handles any Pyomo value extraction errors gracefully

    Example:
        >>> var = model.x[1, 2]
        >>> val = safe_value(var)  # Returns 0 if var was not used
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
    Transform a Pyomo variable into a SuperDict with indices and values.

    Converts a Pyomo variable into a dictionary-like structure containing
    all variable indices and their corresponding values. Optionally filters
    out zero values and rounds the results.

    :param variable: Pyomo variable to convert
    :type variable: Var
    :param clean: If True, only return non-zero values
    :type clean: bool
    :param rounding: Number of decimal places to round values
    :type rounding: int
    :return: SuperDict containing variable indices and values
    :rtype: SuperDict

    Example:
        >>> var = model.x  # Pyomo variable with indices (1,2), (2,3)
        >>> result = var_to_sdict(var, clean=True, rounding=3)
        >>> print(result)  # {(1,2): 1.234, (2,3): 5.678}
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
    Generate a Table object from a Pyomo variable.

    Converts a Pyomo variable into a Table structure with named columns
    for indices and values. Automatically generates column names if not
    provided.

    :param variable: Pyomo variable to convert
    :type variable: Var
    :param keys: Names for the index columns (None for auto-generation)
    :type keys: Optional[List[str]]
    :param clean: If True, only return non-zero values
    :type clean: bool
    :param rounding: Number of decimal places to round values
    :type rounding: int
    :return: Table object with variable data
    :rtype: Table

    Example:
        >>> var = model.x  # Pyomo variable with indices (i, j)
        >>> table = var_to_table(var, keys=['i', 'j'], clean=True)
        >>> print(table)  # [{'i': 1, 'j': 2, 'Value': 1.23}, ...]
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
    Extract the termination condition status from a Pyomo solver result.

    Retrieves the termination condition from the solver result object
    and returns it as a string representation.

    :param result: Pyomo solver result object
    :type result: SolverResults
    :return: String representation of the termination condition
    :rtype: str

    Example:
        >>> result = solver.solve(model)
        >>> status = get_status(result)
        >>> print(f"Solver status: {status}")
    """
    return str(result.solver.termination_condition)


def get_gap(result):
    """
    Calculate the relative optimality gap from a Pyomo solver result.

    Computes the relative gap between the upper and lower bounds of the
    solution, handling both minimization and maximization problems.
    Returns the absolute value of the gap as a percentage.

    :param result: Pyomo solver result object
    :type result: SolverResults
    :return: Relative optimality gap as a percentage
    :rtype: float

    Example:
        >>> result = solver.solve(model)
        >>> gap = get_gap(result)
        >>> print(f"Optimality gap: {gap}%")
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
    Write a warmstart file for the CBC solver.

    Creates a warmstart file that can be used by the CBC solver to
    initialize the solution process. This function is necessary due to
    a bug in CBC that prevents reading warmstart files with absolute
    paths on Windows systems.

    :param filename: Path where the warmstart file should be written
    :type filename: str
    :param instance: Pyomo model instance (created with create_instance)
    :type instance: ConcreteModel
    :param opt: Solver instance (created with solver factory)
    :type opt: SolverFactory
    :return: None

    Example:
        >>> opt = SolverFactory('cbc')
        >>> write_cbc_warmstart_file('warmstart.sol', model, opt)
    """
    opt._presolve(instance)
    opt._write_soln_file(instance, filename)


def variables_to_excel(
    model_solution, path, file_name="variables.xlsx", clean=True, rounding=6
):
    """
    Export all Pyomo variable values to an Excel file.

    Saves all variables from a Pyomo model solution to an Excel file
    with each variable as a separate worksheet. Optionally filters out
    zero values and rounds the results.

    :param model_solution: Pyomo model solution object
    :type model_solution: ConcreteModel
    :param path: Directory path where the file should be saved
    :type path: str
    :param file_name: Name of the Excel file to create
    :type file_name: str
    :param clean: If True, discard variables with zero values
    :type clean: bool
    :param rounding: Number of decimal places to round values
    :type rounding: int
    :return: None
    :raises IOError: If file cannot be written

    Example:
        >>> variables_to_excel(
        ...     model, 'output/', 'solution.xlsx', clean=True, rounding=4
        ... )
    """
    variables = {
        k: var_to_table(v, keys=None, clean=clean, rounding=rounding)
        for k, v in model_solution.component_map(ctype=Var).items()
    }
    write_excel_light(path + file_name, variables)


def variables_to_json(model_solution, path, file_name="variables.json", clean=True):
    """
    Export all Pyomo variable values to a JSON file.

    Saves all variables from a Pyomo model solution to a JSON file
    with structured data. Optionally filters out zero values.

    :param model_solution: Pyomo model solution object
    :type model_solution: ConcreteModel
    :param path: Directory path where the file should be saved
    :type path: str
    :param file_name: Name of the JSON file to create
    :type file_name: str
    :param clean: If True, discard variables with zero values
    :type clean: bool
    :return: None
    :raises IOError: If file cannot be written

    Example:
        >>> variables_to_json(model, 'output/', 'solution.json', clean=True)
    """
    variables = {
        k: var_to_table(v, keys=None, clean=clean)
        for k, v in model_solution.component_map(ctype=Var).items()
    }
    write_json(variables, path + file_name)


def instance_from_excel(instance, path, file_name="variables.xlsx"):
    """
    Load Pyomo model instance data from an Excel file.

    Reads variable values from an Excel file and loads them into a
    Pyomo model instance. The Excel file should contain worksheets
    with variable data in the format created by variables_to_excel.

    :param instance: Pyomo model instance to load data into
    :type instance: ConcreteModel
    :param path: Directory path containing the Excel file
    :type path: str
    :param file_name: Name of the Excel file to load
    :type file_name: str
    :return: Model instance with loaded variable values
    :rtype: ConcreteModel
    :raises IOError: If file cannot be read

    Example:
        >>> model = instance_from_excel(model, 'data/', 'initial_values.xlsx')
    """
    data = load_excel_light(path + file_name)
    return instance_from_dict(instance, data)


def instance_from_json(instance, path, file_name="variables.json", **kwargs):
    """
    Load Pyomo model instance data from a JSON file.

    Reads variable values from a JSON file and loads them into a
    Pyomo model instance. The JSON file should contain structured
    data in the format created by variables_to_json.

    :param instance: Pyomo model instance to load data into
    :type instance: ConcreteModel
    :param path: Directory path containing the JSON file
    :type path: str
    :param file_name: Name of the JSON file to load
    :type file_name: str
    :param kwargs: Additional keyword arguments for load_json
    :return: Model instance with loaded variable values
    :rtype: ConcreteModel
    :raises IOError: If file cannot be read

    Example:
        >>> model = instance_from_json(model, 'data/', 'initial_values.json')
    """
    data = load_json(path + file_name, **kwargs)
    return instance_from_dict(instance, data)


def instance_from_dict(instance, data):
    """
    Load Pyomo model instance data from a dictionary containing tables.

    Processes a dictionary containing table data and loads the values
    into the corresponding variables in a Pyomo model instance.
    Each table should have index columns and a 'Value' column.

    :param instance: Pyomo model instance to load data into
    :type instance: ConcreteModel
    :param data: Dictionary containing table data for variables
    :type data: Dict[str, List[Dict]]
    :return: Model instance with loaded variable values
    :rtype: ConcreteModel

    Example:
        >>> data = {'x': [{'i': 1, 'j': 2, 'Value': 1.5}]}
        >>> model = instance_from_dict(model, data)
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
    Load values into a specific variable in a Pyomo model instance.

    Assigns values to a named variable in the model instance using
    the provided value dictionary with index-value pairs.

    :param instance: Pyomo model instance containing the variable
    :type instance: ConcreteModel
    :param var_name: Name of the variable to load values into
    :type var_name: str
    :param value: Dictionary mapping variable indices to values
    :type value: Dict[tuple, float]
    :return: None
    :raises KeyError: If variable name is not found in the instance

    Example:
        >>> values = {(1, 2): 1.5, (2, 3): 2.0}
        >>> load_variable(model, 'x', values)
    """
    var = instance.__dict__[var_name]
    for k, v in value.items():
        var[k] = v


def model_data_to_excel(model_data, path, file_name="model_data.xlsx"):
    """
    Export Pyomo model data to an Excel file.

    Saves model data (parameters, sets, etc.) to an Excel file with
    each data component as a separate worksheet. Handles both indexed
    and scalar data appropriately.

    :param model_data: Dictionary containing model data for Pyomo model
    :type model_data: Dict[str, Any]
    :param path: Directory path where the file should be saved
    :type path: str
    :param file_name: Name of the Excel file to create
    :type file_name: str
    :return: None
    :raises IOError: If file cannot be written

    Example:
        >>> data = {'demand': {(1,): 100, (2,): 200}, 'capacity': 1000}
        >>> model_data_to_excel(data, 'output/', 'model_data.xlsx')
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
    Export Pyomo solver result to a JSON file.

    Saves a Pyomo solver result object to a JSON file, converting
    Pyomo-specific objects to JSON-serializable formats. Handles
    special Pyomo data types and status objects.

    :param result: Pyomo solver result object
    :type result: SolverResults
    :param path: File path where the JSON should be saved
    :type path: str
    :return: None
    :raises IOError: If file cannot be written

    Example:
        >>> result = solver.solve(model)
        >>> solver_result_to_json(result, 'output/solution_result.json')
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
    Load a Pyomo solver result object from a JSON file.

    Reconstructs a Pyomo SolverResults object from data previously
    saved using solver_result_to_json. Restores the original structure
    and data types where possible.

    :param path: File path to the JSON file containing result data
    :type path: str
    :param kwargs: Additional keyword arguments for load_json
    :return: Reconstructed Pyomo solver result object
    :rtype: SolverResults
    :raises IOError: If file cannot be read

    Example:
        >>> result = solver_result_from_json('output/solution_result.json')
        >>> print(f"Solver status: {result.solver.termination_condition}")
    """
    data = load_json(path, **kwargs)
    result = SolverResults()
    for k, v in result.items():
        for k1, v1 in v.items():
            if data[k][0][k1] is not None:
                v1.value = data[k][0][k1]
    return result
