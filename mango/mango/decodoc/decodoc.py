import atexit
import functools
import inspect
import json
import os
import traceback
import warnings
from itertools import zip_longest

INFO_DICT = {}
SETUP_DICT = {
    "base_directory": "decodoc",
    "json_directory": "json",
    "mermaid_directory": "mermaid",
    "prompts_directory": "prompts",
    "json": True,
    "mermaid": True,
    "prompts": True,
    "confidential": False,
    "autowrite": True,
}


def decodoc(inputs, outputs):
    """
    Decorator to log information about the function

    A decorator that automatically generates and stores metadata about a function's execution, including the function's
    inputs and outputs. This information is stored in a global dictionary `INFO_DICT`, which can later be used for
    debugging, documentation generation, and tracking the state of function calls.

    The decorator not only logs the arguments and return values of a function but also includes valuable metadata, such
    as the function's name, caller, docstring, and the source code. It supports both single and multiple return values
    and handles cases where inputs or outputs are missing or undefined.

    This decorator is particularly useful for creating detailed logs of function calls in complex applications,
    generating documentation, and creating visual representations of function call relationships using Mermaid diagrams.

    :param inputs: A list of names representing the function's input parameters. These should correspond to the actual
                   arguments passed to the function when it is called, and the order is important.
    :param outputs: A list of names representing the function's return values. These should match the names of the
                    variables assigned to the function's return values, and the order is important.

    :return: The decorated function that will store its execution metadata in the global `INFO_DICT`.

    Example Usage
    -------------
    Here's an example that demonstrates how the `decodoc` decorator works:

    >>> from mango.decodoc import decodoc
    >>>
    >>> @decodoc(inputs=['a', 'b'], outputs=['result'])
    >>> def add(a: int, b: int) -> int:
    >>>     return a + b
    >>>
    >>> result = add(5, 3)

    This example will store the following metadata in `INFO_DICT`:

    - Function name: 'add'

    - Caller function: The function that called 'add'

    - Input values: The values and memory addresses of 'a' and 'b'

    - Output values: The value and memory address of 'result'

    Warnings
    --------
    - If a function does not have a docstring, a warning will be issued.
    - If any input or output variable names are missing or invalid, a warning will be generated.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the function ID and the arguments ID to obtain an unique ID
            args_id = [id(arg) for arg in args]
            args_id = "_".join(map(str, args_id))
            function_id = str(id(func)) + "_" + args_id
            INFO_DICT[function_id] = {}

            # Get information about the function
            INFO_DICT[function_id]["name"] = func.__name__
            INFO_DICT[function_id]["caller"] = traceback.extract_stack()[-2].name
            docstring = func.__doc__

            if docstring is None:
                docstring = ""
                warnings.warn(
                    f"Function {func.__name__} does not have a docstring", UserWarning
                )

            INFO_DICT[function_id]["docstring"] = docstring.replace("\n", " ")
            INFO_DICT[function_id]["code"] = inspect.getsource(func).strip()

            # Get the input DataFrames
            inputs_dict = {}
            if not isinstance(args, tuple):
                args_list = [args]
            else:
                args_list = list(args)
            for input_name, input_var in zip_longest(inputs, args_list):
                if input_name is None:
                    warnings.warn(
                        f"Missing variable argument name for {func.__name__}",
                        UserWarning,
                    )
                if input_name == 0:
                    continue
                if input_var is None:
                    inputs_dict[input_name] = "None"
                else:
                    inputs_dict[input_name] = {
                        "memory": id(input_var),
                        "value": str(input_var),
                    }
            INFO_DICT[function_id]["input"] = inputs_dict

            # Execute the function
            result = func(*args, **kwargs)

            # Get the output DataFrames
            outputs_dict = {}
            if not isinstance(result, tuple):
                result_list = [result]
            else:
                result_list = list(result)
            for output_name, output_var in zip_longest(outputs, result_list):
                if output_name is None:
                    warnings.warn(
                        f"Missing return variable name for {func.__name__}", UserWarning
                    )
                if output_name == 0:
                    continue
                if output_var is None:
                    outputs_dict[output_name] = "None"
                else:
                    outputs_dict[output_name] = {
                        "memory": id(output_var),
                        "value": str(output_var),
                    }
            INFO_DICT[function_id]["output"] = outputs_dict

            return result

        return wrapper

    return decorator


def decodoc_setup(setup_dict_arg):
    """
    Update the default setting for decodoc.

    The default settings for decodoc are define in a global dictionary at the beginning of the module. This function
    allows to pass a new dictionary with the settings to update the default settings. This dictionary does not need to
    have all the settings, only the ones that need to be updated.

    :param setup_dict_arg: A dictionary with the parameters to update

    :return: None

    Default Settings
    ----------------

    >>> default_setting = {
            "base_directory": "decodoc",
            "json_directory": "json",
            "mermaid_directory": "mermaid",
            "prompts_directory": "prompts",
            "json": True,
            "mermaid": True,
            "prompts": True,
            "confidential": False,
            "autowrite": True
        }

    Example Usage
    -------------

    >>> decodoc_setup({
            "json": False,
            "confidential": True
        })

    Settings:
    ----------------------------

    - base_directory: The base directory where the files will be saved

    - json_directory: The subdirectory inside the base directory where the JSON files will be saved

    - mermaid_directory: The subdirectory inside the base directory where the Mermaid files will be saved

    - prompts_directory: The subdirectory inside the base directory where the prompts files will be saved

    - json: A boolean to enable or disable the JSON file generation

    - mermaid: A boolean to enable or disable the Mermaid file generation

    - prompts: A boolean to enable or disable the prompts file generation

    - confidential: A boolean to enable or disable the inclusion of the variable values in the generated prompts
    """
    SETUP_DICT.update(setup_dict_arg)


def get_dict():
    """
    Return the dictionary with the information used by decodoc

    :return: A dictionary

    Example Usage
    -------------
    >>> from mango.decodoc import get_dict
    >>>
    >>> info_dict = get_dict()
    """
    return INFO_DICT


def _generate_id(string):
    """
    Generate a unique ID from a string

    :param string: The string to hash

    :return: A string with the ID
    """
    return str(abs(hash(string)))


def _write_json(info_dict, path):
    """
    Write the dictionary to a JSON file

    :param info_dict: A dictionary with the information

    :param path: A string with the path to save the file

    :return: None
    """
    if not SETUP_DICT["json"]:
        return None

    with open(path, "w") as f:
        json.dump(info_dict, f, indent=4)


def _write_mermaid(info_dict, path):
    """
    Write the dictionary to a Mermaid diagram

    :param info_dict: A dictionary with the information

    :param path: A string with the path to save the file

    :return: None
    """
    if not SETUP_DICT["mermaid"]:
        return None

    string = """
```mermaid
graph TD

    """

    for func_id, func_info in info_dict.items():
        for name, var in func_info["input"].items():
            if var == "None":
                label_string = str(_generate_id(name)) + "[(" + str(name) + ")]"
            else:
                label_string = (
                    str(var["memory"])
                    + "@"
                    + '{ shape: braces, label: "'
                    + str(name)
                    + '"}'
                )
            string += f"{label_string} --> {func_id}[/{func_info['name']}/]\n"

        for name, var in func_info["output"].items():
            if var == "None":
                label_string = str(_generate_id(name)) + "[(" + str(name) + ")]"
            else:
                label_string = (
                    str(var["memory"])
                    + "@"
                    + '{shape: braces, label: "'
                    + str(name)
                    + '"}'
                )
            string += f"{func_id}[/{func_info['name']}/] --> {label_string}\n"

    string += "```"

    with open(path, "w") as f:
        f.write(string)


def _write_caller_prompt(raw_info_dict, path, caller):
    """
    Write the prompt for the caller function

    :param raw_info_dict: A dictionary with the information

    :param path: A string with the path to save the file

    :param caller: A string with the name of the caller function

    :return: None
    """
    if not SETUP_DICT["prompts"]:
        print("Prompts are disabled")
        return None

    prompt = f"""# Function Documentation for {caller}
Please redact the documentation for the {caller} function, this function calls some other functions, focus on the 
different relationships between the various executions of the functions that it calls. The response should be structured 
with clear sections and titles, with an overall overview of the whole process. The overall tone should be narrative, 
with long sentences that explain the flow and interconnectedness of the functions. Ensure that the output reflects the 
complexity of the process while being organized and easy to follow. The documentation should include:

- An overview of the {caller} function, summarizing its purpose and main stages.
- A clear explanation of the flow between the functions, detailing how data moves through the process from start to end.
- A conclusion that ties together the entire process, emphasizing the role each function plays in achieving the final result.

Here you have a brief description of the {caller} function and each function used inside, in the order of use during 
execution, the name of the function, the docstring of function, and the input and output ande the code. If a function 
appears more than once it means it has been used more than one time during run time:\n\n"""

    caller_func_info = None
    for func_id, func_info in raw_info_dict.items():
        if func_info["name"] == caller:
            caller_func_info = func_info
    if not caller_func_info:
        return None
    prompt += f"## {caller}\n\n"
    prompt += f"- Docstring: {caller_func_info['docstring']}\n"
    prompt += f"- Input:\n"
    for name, var in caller_func_info["input"].items():
        if var == "None" or SETUP_DICT["confidential"]:
            prompt += f"    - {name}\n"
        else:
            prompt += f"```python\n{name}\n{var['value']}\n```\n"
    prompt += f"- Output:\n"
    for name, var in caller_func_info["output"].items():
        if var == "None" or SETUP_DICT["confidential"]:
            prompt += f"    - {name}\n"
        else:
            prompt += f"```python\n{name}\n{var['value']}\n```\n"
    prompt += f"- Code: \n```python\n{caller_func_info['code']}\n```\n\n"

    info_dict = {
        key: value
        for key, value in raw_info_dict.items()
        if value.get("caller") == caller
    }

    # Iterate over the functions to get only the information needed
    for func_id, func_info in info_dict.items():
        prompt += f"### {func_info['name']}\n\n"
        prompt += f"- Docstring: {func_info['docstring']}\n"
        prompt += f"- Input:\n"
        for name, var in func_info["input"].items():
            if var == "None" or SETUP_DICT["confidential"]:
                prompt += f"    - {name}\n"
            else:
                prompt += f"```python\n{name}\n{var['value']}\n```\n"
        prompt += f"- Output:\n"
        for name, var in func_info["output"].items():
            if var == "None" or SETUP_DICT["confidential"]:
                prompt += f"    - {name}\n"
            else:
                prompt += f"```python\n{name}\n{var['value']}\n```\n"
        prompt += f"- Code: \n```python\n{func_info['code']}\n```\n\n"

    with open(path, "w") as f:
        f.write(prompt)


def _write():
    """
    Write the files with the information

    :return: None
    """

    def create_folder_structure(folder_structure):
        """
        Create a folder structure if it does not exist
        :param List folder_structure: List of folders to create
        :return: None
        """
        _ = [os.makedirs(path, exist_ok=True) for path in folder_structure]

    # Get the directory variables for writing the files
    base_dir = SETUP_DICT["base_directory"]
    json_dir = SETUP_DICT["json_directory"]
    mermaid_dir = SETUP_DICT["mermaid_directory"]
    prompts_dir = SETUP_DICT["prompts_directory"]
    create_folder_structure(
        [
            base_dir,
            f"{base_dir}/{json_dir}",
            f"{base_dir}/{mermaid_dir}",
            f"{base_dir}/{prompts_dir}",
        ]
    )

    # Write the mermaid and json with thr raw dict
    _write_json(INFO_DICT, f"{base_dir}/{json_dir}/raw.json")
    _write_mermaid(INFO_DICT, f"{base_dir}/{mermaid_dir}/raw.md")

    # Filter the dictionary to get only the unique callers
    unique_callers = {
        value["caller"] for value in INFO_DICT.values() if "caller" in value
    }

    # Write the mermaid with the filtered dict, which has the complete dict
    filtered_dict = {
        key: value
        for key, value in INFO_DICT.items()
        if value.get("caller") in unique_callers
        and value.get("name") not in unique_callers
    }
    _write_mermaid(filtered_dict, f"{base_dir}/{mermaid_dir}/complete.md")

    # Write the mermaid and prompt with the caller dict
    for caller in unique_callers:
        if caller == "<module>":
            caller_path = "module"
        else:
            caller_path = caller
        caller_dict = {
            key: value
            for key, value in INFO_DICT.items()
            if value.get("caller") == caller
        }
        _write_mermaid(caller_dict, f"{base_dir}/{mermaid_dir}/{caller_path}.md")
        _write_caller_prompt(
            INFO_DICT, f"{base_dir}/{prompts_dir}/{caller_path}.md", caller
        )

    print(f"decodoc process completed. Check './{base_dir}' for details.")


# Register the write function to be executed at the end of the script
atexit.register(_write)
