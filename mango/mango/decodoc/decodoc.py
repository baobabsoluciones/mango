import atexit
import functools
import inspect
import json
import os
import traceback
from itertools import zip_longest

from mango.logging import get_configured_logger

log = get_configured_logger(__name__)

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
    Decorator to automatically log function execution metadata.

    A decorator that captures and stores comprehensive metadata about function
    execution, including inputs, outputs, caller information, docstrings, and
    source code. This metadata is stored in a global dictionary for later use
    in debugging, documentation generation, and creating visual representations
    of function call relationships.

    The decorator supports both single and multiple return values and handles
    cases where inputs or outputs are missing or undefined. It's particularly
    useful for creating detailed logs in complex applications and generating
    Mermaid diagrams showing function call relationships.

    :param inputs: List of parameter names corresponding to function arguments
    :type inputs: List[str]
    :param outputs: List of return value names corresponding to function outputs
    :type outputs: List[str]
    :return: Decorated function that stores execution metadata
    :rtype: callable
    :raises UserWarning: If function lacks docstring or has missing parameter names

    Example:
        >>> from mango.decodoc import decodoc
        >>>
        >>> @decodoc(inputs=['a', 'b'], outputs=['result'])
        >>> def add(a: int, b: int) -> int:
        >>>     return a + b
        >>>
        >>> result = add(5, 3)
        >>> # Metadata stored in INFO_DICT with function name, caller, inputs, outputs
    """
    # Create a closure to maintain state
    at_exit_registered = False

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal at_exit_registered
            if not at_exit_registered:
                atexit.register(_write)
                log.info("Registered at exit")
                at_exit_registered = True

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
                log.warning(f"Function {func.__name__} does not have a docstring")

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
                    log.warning(f"Missing variable argument name for {func.__name__}")
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
                    log.warning(f"Missing return variable name for {func.__name__}")
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
    Update the default configuration settings for decodoc.

    Modifies the global configuration dictionary with new settings. Only the
    provided settings will be updated; existing settings not specified will
    remain unchanged.

    :param setup_dict_arg: Dictionary containing configuration parameters to update
    :type setup_dict_arg: dict
    :return: None

    Configuration Parameters:
        - base_directory: Base directory for output files (default: "decodoc")
        - json_directory: Subdirectory for JSON files (default: "json")
        - mermaid_directory: Subdirectory for Mermaid files (default: "mermaid")
        - prompts_directory: Subdirectory for prompt files (default: "prompts")
        - json: Enable/disable JSON file generation (default: True)
        - mermaid: Enable/disable Mermaid file generation (default: True)
        - prompts: Enable/disable prompt file generation (default: True)
        - confidential: Hide variable values in prompts (default: False)
        - autowrite: Automatically write files on exit (default: True)

    Example:
        >>> decodoc_setup({
        ...     "json": False,
        ...     "confidential": True,
        ...     "base_directory": "my_output"
        ... })
    """
    SETUP_DICT.update(setup_dict_arg)


def get_dict():
    """
    Retrieve the global information dictionary used by decodoc.

    Returns the global dictionary containing all function execution metadata
    collected by the decodoc decorator. This includes function names, callers,
    inputs, outputs, docstrings, and source code.

    :return: Dictionary containing function execution metadata
    :rtype: dict

    Example:
        >>> from mango.decodoc import get_dict
        >>>
        >>> info_dict = get_dict()
        >>> print(f"Captured {len(info_dict)} function executions")
    """
    return INFO_DICT


def _generate_id(string):
    """
    Generate a unique identifier from a string using hash function.

    Creates a unique identifier by computing the absolute hash value of the
    input string. This is used for creating unique node identifiers in
    Mermaid diagrams.

    :param string: Input string to generate ID from
    :type string: str
    :return: Unique identifier as string
    :rtype: str

    Example:
        >>> _generate_id("my_function")
        '1234567890'
    """
    return str(abs(hash(string)))


def _write_json(info_dict, path):
    """
    Write function execution metadata to a JSON file.

    Serializes the information dictionary to JSON format and writes it to
    the specified file path. Only executes if JSON output is enabled in
    the configuration.

    :param info_dict: Dictionary containing function execution metadata
    :type info_dict: dict
    :param path: File path where JSON data should be written
    :type path: str
    :return: None
    :raises IOError: If file cannot be written
    """
    if not SETUP_DICT["json"]:
        return None

    with open(path, "w") as f:
        json.dump(info_dict, f, indent=4)


def _write_mermaid(info_dict, path):
    """
    Generate and write a Mermaid diagram from function execution metadata.

    Creates a Mermaid graph diagram showing function call relationships,
    input/output connections, and data flow. The diagram is written to
    the specified file path in Markdown format.

    :param info_dict: Dictionary containing function execution metadata
    :type info_dict: dict
    :param path: File path where Mermaid diagram should be written
    :type path: str
    :return: None
    :raises IOError: If file cannot be written
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
    Generate and write a documentation prompt for a caller function.

    Creates a structured prompt for documenting a specific caller function
    and all functions it calls. The prompt includes function metadata,
    docstrings, inputs, outputs, and source code to facilitate automated
    documentation generation.

    :param raw_info_dict: Dictionary containing all function execution metadata
    :type raw_info_dict: dict
    :param path: File path where the prompt should be written
    :type path: str
    :param caller: Name of the caller function to document
    :type caller: str
    :return: None
    :raises IOError: If file cannot be written
    """
    if not SETUP_DICT["prompts"]:
        log.info("Prompts are disabled")
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
    Write all decodoc output files based on current configuration.

    Generates and writes all configured output files including JSON metadata,
    Mermaid diagrams, and documentation prompts. Creates the necessary
    directory structure and processes the global information dictionary
    to produce various output formats.

    :return: None
    :raises IOError: If any output file cannot be written
    """

    def create_folder_structure(folder_structure):
        """
        Create directory structure if it does not exist.

        Creates all directories in the provided list, including any
        necessary parent directories. Uses exist_ok=True to avoid
        errors if directories already exist.

        :param folder_structure: List of directory paths to create
        :type folder_structure: List[str]
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

    log.info(f"decodoc process completed. Check './{base_dir}' for details.")
