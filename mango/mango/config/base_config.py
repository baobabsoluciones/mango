"""
Configuration file parsing classes.

This module provides classes for parsing and managing configuration files
with INI file structure, including parameter validation and type conversion.
"""

import json
import os
import warnings
from configparser import ConfigParser, NoOptionError, NoSectionError
from typing import Union, List, Dict


class ConfigParameter:
    """
    Represents a configuration parameter with validation and type conversion.

    This class defines a parameter in a configuration file with INI syntax.
    Each parameter has a name, value type, default value, validation rules,
    and optional secondary type for complex data structures.

    :param name: Name of the configuration parameter
    :type name: str
    :param value_type: Type function for the parameter value (int, float, bool, str, list, dict)
    :type value_type: callable
    :param default: Default value if parameter is not found or invalid
    :type default: Union[int, float, bool, str, List], optional
    :param validate: List of valid values for the parameter
    :type validate: List, optional
    :param secondary_type: Type function for list elements (used when value_type is list)
    :type secondary_type: callable, optional
    :param dict_types: Dictionary mapping keys to type functions (used when value_type is dict)
    :type dict_types: Dict[str, callable], optional
    :param required: Whether the parameter is required
    :type required: bool
    :param min_value: Minimum allowed value for numeric parameters
    :type min_value: Union[int, float], optional
    :param max_value: Maximum allowed value for numeric parameters
    :type max_value: Union[int, float], optional

    Example:
        >>> param = ConfigParameter(
        ...     name="timeout",
        ...     value_type=int,
        ...     default=30,
        ...     min_value=1,
        ...     max_value=300
        ... )

    """

    def __init__(
        self,
        name: str,
        value_type: callable,
        default: Union[int, float, bool, str, List] = None,
        validate: List = None,
        secondary_type: callable = None,
        dict_types: Dict[str, callable] = None,
        required: bool = True,
        min_value: Union[int, float] = None,
        max_value: Union[int, float] = None,
    ):
        """
        Initialize a configuration parameter with validation rules.

        Sets up the parameter with its name, type, default value, and validation
        constraints. The parameter will be used to parse and validate configuration
        values from INI files.
        """
        self.name = name
        self.value_type = value_type
        self.secondary_type = secondary_type
        self.dict_types = dict_types
        self.default = default
        self.validate = validate
        self.required = required
        self.min_value = min_value
        self.max_value = max_value

    def parse(
        self, section: str, config_parser: ConfigParser
    ) -> Union[int, float, bool, str, List]:
        """
        Parse and validate a parameter value from a configuration section.

        Extracts the parameter value from the specified section using the
        ConfigParser, converts it to the appropriate type, and validates
        it against the defined constraints.

        :param section: Section name in the configuration file
        :type section: str
        :param config_parser: ConfigParser instance containing the configuration data
        :type config_parser: ConfigParser
        :return: Parsed and validated parameter value
        :rtype: Union[int, float, bool, str, List]
        :raises ValueError: If value is outside min/max range or not in validation list
        :raises TypeError: If value type is not supported

        Example:
            >>> parser = ConfigParser()
            >>> parser.read_string('[database]\nport = 5432')
            >>> param = ConfigParameter("port", int, default=3306)
            >>> value = param.parse("database", parser)
            >>> print(value)  # 5432

        """
        if int == self.value_type:
            value = config_parser.getint(section, self.name)
            if self.min_value is not None and value < self.min_value:
                raise ValueError(
                    f"The value for config parameter {self.name} is not valid. "
                    f"It must be greater than {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ValueError(
                    f"The value for config parameter {self.name} is not valid. "
                    f"It must be less than {self.max_value}"
                )
        elif float == self.value_type:
            value = config_parser.getfloat(section, self.name)
            if self.min_value is not None and value < self.min_value:
                raise ValueError(
                    f"The value for config parameter {self.name} is not valid. "
                    f"It must be greater than {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ValueError(
                    f"The value for config parameter {self.name} is not valid. "
                    f"It must be less than {self.max_value}"
                )
        elif bool == self.value_type:
            value = config_parser.getboolean(section, self.name)
        elif str == self.value_type:
            value = config_parser.get(section, self.name)
        elif list == self.value_type:
            value = config_parser.get(section, self.name).split(" ")
            value = [self.secondary_type(x) for x in value]
        elif dict == self.value_type:
            value = config_parser.get(section, self.name)
            value = json.loads(value)
            if self.dict_types is not None:
                value = {k: self.dict_types.get(k, str)(v) for k, v in value.items()}
        else:
            raise TypeError(
                f"The parser can not parse the datatype indicated for {self.name}"
            )

        if self.validate is not None:
            if value not in self.validate:
                raise ValueError(
                    f"The value for config parameter {self.name} is not valid. "
                    f"It must be one of the following: {self.validate}"
                )
        return value

    def __repr__(self):
        """
        Return string representation of the ConfigParameter.

        Creates a string representation that can be used to recreate
        the parameter object, showing the name, type, and default value.

        :return: String representation of the parameter
        :rtype: str

        Example:
            >>> param = ConfigParameter("timeout", int, default=30)
            >>> print(repr(param))
            ConfigParameter('timeout', <class 'int'>, 30)
        """
        if self.default is None:
            return f"ConfigParameter({self.name!r}, {self.value_type!r})"
        return f"ConfigParameter({self.name!r}, {self.value_type!r}, {self.default!r})"


class BaseConfig:
    """
    Base class for handling configuration files with INI structure.

    This class provides a framework for parsing and managing configuration
    files with INI syntax. Parameters are defined in the class and can
    include type conversion, validation, and default values.

    :param file_name: Path to the configuration file
    :type file_name: str
    :param extend_params: Additional parameters to extend the base configuration
    :type extend_params: Dict

    Example:
        >>> class MyConfig(BaseConfig):
        ...     __params = {
        ...         "database": [
        ...             ConfigParameter("host", str, default="localhost"),
        ...             ConfigParameter("port", int, default=5432)
        ...         ]
        ...     }
        >>> config = MyConfig("config.ini", {})
        >>> host = config("host")
    """

    __params = dict()

    def __init__(self, file_name: str, extend_params: Dict):
        """
        Initialize configuration by parsing the specified file.

        Loads and parses the configuration file, merging base parameters
        with extended parameters. Validates all parameters and applies
        default values where necessary.

        :param file_name: Path to the configuration file to load
        :type file_name: str
        :param extend_params: Dictionary of additional parameters to merge
        :type extend_params: Dict
        :raises FileNotFoundError: If the configuration file does not exist
        :raises ValueError: If parameter validation fails
        :raises TypeError: If parameter type conversion fails
        """
        self.params = dict(self.__params)
        for section, params in extend_params.items():
            if self.params.get(section) is not None:
                self.params[section] = self.params[section] + params
            else:
                self.params[section] = params
        self.file_name = file_name
        self.parameters = {}
        self.map_key_to_section = {}
        parser = ConfigParser()
        # Case sensitive
        parser.optionxform = lambda option: option
        if not os.path.isfile(self.file_name):
            raise FileNotFoundError(f"No such config file: {self.file_name}")

        with open(self.file_name) as f:
            parser.read_file(f)

        for section, params in self.params.items():
            self.parameters[section] = {}
            for p in params:
                if p.default is None:
                    self.parameters[section][p.name] = p.parse(section, parser)
                    self.map_key_to_section[p.name] = section
                else:
                    try:
                        self.parameters[section][p.name] = p.parse(section, parser)
                        self.map_key_to_section[p.name] = section
                    except (ValueError, NoOptionError, NoSectionError) as excep:
                        if p.required:
                            self.parameters[section][p.name] = p.default
                            self.map_key_to_section[p.name] = section
                            if isinstance(excep, ValueError):
                                warnings.warn(
                                    f"Config {self.file_name}. Section {section}, parameter {p.name} "
                                    f"has an incorrect format, using default value: {p.default}"
                                )
                            elif isinstance(excep, NoOptionError):
                                warnings.warn(
                                    f"Config {self.file_name}. Section {section}, parameter {p.name} is not set, "
                                    f"using default value: {p.default}"
                                )
                            elif isinstance(excep, NoSectionError):
                                warnings.warn(
                                    f"Config {self.file_name}. Section {section} does not appear on file, "
                                    f"setting parameter {p.name} to default value: {p.default}"
                                )

    def __call__(self, key, default=None):
        """
        Retrieve a configuration parameter value by key.

        Allows the configuration object to be called as a function to
        retrieve parameter values. Returns the default value if the
        parameter is not found.

        :param key: Name of the parameter to retrieve
        :type key: str
        :param default: Default value to return if parameter is not found
        :type default: Any, optional
        :return: Parameter value or default value
        :rtype: Any

        Example:
            >>> config = MyConfig("config.ini", {})
            >>> host = config("database_host", "localhost")
            >>> port = config("database_port")
        """
        section = self.map_key_to_section.get(key, None)
        if section is None:
            return default
        value = self.parameters.get(section).get(key, None)
        if value is None:
            return default
        return value

    def modify_value(self, key, value):
        """
        Modify the value of a configuration parameter.

        Updates the value of an existing parameter after the configuration
        has been loaded. The parameter must exist in the configuration.

        :param key: Name of the parameter to modify
        :type key: str
        :param value: New value for the parameter
        :type value: Any
        :return: None
        :raises KeyError: If the parameter is not found in the configuration

        Example:
            >>> config = MyConfig("config.ini", {})
            >>> config.modify_value("database_port", 3306)
        """
        section = self.map_key_to_section.get(key, None)
        if section is None:
            return KeyError(f"Parameter {key} not found")
        self.parameters.get(section)[key] = value

    @classmethod
    def create_config_template(cls, output_path: str):
        """
        Create a configuration template file with default values.

        Generates a configuration file template based on the defined
        parameters. Parameters with default values will use those values,
        while parameters without defaults will be set to empty strings.

        :param output_path: Path where the template file should be created
        :type output_path: str
        :return: None
        :raises IOError: If the file cannot be written

        Example:
            >>> MyConfig.create_config_template("config_template.ini")
        """
        parser = ConfigParser()
        # Case sensitive
        parser.optionxform = lambda option: option
        # Available params are attributes in the form _ChildClass__params
        params = cls.__getattribute__(cls, "_{}__params".format(cls.__name__))
        for section, params in params.items():
            parser.add_section(section)
            for p in params:
                if p.default is None:
                    parser.set(section, p.name, value="")
                else:
                    parser.set(section, p.name, value=str(p.default))
        with open(output_path, "w") as f:
            parser.write(f)
