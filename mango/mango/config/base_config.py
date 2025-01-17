"""
This file contains the classes that are used to parse the configuration files
"""

import json
import os
import warnings
from configparser import ConfigParser, NoOptionError, NoSectionError
from typing import Union, List, Dict


class ConfigParameter:
    """
    This class represents a parameter in a configuration file with INI File syntax.
    Each parameter has a name (their section gets parsed from the file), value type, default value, list of valid values
    and a secondary type (if the value type is a list).
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
        The __init__ function is called when a new instance of the class is created.
        It sets up the object with attributes that were defined in the class definition.

        :param str name: set the name of the parameter
        :param callable value_type: specify the type of data that is allowed
        :param default: set the default value of the parameter
        :param validate: specify a list of values the parameter can take
        :param callable secondary_type: if the value_type is a list, specify the type of data that is allowed in the list
        :param dict_types: specify the type of data that is allowed in the dictionary
        :param bool required: specify if the parameter is required or not
        :param min_value: specify the minimum value of the parameter (only works if value_type is int or float)
        :param max_value: specify the maximum value of the parameter (only works if value_type is int or float)
        :doc-author: baobab soluciones
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
        The parse function takes a section name and a ConfigParser object.
        It returns the value of the parameter with name `name` from that section.

        :param str section: specify the section in the config file
        :param config_parser: the config parser class
        :return: the value of the config parameter
        :rtype: Union[int, float, bool, str, list]
        :doc-author: baobab soluciones
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
        The __repr__ function is what gets called when you try to &quot;print&quot; an object.

        :return: A string representation of the object
        :doc-author: baobab soluciones
        """
        if self.default is None:
            return f"ConfigParameter({self.name!r}, {self.value_type!r})"
        return f"ConfigParameter({self.name!r}, {self.value_type!r}, {self.default!r})"


class BaseConfig:
    """
    This class is used to handle configuration files with INI file structure.
    Parameters can be defined, their section gets parsed from the configuration file
    but defined in the params dictionary, and for each parameter a value,
    type, default value, validation and secondary type can be defined.
    """

    __params = dict()

    def __init__(self, file_name: str, extend_params: Dict):
        """
        The __init__ function is called when an instance of the class is created.
        It handles the initial parsing of the config file

        :param file_name: specify the name of the config file
        :param extend_params: extend the parameters that are defined in the config file
        :doc-author: baobab soluciones
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
        The __call__ function allows the class to be called as a function.

        :param key: the name of the parameter we want to extract
        :return: The value of the parameter
        :doc-author: baobab soluciones
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
        The modify value allows to modify the value of a parameter after it has been loaded

        :param key: the name of the parameter we want to extract
        :return: The value of the parameter
        :doc-author: baobab soluciones
        """
        section = self.map_key_to_section.get(key, None)
        if section is None:
            return KeyError(f"Parameter {key} not found")
        self.parameters.get(section)[key] = value

    @classmethod
    def create_config_template(cls, output_path: str):
        """
        The create_config_template function creates a configuration file with default values for each parameter.
        If the parameter has a default value, it will be used, otherwise an empty string will be used.

        :param output_file: specify the output path for the template of the config file
        :doc-author: baobab soluciones
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
