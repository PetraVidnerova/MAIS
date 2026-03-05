"""Configuration file utilities for the MAIS simulation.

This module provides classes and helpers for reading, writing, and generating
INI-style configuration files used to parameterise simulation runs.

Key components:

- :func:`string_to_value`: Type-coercing string parser used when reading
  INI values.
- :class:`ConfigFile`: Thin wrapper around :class:`configparser.ConfigParser`
  for loading, saving, and querying individual INI files.
- :class:`ConfigFileGenerator`: Expands a template INI file containing
  semicolon-separated parameter lists into a stream of fully-specified
  :class:`ConfigFile` instances, one per parameter combination.
"""

import configparser
import os
import io

from sklearn.model_selection import ParameterGrid

def string_to_value(s):
    """Convert a raw INI string value to the most appropriate Python type.

    Tries type conversions in the following order:

    1. ``int`` – if the entire string represents an integer.
    2. ``float`` – if the entire string represents a floating-point number.
    3. ``list`` – if the string contains a comma; it is split on commas and
       each token is stripped of surrounding whitespace.
    4. ``str`` – the original string is returned unchanged.

    Args:
        s (str): Raw string value read from a configuration file.

    Returns:
        int or float or list of str or str: The converted value.
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            pass

    if "," in s:
        list_of_values = s.split(",")
        return [val.strip() for val in list_of_values]
    else:
        return s



class ConfigFile():

    """Wrapper around :class:`configparser.ConfigParser` for INI-style config files.

    Provides convenience methods for loading from and saving to ``.ini``
    files, serialising to a string, and reading sections as type-converted
    dictionaries. Key-case is preserved (``optionxform = str``).

    Args:
        param_dict (dict, optional): If provided, a mapping of
            ``{section_name: {key: value}}`` pairs used to pre-populate
            the underlying :class:`configparser.ConfigParser`. Defaults to
            ``None`` (empty configuration).
    """

    def __init__(self, param_dict=None):

        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        if param_dict:
            for name, value in param_dict.items():
                self.config[name] = value

    def save(self, filename):
        """Write the configuration to a file or file-like object.

        Args:
            filename (str or file-like): If a string, the configuration is
                written to that file path (UTF-8 encoded). Otherwise the
                object is treated as a writable file-like and
                ``config.write()`` is called on it directly.
        """
        if type(filename) == str:
            with open(filename, 'w', encoding="utf-8") as configfile:
                self.config.write(configfile)
        else:
            self.config.write(filename)

    def to_string(self):
        """Serialise the configuration to an INI-formatted string.

        Returns:
            str: The full INI representation of the configuration, equivalent
            to what would be written by :meth:`save`.
        """
        output = io.StringIO()
        self.config.write(output)
        ret = output.getvalue()
        output.close()
        return ret

    def load(self, filename):
        """Read a configuration from an INI file.

        Args:
            filename (str): Path to the ``.ini`` file to read.

        Raises:
            ValueError: If ``filename`` does not exist on the filesystem.
        """
        if not os.path.exists(filename):
            raise ValueError(f"Config file {filename} not exists. Provide name (including path) to a valid config file.")
        self.config.read(filename)

    def section_as_dict(self, section_name):
        """Return the contents of a section as a type-converted dictionary.

        Each raw string value in the section is converted by
        :func:`string_to_value` to the most appropriate Python type.

        Args:
            section_name (str): Name of the INI section to retrieve.

        Returns:
            dict: Mapping of ``{key (str): value}`` for all entries in the
            section, with values converted by :func:`string_to_value`. Returns
            an empty dict if the section does not exist.
        """
        sdict = self.config._sections.get(section_name, {})
        return {name: string_to_value(value) for name, value in sdict.items()}

    def fix_output_id(self):
        """Resolve and replace the ``OUTPUT_ID.id`` field with a descriptive string.

        Reads the ``id`` entry from the ``[OUTPUT_ID]`` section. If it
        contains one or more ``section:key`` references (as a list or a
        single string), each reference is resolved to its current value in
        the configuration, and a composite identifier string of the form
        ``_Section_key=value`` is constructed and stored back into
        ``OUTPUT_ID.id``. Spaces in the resulting string are replaced with
        underscores.

        If ``OUTPUT_ID.id`` is not present, the method returns without making
        any changes.
        """
        output_id = self.section_as_dict("OUTPUT_ID").get("id", None)
        if output_id is None:
            return
        text_id = ""
        if not isinstance(output_id, list):
            output_id = [output_id]
        for variable in output_id:
            section, name = variable.split(":")
            text_id += f"_{section}_{name}={self.section_as_dict(section).get(name, None)}"
        text_id = text_id.replace(" ", "_")
        self.config["OUTPUT_ID"]["id"] = text_id



class ConfigFileGenerator():
    """Generator that expands a template INI file into individual :class:`ConfigFile` instances.

    The template INI file may contain semicolon-separated lists of values for
    any key. The generator computes the Cartesian product of all such lists
    (using :class:`sklearn.model_selection.ParameterGrid`) and yields one
    fully-specified :class:`ConfigFile` per parameter combination.

    After each config file is generated, :meth:`ConfigFile.fix_output_id` is
    called to resolve any ``OUTPUT_ID`` references.
    """

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str # this is to keep the case of the keys in the config file

    def _explode_lists(self, section):
        """Split each value in a config section on the ``';'`` separator.

        Args:
            section (dict): A dictionary of ``{key: raw_string_value}`` pairs
                from a single INI section.

        Returns:
            dict: Mapping of ``{key: list of str}`` where each raw value has
            been split into a list of alternative values.
        """
        return {
            name : value.split(";")
            for name, value in section.items()
        }

    def load(self, filename):
        """Load a template INI file and yield one :class:`ConfigFile` per parameter combination.

        Each key that contains a ``';'``-separated list of values is treated
        as a parameter with multiple options. The full Cartesian product of
        all such options (across all keys and sections) is enumerated, and
        each combination is yielded as a separate :class:`ConfigFile`.

        Args:
            filename (str): Path to the template ``.ini`` file. The file may
                contain semicolon-separated lists of values for any key.

        Yields:
            ConfigFile: A fully-specified configuration for one parameter
            combination, with ``OUTPUT_ID`` resolved via
            :meth:`ConfigFile.fix_output_id`.

        Raises:
            ValueError: If ``filename`` does not exist on the filesystem.
        """
        if not os.path.exists(filename):
            raise ValueError(f"Config file {filename} not exists. Provide name (including path) to a valid config file.")
        self.config.read(filename)


        variable_names = {
            section: list(self.config._sections[section].keys())
            for section in self.config.sections()
        }   

        # convert to dict 
        param_dict = {
            section: self._explode_lists(self.config._sections[section])
            for section in self.config.sections()
        }
        # convert each section to the list of final sections
        param_dict = {
            name: list(ParameterGrid(section))
            for name, section in param_dict.items()
        }
        # do the outer parameter grid
        param_dict = ParameterGrid(param_dict)
        for params in param_dict:
            x = ConfigFile(param_dict=params)
            x.fix_output_id()
            yield x


if __name__ == "__main__":

    test_generator = ConfigFileGenerator()
    for config in test_generator.load("../../config/info_verona.ini"):
        print(config.section_as_dict("OUTPUT_ID").get("id", None))
        print(config.to_string())


    exit()
    test_dict = {
        "TASK": {"num_nodes": 10000},
        "MODEL": {"beta": 0.155,
                  "gamma": 1/12.39,
                  "sigma": 1/5.2
                  }
    }

    test_config = ConfigFile(test_dict)
    test_config.save("test.ini")

    new_config = ConfigFile()
    new_config.load("test.ini")

    print(new_config.section_as_dict("TASK"))
    print(new_config.section_as_dict("MODEL"))
