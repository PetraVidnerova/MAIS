import configparser
import os
import io


def string_to_value(s):
    """
    If string is convertable to int, returns int,
    if it is convertable to float, returns float,
    if it is comma separated, returns list of strings,
    otherwise returns original string.
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

    """
    Class encapsulating the ConfigParser. 
    Deals with INI files.
    """

    def __init__(self, param_dict=None):

        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        if param_dict:
            for name, value in param_dict.items():
                self.config[name] = value

    def save(self, filename):
        if type(filename) == str:
            with open(filename, 'w', encoding="utf-8") as configfile:
                self.config.write(configfile)
        else:
            self.config.write(filename)

    def to_string(self):
        output = io.StringIO()
        self.config.write(output)
        ret = output.getvalue()
        output.close()
        return ret

    def load(self, filename):
        if not os.path.exists(filename):
            raise ValueError(f"Config file {filename} not exists. Provide name (including path) to a valid config file.")
        self.config.read(filename)

    def section_as_dict(self, section_name):
        sdict = self.config._sections.get(section_name, {})
        return {name: string_to_value(value) for name, value in sdict.items()}


if __name__ == "__main__":

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
