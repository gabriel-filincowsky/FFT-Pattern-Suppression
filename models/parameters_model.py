from utils.config_manager import load_config, save_config, reset_to_defaults

class ParametersModel:
    """
    Manages processing parameters, handling loading, saving, and resetting configurations.
    """
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.parameters = load_config(self.config_path)

    def get_parameter(self, key: str):
        """
        Retrieve the value of a specific parameter.
        
        :param key: The parameter name.
        :return: The parameter value.
        """
        return self.parameters.get(key, None)

    def set_parameter(self, key: str, value):
        """
        Set the value of a specific parameter.
        
        :param key: The parameter name.
        :param value: The new value for the parameter.
        """
        self.parameters[key] = value

    def load_parameters(self):
        """
        Load parameters from the configuration file.
        """
        self.parameters = load_config(self.config_path)

    def save_parameters(self):
        """
        Save current parameters to the configuration file.
        """
        save_config(self.parameters, self.config_path)

    def reset_parameters(self):
        """
        Reset parameters to default values.
        """
        self.parameters = reset_to_defaults()
        self.save_parameters()