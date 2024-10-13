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

    def get_all_parameters(self):
        """
        Retrieve a copy of all parameters.
        
        :return: A copy of the parameters dictionary.
        """
        return self.parameters.copy()

    def __init__(self):
        # Initialize parameters dictionary with default values
        self.parameters = {
            'High-Pass Filter Radius': 1.0,
            'Gaussian Blur (%)': 0.0,
            'Gamma Correction': 1.0,
            'Enable Frequency Peak Suppression': False,
            'Exclude Radius (%)': 10.0,
            'Aspect Ratio': 1.0,
            'Orientation': 0.0,
            'Exclude Falloff (%)': 10.0,
            'Peak Min Distance': 10,
            'Peak Threshold': 0.5,
            'Mask Radius (%)': 5.0,
            'Peak Mask Falloff (%)': 10.0,
            'Enable Attenuation': False,
            'Enable Anti-Aliasing Filter': False,
            'Anti-Aliasing Intensity (%)': 50.0,
            # Add other parameters as needed
        }

    def set_parameter(self, param_name, param_value):
        """Set the value of a parameter."""
        self.parameters[param_name] = param_value

    def get_parameters(self):
        """Return a copy of current parameters."""
        return self.parameters.copy()