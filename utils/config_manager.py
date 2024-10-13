import json
import os
import logging
from utils.constants import (
    MAX_HIGH_PASS_RADIUS,
    MIN_HIGH_PASS_RADIUS,
    MAX_GAUSSIAN_BLUR_PCT,
    MIN_GAUSSIAN_BLUR_PCT,
    MAX_PEAK_THRESHOLD,
    MIN_PEAK_THRESHOLD,
)

DEFAULT_CONFIG_PATH = "config/default_parameters.json"

def load_default_config(path=DEFAULT_CONFIG_PATH):
    """
    Load default configuration from a JSON file.
    
    :param path: Path to the default configuration JSON file
    :return: Dictionary containing the default configuration
    """
    try:
        with open(path, 'r') as f:
            default_config = json.load(f)
        logging.getLogger(__name__).info(f"Default configuration loaded from {path}.")
        return default_config
    except FileNotFoundError:
        logging.getLogger(__name__).error(f"Default config file {path} not found.")
        return {}
    except json.JSONDecodeError as e:
        logging.getLogger(__name__).error(f"JSON decode error in {path}: {e}")
        return {}

DEFAULT_CONFIG = load_default_config()

# Define validation rules for each parameter
VALIDATION_RULES = {
    "High-Pass Filter Radius": {
        "type": float,
        "min": MIN_HIGH_PASS_RADIUS,
        "max": MAX_HIGH_PASS_RADIUS,
        "precision": 1
    },
    "Gaussian Blur (%)": {
        "type": float,
        "min": MIN_GAUSSIAN_BLUR_PCT,
        "max": MAX_GAUSSIAN_BLUR_PCT,
        "precision": 1
    },
    "Peak Threshold": {
        "type": float,
        "min": MIN_PEAK_THRESHOLD,
        "max": MAX_PEAK_THRESHOLD
    },
    # Add other parameters with their respective rules ...
}

def save_config(config: dict, config_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    :param config: Configuration dictionary to save
    :param config_path: Path to save the configuration file
    """
    logger = logging.getLogger(__name__)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}.")

class ConfigManager:
    """
    Manages the application's configuration, including loading, saving, and validating parameters.
    """

    def __init__(self, config_path="config/config.json"):
        """
        Initialize the ConfigManager.
        
        :param config_path: Path to the configuration JSON file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_config()  # Ensure load_config is called during initialization

    def load_config(self) -> dict:
        """
        Load configuration from the JSON file.
        
        :return: Loaded configuration dictionary
        """
        logger = self.logger
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file {self.config_path} not found. Using default configuration.")
            self.config = DEFAULT_CONFIG.copy()
            return self.config
        with open(self.config_path, 'r') as f:
            try:
                config = json.load(f)
                if self.validate_config(config):
                    logger.info(f"Configuration loaded successfully from {self.config_path}.")
                    self.config = config
                else:
                    logger.error("Configuration validation failed. Using default configuration.")
                    self.config = DEFAULT_CONFIG.copy()
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {self.config_path}. Using default configuration.")
                self.config = DEFAULT_CONFIG.copy()
        return self.config

    def validate_config(self, config: dict) -> bool:
        """
        Validate configuration parameters based on predefined rules.
        
        :param config: Configuration dictionary to validate
        :return: True if valid, False otherwise
        """
        logger = logging.getLogger(self.__class__.__name__)
        for param, rules in VALIDATION_RULES.items():
            if param not in config:
                logger.error(f"Missing configuration parameter: '{param}'")
                return False
            
            value = config[param]
            expected_type = rules["type"]
            
            # Type Validation
            if not isinstance(value, expected_type):
                logger.error(f"Invalid type for '{param}': Expected {expected_type.__name__}, got {type(value).__name__}")
                return False
            
            # Min Value Validation
            if "min" in rules and value < rules["min"]:
                logger.error(f"Value for '{param}' is below the minimum of {rules['min']}.")
                return False
            
            # Max Value Validation
            if "max" in rules and value > rules["max"]:
                logger.error(f"Value for '{param}' is above the maximum of {rules['max']}.")
                return False
            
            # Precision Validation
            if "precision" in rules and isinstance(value, float):
                decimal_places = len(str(value).split(".")[1])
                if decimal_places > rules["precision"]:
                    logger.error(f"'{param}' must have {rules['precision']} decimal place(s) precision.")
                    return False
        
        logger.info("All configuration parameters are valid.")
        return True

    def get_parameter(self, name, default=None):
        """
        Retrieve a parameter value from the configuration.
        
        :param name: Name of the parameter
        :param default: Default value if parameter is not found
        :return: Parameter value
        """
        return self.config.get(name, default)

    def set_parameter(self, name, value):
        """
        Set a parameter value in the configuration.
        
        :param name: Name of the parameter
        :param value: Value to set
        """
        self.logger.debug(f"Attempting to set parameter '{name}' to '{value}'.")
        # Validate the parameter before setting
        if name in VALIDATION_RULES:
            rules = VALIDATION_RULES[name]
            if not isinstance(value, rules["type"]):
                self.logger.error(f"Invalid type for '{name}': Expected {rules['type'].__name__}, got {type(value).__name__}")
                return
            if "min" in rules and value < rules["min"]:
                self.logger.error(f"Value for '{name}' is below the minimum of {rules['min']}.")
                return
            if "max" in rules and value > rules["max"]:
                self.logger.error(f"Value for '{name}' is above the maximum of {rules['max']}.")
                return
            if "precision" in rules and isinstance(value, float):
                decimal_places = len(str(value).split(".")[1])
                if decimal_places > rules["precision"]:
                    self.logger.error(f"'{name}' must have {rules['precision']} decimal place(s) precision.")
                    return
        # Set the parameter if all validations pass
        self.config[name] = value
        self.save_config()
        self.logger.info(f"Parameter '{name}' set to '{value}' successfully.")

    def save_config(self):
        """Save the current configuration to the JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info("Configuration saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def get_all_parameters(self) -> dict:
        """
        Return a copy of all configuration parameters.
        
        :return: A dictionary of all configuration parameters
        """
        return self.config.copy()

    def reset_to_defaults(self) -> dict:
        """
        Reset configuration to default parameters.
        
        :return: A copy of the default configuration
        """
        return load_default_config().copy()