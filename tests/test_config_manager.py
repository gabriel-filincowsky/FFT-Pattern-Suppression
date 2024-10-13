import unittest
import os
import json
from unittest.mock import patch, mock_open
from utils.config_manager import ConfigManager, DEFAULT_CONFIG, VALIDATION_RULES
from utils.constants import (
    MAX_HIGH_PASS_RADIUS,
    MIN_HIGH_PASS_RADIUS,
    MAX_GAUSSIAN_BLUR_PCT,
    MIN_GAUSSIAN_BLUR_PCT,
    MAX_PEAK_THRESHOLD,
    MIN_PEAK_THRESHOLD,
)

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.test_config_path = "config/test_config.json"
        self.config_manager = ConfigManager(config_path=self.test_config_path)

    def tearDown(self):
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps(DEFAULT_CONFIG))
    def test_load_valid_config(self, mock_file):
        config = self.config_manager.load_config()
        self.assertTrue(self.config_manager.validate_config(config))
        self.assertEqual(config["High-Pass Filter Radius"], DEFAULT_CONFIG["High-Pass Filter Radius"])
        self.assertEqual(config["Enable Frequency Peak Suppression"], DEFAULT_CONFIG["Enable Frequency Peak Suppression"])

    def test_invalid_parameter_type(self):
        invalid_config = DEFAULT_CONFIG.copy()
        invalid_config["High-Pass Filter Radius"] = "invalid_type"
        self.assertFalse(self.config_manager.validate_config(invalid_config))

    def test_missing_parameter(self):
        invalid_config = DEFAULT_CONFIG.copy()
        del invalid_config["Gaussian Blur (%)"]
        self.assertFalse(self.config_manager.validate_config(invalid_config))

    def test_min_max_constraints(self):
        invalid_config = DEFAULT_CONFIG.copy()
        
        # Test below minimum
        invalid_config["High-Pass Filter Radius"] = MIN_HIGH_PASS_RADIUS - 0.1
        self.assertFalse(self.config_manager.validate_config(invalid_config))
        
        # Test above maximum
        invalid_config["High-Pass Filter Radius"] = MAX_HIGH_PASS_RADIUS + 0.1
        self.assertFalse(self.config_manager.validate_config(invalid_config))
        
        # Test valid value
        invalid_config["High-Pass Filter Radius"] = (MIN_HIGH_PASS_RADIUS + MAX_HIGH_PASS_RADIUS) / 2
        self.assertTrue(self.config_manager.validate_config(invalid_config))

    def test_precision_constraint(self):
        invalid_config = DEFAULT_CONFIG.copy()
        invalid_config["High-Pass Filter Radius"] = 10.05  # More than one decimal place
        self.assertFalse(self.config_manager.validate_config(invalid_config))

        invalid_config["High-Pass Filter Radius"] = 10.0  # Exactly one decimal place
        self.assertTrue(self.config_manager.validate_config(invalid_config))

    def test_set_parameter(self):
        self.config_manager.set_parameter("High-Pass Filter Radius", 15.0)
        self.assertEqual(self.config_manager.get_parameter("High-Pass Filter Radius"), 15.0)

    def test_get_all_parameters(self):
        all_params = self.config_manager.get_all_parameters()
        self.assertEqual(all_params, self.config_manager.config)
        self.assertIsNot(all_params, self.config_manager.config)  # Ensure it's a copy

    @patch('utils.config_manager.load_default_config')
    def test_reset_to_defaults(self, mock_load_default):
        mock_load_default.return_value = DEFAULT_CONFIG
        reset_config = self.config_manager.reset_to_defaults()
        self.assertEqual(reset_config, DEFAULT_CONFIG)
        self.assertIsNot(reset_config, DEFAULT_CONFIG)  # Ensure it's a copy

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_config(self, mock_json_dump, mock_file):
        self.config_manager.save_config()
        mock_file.assert_called_once_with(self.test_config_path, 'w')
        mock_json_dump.assert_called_once_with(self.config_manager.config, mock_file(), indent=4)

if __name__ == '__main__':
    unittest.main()