import json
import os

DEFAULT_CONFIG = {
    "High-Pass Filter Radius": 10.0,
    "Gaussian Blur (%)": 1.0,
    "Peak Min Distance": 10,
    "Peak Threshold": 0.0010,
    "Radius (%)": 10.0,
    "Aspect Ratio": 1.0,
    "Orientation": 0,
    "Falloff (%)": 0.0,
    "Mask Radius (%)": 1.0,
    "Peak Mask Falloff (%)": 0.0,
    "Gamma Correction": 1.0,
    "Anti-Aliasing Intensity (%)": 50.0,
    "Enable Frequency Peak Suppression": True,
    "Enable Attenuation": True,
    "Enable Anti-Aliasing Filter": True
}

def load_config(config_path):
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG.copy()
    with open(config_path, 'r') as f:
        try:
            config = json.load(f)
            return config
        except json.JSONDecodeError:
            return DEFAULT_CONFIG.copy()

def save_config(config, config_path):
    """Save configuration to a JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def reset_to_defaults():
    """Reset configuration to default parameters."""
    return DEFAULT_CONFIG.copy()