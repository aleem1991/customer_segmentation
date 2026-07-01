import os
import yaml
from typing import Any, Dict

# Determine the absolute project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config() -> Dict[str, Any]:
    """Loads and parses the config.yaml file, resolving relative paths to absolute ones."""
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Dynamically resolve relative paths to absolute paths
    if "paths" in config:
        for key, rel_path in config["paths"].items():
            config["paths"][key] = os.path.abspath(os.path.join(PROJECT_ROOT, rel_path))
            
    return config

# Load configuration dynamically on import
CONFIG = load_config()
