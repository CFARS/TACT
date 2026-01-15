from typing import Protocol
import json

class ConfigReader(Protocol):
    def get_column_mapping(self) -> dict:
        pass

class JsonConfigReader:
    def __init__(self, config_path: str):
        self.config_path = config_path
    
    def get_column_mapping(self) -> dict:
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            return config['input_data_column_mapping']