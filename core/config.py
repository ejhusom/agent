import os
import tomli

class Config:
    """
    A simple class for reading and storing configuration from a config.toml file.

    Usage:
    1. Create an instance of the Config class by providing the path to the config.toml file.
       Example: config = Config("path/to/config.toml")
    2. Access the configuration values as attributes.
       Example: value = config.data["section"]["key"]
    """

    def __init__(self, config_path: str):
        """
        Initialize the Config class by loading the configuration file.

        :param config_path: Path to the config.toml file.
        """
        self.config = self._load_config(config_path)

        self.config["provider"] = self.config.get("provider", "anthropic")
        self.config["model"] = self.config.get("model", "claude-sonnet-4-20250514")
        self.config["temperature"] = self.config.get("temperature", 0.0)
        self.config["max_tokens"] = self.config.get("max_tokens", 8192)
        self.config["sandbox_timeout"] = self.config.get("sandbox_timeout", 60)
        self.config["sandbox_workspace"] = self.config.get("sandbox_workspace", "./sandbox")
        self.config["sandbox_allow_write"] = self.config.get("sandbox_allow_write", False)
        self.config["agents_persist_dir"] = self.config.get("agents_persist_dir", "./persistent-agents")
        self.config["tools_persist_dir"] = self.config.get("tools_persist_dir", "./persistent-tools")

        if "api_key" in self.config:
            self.config["api_key"] = self.config["api_key"]
        elif self.config["provider"] in ["anthropic", "openai"]:
            try:
                if self.config["provider"] == "anthropic":
                    self.config["api_key"] = os.environ["ANTHROPIC_API_KEY"]
                elif self.config["provider"] == "openai":
                    self.config["api_key"] = os.environ["OPENAI_API_KEY"]
            except KeyError:
                self.config["api_key"] = None
                print(f"API key not found for provider: {self.config['provider']}")
        else:
            self.config["api_key"] = None

    def _load_config(self, config_path: str) -> dict:
        """
        Load the configuration file.

        :param config_path: Path to the config.toml file.
        :return: Parsed configuration as a dictionary.
        """
        try:
            with open(config_path, "rb") as file:
                return tomli.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        except tomli.TOMLDecodeError as e:
            raise ValueError(f"Error parsing config file: {e}")

config = Config("config.toml").config