"""
Configuration management.

Supports:
- TOML config files (.iexplain.toml)
- Environment variable overrides
- Sensible defaults
- Validation
"""

import os
from pathlib import Path

# TOML parsing - Python 3.11+ has tomllib built-in
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


@dataclass
class WorkspaceConfig:
    """Workspace directory configuration."""
    tools_dir: Optional[str] = None
    agents_dir: Optional[str] = None
    workspace_dir: Optional[str] = None


@dataclass
class SandboxConfig:
    """Sandbox execution configuration."""
    timeout: int = 10
    allow_write: bool = False


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_iterations: int = 10
    fail_on_tool_error: bool = False  # Configurable error handling


@dataclass
class LoggingConfig:
    """Logging configuration."""
    enabled: bool = True
    log_dir: Optional[str] = None
    include_messages: bool = True  # Log full LLM messages
    include_results: bool = True   # Log tool results
    retention_days: Optional[int] = 30


@dataclass
class Config:
    """Main configuration object."""
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_file(cls, path: Path) -> 'Config':
        """
        Load configuration from TOML file.
        
        Args:
            path: Path to .iexplain.toml file
        
        Returns:
            Config object
        
        Raises:
            ConfigError: If file cannot be loaded
        """
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        
        try:
            with open(path, 'rb') as f:
                data = tomllib.load(f)
            
            return cls._from_dict(data)
        
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"Invalid TOML in {path}: {e}") from e
        
        except Exception as e:
            raise ConfigError(f"Failed to load config from {path}: {e}") from e
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        config = cls()
        
        # Workspace config
        if 'workspace' in data:
            ws = data['workspace']
            config.workspace = WorkspaceConfig(
                tools_dir=ws.get('tools_dir'),
                agents_dir=ws.get('agents_dir'),
                workspace_dir=ws.get('workspace_dir')
            )
        
        # Sandbox config
        if 'sandbox' in data:
            sb = data['sandbox']
            config.sandbox = SandboxConfig(
                timeout=sb.get('timeout', 10),
                allow_write=sb.get('allow_write', False)
            )
        
        # LLM config
        if 'llm' in data:
            llm = data['llm']
            config.llm = LLMConfig(
                provider=llm.get('provider', 'anthropic'),
                model=llm.get('model', 'claude-sonnet-4-20250514'),
                temperature=llm.get('temperature', 0.0),
                max_tokens=llm.get('max_tokens', 4096)
            )
        
        # Agent config
        if 'agent' in data:
            agent = data['agent']
            config.agent = AgentConfig(
                max_iterations=agent.get('max_iterations', 10),
                fail_on_tool_error=agent.get('fail_on_tool_error', False)
            )
        
        # Logging config
        if 'logging' in data:
            log = data['logging']
            config.logging = LoggingConfig(
                enabled=log.get('enabled', True),
                log_dir=log.get('log_dir'),
                include_messages=log.get('include_messages', True),
                include_results=log.get('include_results', True),
                retention_days=log.get('retention_days', 30)
            )
        
        return config
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Load configuration from environment variables.
        
        Environment variables:
            IEXPLAIN_TOOLS_DIR
            IEXPLAIN_AGENTS_DIR
            IEXPLAIN_WORKSPACE_DIR
            IEXPLAIN_SANDBOX_TIMEOUT
            IEXPLAIN_LLM_PROVIDER
            IEXPLAIN_LLM_MODEL
            IEXPLAIN_LLM_TEMPERATURE
            IEXPLAIN_LOGGING_ENABLED
            IEXPLAIN_LOG_DIR
        
        Returns:
            Config object with environment overrides
        """
        config = cls()
        
        # Workspace
        if tools_dir := os.getenv('IEXPLAIN_TOOLS_DIR'):
            config.workspace.tools_dir = tools_dir
        if agents_dir := os.getenv('IEXPLAIN_AGENTS_DIR'):
            config.workspace.agents_dir = agents_dir
        if workspace_dir := os.getenv('IEXPLAIN_WORKSPACE_DIR'):
            config.workspace.workspace_dir = workspace_dir
        
        # Sandbox
        if timeout := os.getenv('IEXPLAIN_SANDBOX_TIMEOUT'):
            try:
                config.sandbox.timeout = int(timeout)
            except ValueError:
                raise ConfigError(f"Invalid IEXPLAIN_SANDBOX_TIMEOUT: {timeout}")
        
        # LLM
        if provider := os.getenv('IEXPLAIN_LLM_PROVIDER'):
            config.llm.provider = provider
        if model := os.getenv('IEXPLAIN_LLM_MODEL'):
            config.llm.model = model
        if temperature := os.getenv('IEXPLAIN_LLM_TEMPERATURE'):
            try:
                config.llm.temperature = float(temperature)
            except ValueError:
                raise ConfigError(f"Invalid IEXPLAIN_LLM_TEMPERATURE: {temperature}")
        
        # Agent
        if max_iter := os.getenv('IEXPLAIN_AGENT_MAX_ITERATIONS'):
            try:
                config.agent.max_iterations = int(max_iter)
            except ValueError:
                raise ConfigError(f"Invalid IEXPLAIN_AGENT_MAX_ITERATIONS: {max_iter}")
        
        # Logging
        if enabled := os.getenv('IEXPLAIN_LOGGING_ENABLED'):
            config.logging.enabled = enabled.lower() in ('true', '1', 'yes', 'on')
        if log_dir := os.getenv('IEXPLAIN_LOG_DIR'):
            config.logging.log_dir = log_dir
        
        return config
    
    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> 'Config':
        """
        Load configuration with priority: env > file > defaults.
        
        Args:
            config_file: Optional explicit config file path
        
        Returns:
            Config object with all sources merged
        """
        # Start with defaults
        config = cls()
        
        # Find and load config file
        if config_file is None:
            config_file = cls._find_config_file()
        
        if config_file and config_file.exists():
            file_config = cls.from_file(config_file)
            config = cls._merge(config, file_config)
        
        # Apply environment overrides
        env_config = cls.from_env()
        config = cls._merge(config, env_config)
        
        return config
    
    @classmethod
    def _find_config_file(cls) -> Optional[Path]:
        """
        Find config file in standard locations.
        
        Search order:
        1. ./.iexplain.toml (project)
        2. ~/.config/iexplain/config.toml (user)
        
        Returns:
            Path to config file, or None if not found
        """
        # Check current directory
        project_config = Path.cwd() / ".iexplain.toml"
        if project_config.exists():
            return project_config
        
        # Check user config directory
        user_config_dir = Path.home() / ".config" / "iexplain"
        user_config = user_config_dir / "config.toml"
        if user_config.exists():
            return user_config
        
        return None
    
    @classmethod
    def _merge(cls, base: 'Config', override: 'Config') -> 'Config':
        """
        Merge two configs, with override taking precedence.
        
        Only non-None/non-default values from override are applied.
        """
        merged = cls()
        
        # Helper to check if value is default
        defaults = cls()
        
        # Merge workspace
        merged.workspace = WorkspaceConfig(
            tools_dir=override.workspace.tools_dir if override.workspace.tools_dir is not None else base.workspace.tools_dir,
            agents_dir=override.workspace.agents_dir if override.workspace.agents_dir is not None else base.workspace.agents_dir,
            workspace_dir=override.workspace.workspace_dir if override.workspace.workspace_dir is not None else base.workspace.workspace_dir
        )
        
        # Merge sandbox (only override if different from default)
        merged.sandbox = SandboxConfig(
            timeout=override.sandbox.timeout if override.sandbox.timeout != defaults.sandbox.timeout else base.sandbox.timeout,
            allow_write=override.sandbox.allow_write if override.sandbox.allow_write != defaults.sandbox.allow_write else base.sandbox.allow_write
        )
        
        # Merge LLM (only override if different from default)
        merged.llm = LLMConfig(
            provider=override.llm.provider if override.llm.provider != defaults.llm.provider else base.llm.provider,
            model=override.llm.model if override.llm.model != defaults.llm.model else base.llm.model,
            temperature=override.llm.temperature if override.llm.temperature != defaults.llm.temperature else base.llm.temperature,
            max_tokens=override.llm.max_tokens if override.llm.max_tokens != defaults.llm.max_tokens else base.llm.max_tokens
        )
        
        # Merge agent
        merged.agent = AgentConfig(
            max_iterations=override.agent.max_iterations if override.agent.max_iterations != defaults.agent.max_iterations else base.agent.max_iterations,
            fail_on_tool_error=override.agent.fail_on_tool_error if override.agent.fail_on_tool_error != defaults.agent.fail_on_tool_error else base.agent.fail_on_tool_error
        )
        
        # Merge logging
        merged.logging = LoggingConfig(
            enabled=override.logging.enabled if override.logging.enabled != defaults.logging.enabled else base.logging.enabled,
            log_dir=override.logging.log_dir if override.logging.log_dir is not None else base.logging.log_dir,
            include_messages=override.logging.include_messages if override.logging.include_messages != defaults.logging.include_messages else base.logging.include_messages,
            include_results=override.logging.include_results if override.logging.include_results != defaults.logging.include_results else base.logging.include_results,
            retention_days=override.logging.retention_days if override.logging.retention_days != defaults.logging.retention_days else base.logging.retention_days
        )
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, excluding None values."""
        def clean_dict(d: dict) -> dict:
            """Remove None values recursively."""
            return {k: clean_dict(v) if isinstance(v, dict) else v 
                    for k, v in d.items() if v is not None}
        
        data = {
            'workspace': asdict(self.workspace),
            'sandbox': asdict(self.sandbox),
            'llm': asdict(self.llm),
            'agent': asdict(self.agent),
            'logging': asdict(self.logging)
        }
        
        return clean_dict(data)
    
    def save(self, path: Path) -> None:
        """
        Save configuration to TOML file.
        
        Args:
            path: Path to save config
        """
        import tomli_w  # For writing TOML
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                tomli_w.dump(self.to_dict(), f)
        
        except Exception as e:
            raise ConfigError(f"Failed to save config to {path}: {e}") from e
