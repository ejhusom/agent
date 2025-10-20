"""
Tests for configuration system.

Tests cover:
- Loading from TOML files
- Environment variable overrides
- Config merging (priority: env > file > defaults)
- Invalid config handling
- Config file discovery
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import (
    Config,
    ConfigError,
    WorkspaceConfig,
    SandboxConfig,
    LLMConfig,
    AgentConfig,
    LoggingConfig
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    import shutil
    shutil.rmtree(temp)


@pytest.fixture
def sample_config_toml():
    """Sample valid TOML config."""
    return """
[workspace]
tools_dir = "/custom/tools"
agents_dir = "/custom/agents"

[sandbox]
timeout = 30
allow_write = true

[llm]
provider = "openai"
model = "gpt-4"
temperature = 0.7
max_tokens = 2000

[agent]
max_iterations = 20
fail_on_tool_error = true

[logging]
enabled = false
log_dir = "/var/log/iexplain"
include_messages = false
include_results = false
retention_days = 7
"""


# ============================================================================
# Default Config Tests
# ============================================================================

def test_default_config():
    """Test that default config has sensible values."""
    config = Config()
    
    # Workspace defaults
    assert config.workspace.tools_dir is None
    assert config.workspace.agents_dir is None
    
    # Sandbox defaults
    assert config.sandbox.timeout == 10
    assert config.sandbox.allow_write is False
    
    # LLM defaults
    assert config.llm.provider == "anthropic"
    assert config.llm.model == "claude-sonnet-4-20250514"
    assert config.llm.temperature == 0.0
    assert config.llm.max_tokens == 4096
    
    # Agent defaults
    assert config.agent.max_iterations == 10
    assert config.agent.fail_on_tool_error is False
    
    # Logging defaults
    assert config.logging.enabled is True
    assert config.logging.include_messages is True
    assert config.logging.include_results is True
    assert config.logging.retention_days == 30


# ============================================================================
# File Loading Tests
# ============================================================================

def test_load_from_file(temp_dir, sample_config_toml):
    """Test loading config from TOML file."""
    config_file = temp_dir / "config.toml"
    config_file.write_text(sample_config_toml)
    
    config = Config.from_file(config_file)
    
    # Check loaded values
    assert config.workspace.tools_dir == "/custom/tools"
    assert config.sandbox.timeout == 30
    assert config.llm.provider == "openai"
    assert config.agent.max_iterations == 20
    assert config.logging.enabled is False


def test_load_nonexistent_file(temp_dir):
    """Test loading from nonexistent file raises error."""
    config_file = temp_dir / "nonexistent.toml"
    
    with pytest.raises(ConfigError, match="Config file not found"):
        Config.from_file(config_file)


def test_load_invalid_toml(temp_dir):
    """Test loading invalid TOML raises error."""
    config_file = temp_dir / "bad.toml"
    config_file.write_text("this is not [ valid toml {")
    
    with pytest.raises(ConfigError, match="Invalid TOML"):
        Config.from_file(config_file)


def test_load_partial_config(temp_dir):
    """Test loading config with only some sections."""
    config_file = temp_dir / "partial.toml"
    config_file.write_text("""
[llm]
model = "gpt-4"
""")
    
    config = Config.from_file(config_file)
    
    # LLM section loaded
    assert config.llm.model == "gpt-4"
    
    # Other sections have defaults
    assert config.sandbox.timeout == 10
    assert config.agent.max_iterations == 10


# ============================================================================
# Environment Variable Tests
# ============================================================================

def test_load_from_env():
    """Test loading config from environment variables."""
    env = {
        'IEXPLAIN_TOOLS_DIR': '/env/tools',
        'IEXPLAIN_SANDBOX_TIMEOUT': '60',
        'IEXPLAIN_LLM_PROVIDER': 'ollama',
        'IEXPLAIN_LLM_MODEL': 'llama2',
        'IEXPLAIN_LLM_TEMPERATURE': '0.5',
        'IEXPLAIN_AGENT_MAX_ITERATIONS': '15',
        'IEXPLAIN_LOGGING_ENABLED': 'false',
        'IEXPLAIN_LOG_DIR': '/env/logs'
    }
    
    with patch.dict('os.environ', env):
        config = Config.from_env()
    
    assert config.workspace.tools_dir == '/env/tools'
    assert config.sandbox.timeout == 60
    assert config.llm.provider == 'ollama'
    assert config.llm.model == 'llama2'
    assert config.llm.temperature == 0.5
    assert config.agent.max_iterations == 15
    assert config.logging.enabled is False
    assert config.logging.log_dir == '/env/logs'


def test_env_invalid_timeout():
    """Test that invalid timeout in env raises error."""
    with patch.dict('os.environ', {'IEXPLAIN_SANDBOX_TIMEOUT': 'not-a-number'}):
        with pytest.raises(ConfigError, match="Invalid IEXPLAIN_SANDBOX_TIMEOUT"):
            Config.from_env()


def test_env_invalid_temperature():
    """Test that invalid temperature in env raises error."""
    with patch.dict('os.environ', {'IEXPLAIN_LLM_TEMPERATURE': 'not-a-float'}):
        with pytest.raises(ConfigError, match="Invalid IEXPLAIN_LLM_TEMPERATURE"):
            Config.from_env()


def test_env_logging_enabled_values():
    """Test various boolean values for logging enabled."""
    true_values = ['true', 'True', '1', 'yes', 'YES', 'on', 'ON']
    
    for val in true_values:
        with patch.dict('os.environ', {'IEXPLAIN_LOGGING_ENABLED': val}):
            config = Config.from_env()
            assert config.logging.enabled is True, f"Failed for '{val}'"
    
    false_values = ['false', 'False', '0', 'no', 'NO', 'off', 'OFF']
    
    for val in false_values:
        with patch.dict('os.environ', {'IEXPLAIN_LOGGING_ENABLED': val}):
            config = Config.from_env()
            assert config.logging.enabled is False, f"Failed for '{val}'"


# ============================================================================
# Config Discovery Tests
# ============================================================================

def test_find_config_project(temp_dir):
    """Test finding project config file."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("[llm]\nmodel = 'test'")
    
    with patch('pathlib.Path.cwd', return_value=temp_dir):
        found = Config._find_config_file()
        assert found == config_file


def test_find_config_user(temp_dir):
    """Test finding user config file."""
    user_config_dir = temp_dir / ".config" / "iexplain"
    user_config_dir.mkdir(parents=True)
    config_file = user_config_dir / "config.toml"
    config_file.write_text("[llm]\nmodel = 'test'")
    
    with patch('pathlib.Path.cwd', return_value=temp_dir / "other"), \
         patch('pathlib.Path.home', return_value=temp_dir):
        found = Config._find_config_file()
        assert found == config_file


def test_find_config_none():
    """Test that None is returned when no config found."""
    with patch('pathlib.Path.cwd', return_value=Path('/nonexistent')), \
         patch('pathlib.Path.home', return_value=Path('/nonexistent')):
        found = Config._find_config_file()
        assert found is None


# ============================================================================
# Config Merging Tests
# ============================================================================

def test_merge_configs():
    """Test merging two configs with override priority."""
    base = Config()
    base.llm.model = "base-model"
    base.sandbox.timeout = 10
    
    override = Config()
    override.llm.model = "override-model"
    override.sandbox.timeout = 30
    
    merged = Config._merge(base, override)
    
    assert merged.llm.model == "override-model"
    assert merged.sandbox.timeout == 30


def test_merge_partial_override():
    """Test merging where override only sets some values."""
    base = Config()
    base.workspace.tools_dir = "/base/tools"
    base.workspace.agents_dir = "/base/agents"
    
    override = Config()
    override.workspace.tools_dir = "/override/tools"
    # agents_dir not set in override
    
    merged = Config._merge(base, override)
    
    assert merged.workspace.tools_dir == "/override/tools"
    assert merged.workspace.agents_dir == "/base/agents"


def test_load_with_priority(temp_dir, sample_config_toml):
    """Test full load with env > file > defaults priority."""
    # Create config file
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text(sample_config_toml)
    
    # Set env var (clear existing env to isolate test)
    env = {'IEXPLAIN_LLM_MODEL': 'env-model'}
    
    with patch('pathlib.Path.cwd', return_value=temp_dir), \
         patch.dict('os.environ', env, clear=True):
        config = Config.load()
    
    # From env (highest priority)
    assert config.llm.model == 'env-model'
    
    # From file
    assert config.llm.provider == 'openai'
    assert config.sandbox.timeout == 30


# ============================================================================
# Config Serialization Tests
# ============================================================================

def test_to_dict():
    """Test converting config to dictionary."""
    config = Config()
    config.llm.model = "test-model"
    config.sandbox.timeout = 99
    
    data = config.to_dict()
    
    assert data['llm']['model'] == "test-model"
    assert data['sandbox']['timeout'] == 99


def test_save_config(temp_dir):
    """Test saving config to file."""
    config = Config()
    config.llm.model = "saved-model"
    config.sandbox.timeout = 45
    
    config_file = temp_dir / "saved.toml"
    config.save(config_file)
    
    assert config_file.exists()
    
    # Load it back
    loaded = Config.from_file(config_file)
    assert loaded.llm.model == "saved-model"
    assert loaded.sandbox.timeout == 45


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_config_file(temp_dir):
    """Test loading empty config file."""
    config_file = temp_dir / "empty.toml"
    config_file.write_text("")
    
    config = Config.from_file(config_file)
    
    # Should have all defaults
    assert config.llm.model == "claude-sonnet-4-20250514"
    assert config.sandbox.timeout == 10


def test_config_with_unknown_sections(temp_dir):
    """Test that unknown sections don't break loading."""
    config_file = temp_dir / "extra.toml"
    config_file.write_text("""
[llm]
model = "test"

[unknown_section]
unknown_key = "value"
""")
    
    config = Config.from_file(config_file)
    
    # Known section loaded
    assert config.llm.model == "test"
    
    # Unknown section ignored (no error)


def test_config_with_extra_keys(temp_dir):
    """Test that extra keys in known sections don't break loading."""
    config_file = temp_dir / "extra_keys.toml"
    config_file.write_text("""
[llm]
model = "test"
extra_key = "ignored"
""")
    
    config = Config.from_file(config_file)
    
    # Known key loaded
    assert config.llm.model == "test"
    
    # Extra key ignored


def test_explicit_config_file():
    """Test loading with explicit config file path."""
    temp = tempfile.mktemp(suffix=".toml")
    try:
        Path(temp).write_text('[llm]\nmodel = "explicit"')
        
        config = Config.load(config_file=Path(temp))
        assert config.llm.model == "explicit"
    
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
