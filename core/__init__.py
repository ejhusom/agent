"""
iExplain initialization and setup.

This module provides easy initialization of the complete iExplain system
with Config + Workspace + Logger integration.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .config import Config
from .workspace import Workspace
from .logger import ConversationLogger


@dataclass
class IExplainContext:
    """
    Container for iExplain system components.
    
    Bundles Config, Workspace, and Logger for easy passing to agents.
    """
    config: Config
    workspace: Workspace
    logger: Optional[ConversationLogger]
    
    def close(self):
        """Close resources."""
        if self.logger:
            self.logger.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def initialize(
    config_file: Optional[Path] = None,
    user_query: Optional[str] = None,
    enable_logging: Optional[bool] = None
) -> IExplainContext:
    """
    Initialize iExplain system with all components.
    
    Args:
        config_file: Optional explicit config file path
        user_query: Optional user query for logging
        enable_logging: Override logging enabled setting
    
    Returns:
        IExplainContext with config, workspace, and logger
    
    Example:
        with initialize(user_query="Analyze logs") as ctx:
            supervisor = Supervisor(
                llm_client=...,
                context=ctx
            )
            result = supervisor.run("Analyze error.log")
    """
    # Load configuration
    config = Config.load(config_file=config_file)
    
    # Override logging if specified
    if enable_logging is not None:
        config.logging.enabled = enable_logging
    
    # Create workspace
    workspace = Workspace(
        tools_dir=config.workspace.tools_dir,
        agents_dir=config.workspace.agents_dir,
        workspace_dir=config.workspace.workspace_dir
    )
    
    # Create logger if enabled
    logger = None
    if config.logging.enabled:
        log_dir = Path(config.logging.log_dir) if config.logging.log_dir else workspace.workspace_dir / "logs"
        log_dir = log_dir.expanduser().resolve()
        
        logger = ConversationLogger(
            log_dir=log_dir,
            config=config.to_dict(),
            user_query=user_query
        )
    
    return IExplainContext(
        config=config,
        workspace=workspace,
        logger=logger
    )
