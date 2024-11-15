#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from loguru import logger


@dataclass
class NodeConfig:
    """Configuration for a single node in the conversation flow.

    Attributes:
        message: Dict containing role and content for the LLM at this node
        functions: List of available function definitions for this node
        actions: Optional list of actions to execute when entering this node
    """

    message: dict
    functions: List[dict]
    actions: Optional[List[dict]] = None


class ConversationFlow:
    """Manages state transitions in a conversation flow.

    This class handles the state machine logic for conversation flows, where each state
    (node) has its own message, available functions, and optional actions. It manages
    transitions between states based on function calls and handles both regular and
    terminal functions.

    Attributes:
        nodes: Dictionary mapping node IDs to their configurations
        current_node: ID of the currently active node
    """

    def __init__(self, flow_config: dict):
        """Initialize the conversation flow.

        Args:
            flow_config: Dictionary containing the complete flow configuration,
                        must include 'initial_node' and 'nodes' keys

        Raises:
            ValueError: If required configuration keys are missing
        """
        self.nodes: Dict[str, NodeConfig] = {}
        self.current_node: str = flow_config["initial_node"]
        self._load_config(flow_config)

    def _load_config(self, config: dict):
        """Load and validate the flow configuration.

        Args:
            config: Dictionary containing the flow configuration

        Raises:
            ValueError: If required configuration keys are missing
        """
        if "initial_node" not in config:
            raise ValueError("Flow config must specify 'initial_node'")
        if "nodes" not in config:
            raise ValueError("Flow config must specify 'nodes'")

        for node_id, node_config in config["nodes"].items():
            self.nodes[node_id] = NodeConfig(
                message=node_config["message"],
                functions=node_config["functions"],
                actions=node_config.get("actions"),
            )

    def get_current_message(self) -> dict:
        """Get the message configuration for the current node.

        Returns:
            Dictionary containing role and content for the current node's message
        """
        return self.nodes[self.current_node].message

    def get_current_functions(self) -> List[dict]:
        """Get the available functions for the current node.

        Returns:
            List of function definitions available in the current node
        """
        return self.nodes[self.current_node].functions

    def get_current_actions(self) -> Optional[List[dict]]:
        """Get the actions for the current node.

        Returns:
            List of actions to execute when entering the node, or None if no actions
        """
        return self.nodes[self.current_node].actions

    def get_available_function_names(self) -> Set[str]:
        """Get the names of available functions for the current node.

        Returns:
            Set of function names that can be called from the current node
        """
        names = {f["function"]["name"] for f in self.nodes[self.current_node].functions}
        logger.debug(f"Available function names for node {self.current_node}: {names}")
        return names

    def transition(self, function_name: str) -> Optional[str]:
        """Attempt to transition based on a function call.

        Handles both regular transitions (where the function name matches a node)
        and terminal functions (which execute but don't change nodes).

        Args:
            function_name: Name of the function that was called

        Returns:
            The name of the new node after transition, or None if transition failed.
            For terminal functions, returns the current node name.
        """
        available_functions = self.get_available_function_names()
        logger.debug(f"Attempting transition from {self.current_node} to {function_name}")

        if function_name in available_functions:
            if function_name in self.nodes:
                # Regular transition to a new node
                self.current_node = function_name
                logger.info(f"Transitioned to node: {self.current_node}")
                return self.current_node
            else:
                # Handle terminal function calls
                logger.info(f"Executed terminal function: {function_name}")
                return self.current_node
        return None
