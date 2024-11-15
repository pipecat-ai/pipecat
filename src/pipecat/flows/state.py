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
    """Configuration for a single node in the flow.

    A node represents a state in the conversation flow, containing all the
    information needed for that particular point in the conversation.

    Attributes:
        messages: List of message dicts to be added to LLM context at this node.
                 Each message should have 'role' (system/user/assistant) and 'content'.
                 Messages are added in order, allowing for complex prompt building.
        functions: List of available function definitions for this node
        pre_actions: Optional list of actions to execute before LLM inference
        post_actions: Optional list of actions to execute after LLM inference
    """

    messages: List[dict]
    functions: List[dict]
    pre_actions: Optional[List[dict]] = None
    post_actions: Optional[List[dict]] = None


class FlowState:
    """Manages the state and transitions between nodes in a conversation flow.

    This class handles the state machine logic for conversation flows, where each node
    represents a distinct state with its own messages, available functions, and optional
    pre- and post-actions. It manages transitions between nodes based on function calls
    and handles both regular and terminal functions.

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
                messages=node_config["messages"],
                functions=node_config["functions"],
                pre_actions=node_config.get("pre_actions"),
                post_actions=node_config.get("post_actions"),
            )

    def get_current_messages(self) -> List[dict]:
        """Get the messages for the current node.

        Returns:
            List of message dictionaries for the current node
        """
        return self.nodes[self.current_node].messages

    def get_current_functions(self) -> List[dict]:
        """Get the available functions for the current node.

        Returns:
            List of function definitions available in the current node
        """
        return self.nodes[self.current_node].functions

    def get_current_pre_actions(self) -> Optional[List[dict]]:
        """Get the pre-actions for the current node.

        Pre-actions are executed before updating the LLM context when
        transitioning to this node.

        Returns:
            List of pre-actions to execute, or None if no pre-actions
        """
        return self.nodes[self.current_node].pre_actions

    def get_current_post_actions(self) -> Optional[List[dict]]:
        """Get the post-actions for the current node.

        Post-actions are executed after updating the LLM context when
        transitioning to this node.

        Returns:
            List of post-actions to execute, or None if no post-actions
        """
        return self.nodes[self.current_node].post_actions

    def get_available_function_names(self) -> Set[str]:
        """Get the names of available functions for the current node.

        Returns:
            Set of function names that can be called from the current node
        """
        names = {f["function"]["name"] for f in self.nodes[self.current_node].functions}
        logger.debug(f"Available function names for node {self.current_node}: {names}")
        return names

    def transition(self, function_name: str) -> Optional[str]:
        """Attempt to transition to a new node based on a function call.

        Handles both regular transitions (where the function name matches a node)
        and terminal functions (which execute but don't change nodes).

        Args:
            function_name: Name of the function that was called

        Returns:
            The ID of the new node after transition, or None if transition failed.
            For terminal functions, returns the current node ID.
        """
        available_functions = self.get_available_function_names()
        logger.debug(f"Attempting transition from {self.current_node} to {function_name}")

        if function_name in available_functions:
            if function_name in self.nodes:
                # Regular transition to a new node
                previous_node = self.current_node
                self.current_node = function_name
                logger.info(f"Transitioned from {previous_node} to node: {self.current_node}")
                return self.current_node
            else:
                # Handle terminal function calls (functions that don't lead to new nodes)
                logger.info(f"Executed terminal function: {function_name}")
                return self.current_node
        return None

    def get_current_node(self) -> str:
        """Get the current node ID."""
        return self.current_node
