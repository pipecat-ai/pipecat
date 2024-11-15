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
    """Configuration for a single node in the conversation flow"""

    message: dict
    functions: List[dict]
    actions: Optional[List[dict]] = None


class ConversationFlow:
    """Manages the state and transitions of the conversation flow"""

    def __init__(self, flow_config: dict):
        self.nodes: Dict[str, NodeConfig] = {}
        self.current_node: str = flow_config["initial_node"]
        self._load_config(flow_config)

    def _load_config(self, config: dict):
        """Load and validate the flow configuration"""
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
        """Get the message for the current node"""
        return self.nodes[self.current_node].message

    def get_current_functions(self) -> List[dict]:
        """Get the available functions for the current node"""
        return self.nodes[self.current_node].functions

    def get_current_actions(self) -> Optional[List[dict]]:
        """Get the actions for the current node"""
        return self.nodes[self.current_node].actions

    def get_available_function_names(self) -> Set[str]:
        """Get the names of available functions for the current node"""
        names = {f["function"]["name"] for f in self.nodes[self.current_node].functions}
        logger.debug(f"Available function names for node {self.current_node}: {names}")
        return names

    def transition(self, function_name: str) -> Optional[str]:
        """Attempt to transition based on function call"""
        available_functions = self.get_available_function_names()
        logger.info(f"Attempting transition with {function_name}")
        logger.info(f"Current node: {self.current_node}")
        logger.info(f"Available functions: {available_functions}")

        if function_name in available_functions:
            if function_name in self.nodes:
                self.current_node = function_name
                logger.info(f"Transitioned to node: {self.current_node}")
                return self.current_node
            else:
                # Handle terminal function calls
                logger.info(f"Executed terminal function: {function_name}")
                # Optionally, could transition to a "completion" node or stay in current node
                return self.current_node
        return None
