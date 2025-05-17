import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .utils import convert_old_to_new_format


class NodeType(str, Enum):
    DEFAULT = "default"
    API_CALL = "api_call"
    PRE_CALL_API = "pre_call_api"
    POST_CALL_API = "post_call_api"
    KNOWLEDGE_BASE = "knowledge_base"
    END_CALL = "end_call"
    TRANSFER_CALL = "transfer_call"


class ApiCallRequestType(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class VariableDefinition(BaseModel):
    name: str
    type: Literal["string", "integer", "boolean"]
    description: str

    @field_validator("type", mode="before")
    @classmethod
    def lowercase_type(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v


class VariablesConfig(BaseModel):
    is_enabled: bool = False
    data: List[VariableDefinition] = Field(default_factory=list)


class ResponseDataMapping(BaseModel):
    variable_name: str = Field(alias="variableName")
    json_path: str = Field(alias="jsonPath")


class ResponseDataConfig(BaseModel):
    is_enabled: bool = False
    data: List[ResponseDataMapping] = Field(default=[])


class ApiCallAuth(BaseModel):
    type: Optional[str] = None
    token: Optional[str] = None


class HttpHeaders(BaseModel):
    is_enabled: bool = False
    data: Dict[str, str] = Field(default_factory=dict)


class HttpAuthorization(BaseModel):
    is_enabled: bool = False
    data: Optional[ApiCallAuth] = None


class HttpBody(BaseModel):
    is_enabled: bool = False
    data: Optional[str] = None


class HttpRequest(BaseModel):
    method: ApiCallRequestType
    url: str
    headers: HttpHeaders = Field(default_factory=HttpHeaders)
    authorization: HttpAuthorization = Field(default_factory=HttpAuthorization)
    body: HttpBody = Field(default_factory=HttpBody)
    timeout: int = 10


class Pathway(BaseModel):
    id: str
    target_node_id: str
    target_node: Optional[Any] = None
    condition: str
    description: Optional[str] = None
    is_conditional_edge: bool = False
    is_fallback_edge: bool = False


class Node(BaseModel):
    id: str
    name: str
    type: NodeType
    action: str
    static_text: bool = False
    loop_condition: Optional[str] = None
    global_node: Optional[bool] = None
    global_node_label: Optional[str] = None
    global_node_desc: Optional[str] = None
    variables: VariablesConfig = Field(default_factory=VariablesConfig)
    is_start_node: bool = False
    http_request: Optional[HttpRequest] = None
    response_data: ResponseDataConfig = Field(default_factory=ResponseDataConfig)
    knowledge_base: Optional[str] = None
    use_global_knowledge_base: Optional[bool] = None
    transfer_number: Optional[str] = None
    use_gpt: bool = False
    pathways: Dict[str, Pathway] = Field(default_factory=dict)

    # Other attributes
    knowledge_base_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def add_pathway(
        self,
        pathway_id: str,
        target_node: "Node",
        condition: str,
        description: Optional[str] = None,
        is_conditional_edge: bool = False,
        is_fallback_edge: bool = False,
    ):
        pathway = Pathway(
            id=pathway_id,
            target_node_id=target_node.id,
            target_node=target_node,
            condition=condition,
            description=description,
            is_conditional_edge=is_conditional_edge,
            is_fallback_edge=is_fallback_edge,
        )
        self.pathways[pathway_id] = pathway


class ConversationalPathway(BaseModel):
    name: Optional[str] = None
    nodes: Dict[str, Node] = Field(default_factory=dict)
    start_node: Optional[Node] = None

    class Config:
        arbitrary_types_allowed = True

    def set_graph_name(self, name: str):
        self.name = name

    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: NodeType,
        action: str,
        static_text: bool = False,
        loop_condition: Optional[str] = None,
        global_node: Optional[bool] = None,
        global_node_label: Optional[str] = None,
        global_node_desc: Optional[str] = None,
        variables: Optional[Union[VariablesConfig, dict]] = None,
        is_start_node: bool = False,
        http_request: Optional[Union[HttpRequest, dict]] = None,
        response_data: Optional[Union[ResponseDataConfig, dict]] = None,
        knowledge_base: Optional[str] = None,
        use_global_knowledge_base: Optional[bool] = None,
        transfer_number: Optional[str] = None,
        use_gpt: bool = False,
    ) -> Node:
        # Convert dict to VariablesConfig if needed
        if isinstance(variables, dict):
            variables = VariablesConfig.model_validate(variables)
        elif variables is None:
            variables = VariablesConfig()

        # Convert dict to HttpRequest if needed
        if isinstance(http_request, dict):
            http_request = HttpRequest.model_validate(http_request)

        # Convert dict to ResponseDataConfig if needed
        if isinstance(response_data, dict):
            response_data = ResponseDataConfig.model_validate(response_data)
        elif response_data is None:
            response_data = ResponseDataConfig()

        node = Node(
            id=node_id,
            name=name,
            type=node_type,
            action=action,
            static_text=static_text,
            loop_condition=loop_condition,
            global_node=global_node,
            global_node_label=global_node_label,
            global_node_desc=global_node_desc,
            variables=variables,
            is_start_node=is_start_node,
            http_request=http_request,
            response_data=response_data,
            knowledge_base=knowledge_base,
            use_global_knowledge_base=use_global_knowledge_base,
            transfer_number=transfer_number,
            use_gpt=use_gpt,
        )
        self.nodes[node_id] = node
        if is_start_node:
            self.start_node = node
        return node

    def add_pathway(
        self,
        pathway_id: str,
        from_node_id: str,
        to_node_id: str,
        condition: str,
        description: Optional[str] = None,
        is_conditional_edge: bool = False,
        is_fallback_edge: bool = False,
    ):
        self.nodes[from_node_id].add_pathway(
            pathway_id=pathway_id,
            target_node=self.nodes[to_node_id],
            condition=condition,
            description=description,
            is_conditional_edge=is_conditional_edge,
            is_fallback_edge=is_fallback_edge,
        )

    def build_from_json(self, data: Union[str, dict, list]):
        """
        Builds the conversation flow from JSON data with validation.

        Args:
            data: JSON string or dict containing the conversation flow data

        Raises:
            ValidationError: If any validation checks fail
            ValueError: If data format is invalid
        """
        # Handle input data format
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON format")

        if isinstance(data, dict):
            if "nodes" in data and "edges" in data:
                nodes = convert_old_to_new_format(data)
            elif "nodes" in data:
                nodes = data["nodes"]
            else:
                raise ValidationError("Invalid JSON format")
        elif isinstance(data, list):
            nodes = data
        else:
            raise ValidationError("Invalid JSON format")

        # Validation tracking
        node_ids = set()
        start_nodes_count = 0
        referenced_target_nodes = set()

        # First pass: Collect metadata
        for node in nodes:
            # Track node IDs and start nodes
            node_id = node.get("id")
            if node_id in node_ids:
                raise ValidationError(f"Duplicate node ID found: {node_id}")
            node_ids.add(node_id)

            if node.get("is_start_node", False):
                start_nodes_count += 1

            # Track referenced nodes
            for pathway in node.get("pathways", []):
                referenced_target_nodes.add(pathway["target_node_id"])

        # Validate overall flow structure
        if start_nodes_count == 0:
            raise ValidationError("No start node found in conversation flow")
        elif start_nodes_count > 1:
            raise ValidationError("Multiple start nodes found in conversation flow")

        missing_nodes = referenced_target_nodes - node_ids
        if missing_nodes:
            raise ValidationError(f"Referenced target nodes not found: {missing_nodes}")

        # Second pass: Build the flow
        for node_data in nodes:
            node_id: str = node_data["id"]
            node_type_str: str = node_data["type"]

            if node_type_str.endswith("_node"):
                node_type_str = node_type_str.replace("_node", "")

            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                raise ValidationError(f"Invalid node type: {node_type_str}")

            # Process variables
            variables = {"is_enabled": False, "data": []}

            # Handle new format
            if node_data.get("variables"):
                variables = node_data["variables"]

            # Process HTTP request for API call nodes
            http_request = None
            response_data = None

            if node_type in [NodeType.API_CALL, NodeType.PRE_CALL_API, NodeType.POST_CALL_API]:
                http_request = node_data["http_request"]

                # Process response data mapping
                if "response_data" in node_data:
                    response_data = node_data["response_data"]

            # Extract all attributes from the node data
            attributes = {
                "node_id": node_id,
                "name": node_data.get("name"),
                "node_type": node_type,
                "action": node_data.get("action"),
                "static_text": node_data.get("static_text", False),
                "loop_condition": node_data.get("loop_condition"),
                "global_node": node_data.get("global_node"),
                "global_node_label": node_data.get("global_node_label"),
                "global_node_desc": node_data.get("global_node_desc"),
                "variables": variables,
                "is_start_node": node_data.get("is_start_node", False),
                "http_request": http_request,
                "response_data": response_data,
                "knowledge_base": node_data.get("knowledge_base"),
                "use_global_knowledge_base": node_data.get("use_global_knowledge_base"),
                "transfer_number": node_data.get("transfer_number"),
                "use_gpt": node_data.get("use_gpt", False),
            }

            # Add the node to the pathway
            self.add_node(**attributes)

        # Add pathways after all nodes are created
        for node_data in nodes:
            for pathway_data in node_data.get("pathways", []):
                self.add_pathway(
                    pathway_data["id"],
                    from_node_id=node_data["id"],
                    to_node_id=pathway_data["target_node_id"],
                    condition=pathway_data["condition"],
                    description=pathway_data.get("description"),
                    is_conditional_edge=pathway_data.get("is_conditional_edge", False),
                    is_fallback_edge=pathway_data.get("is_fallback_edge", False),
                )

        return self


class ValidationError(Exception):
    """Custom exception for conversation flow validation errors."""

    pass
