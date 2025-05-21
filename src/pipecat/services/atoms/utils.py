from typing import Dict, List


def convert_old_to_new_format(old_format_data: Dict) -> List:
    """
    Convert the old workflow graph format to the new format.

    Args:
        old_format_data (dict): Original workflow graph data

    Returns:
        list: Converted workflow graph data
    """
    new_format = []

    # Create edge mapping
    edge_mapping = {}
    for edge in old_format_data["edges"]:
        source_id = edge["source"]
        if source_id not in edge_mapping:
            edge_mapping[source_id] = []

        # Include the new edge attributes if present
        pathway = {
            "id": edge["id"],
            "condition": edge["label"],
            "target_node_id": edge["target"],
            "description": edge.get("data", {}).get("description"),
            "is_conditional_edge": edge.get("data", {}).get("is_conditional_edge", False),
            "is_fallback_edge": edge.get("data", {}).get("is_fallback_edge", False),
        }

        edge_mapping[source_id].append(pathway)

    # Convert nodes
    for node in old_format_data["nodes"]:
        node_type = node["type"]

        # Basic node data common to all types
        new_node = {
            "id": node["id"],
            "name": node["data"].get("label"),
            "type": node_type,
            "action": node["data"].get("action"),
            "static_text": node["data"].get("staticText", False),
            "loop_condition": node["data"].get("loopCondition"),
            "global_node": node["data"].get("globalNode"),
            "global_node_label": node["data"].get("globalNodeLabel"),
            "global_node_desc": node["data"].get("globalNodeDescription"),
            "is_start_node": node["data"].get("isStartNode", False),
            "knowledge_base": node["data"].get("knowledgeBase"),
            "use_global_knowledge_base": node["data"].get("useGlobalKnowledgeBase", False),
            "transfer_number": node["data"].get("transferNumber"),
            "use_gpt": node["data"].get("useGPT", False),
            "pathways": edge_mapping.get(node["id"], []),
        }

        # Handle variables conversion
        variables_data = node["data"].get("variables", {"isEnabled": False}) or {"isEnabled": False}
        if variables_data.get("isEnabled", False):
            variables_data = variables_data.get("data", [])
            new_node["variables"] = {"is_enabled": True, "data": variables_data}
        else:
            new_node["variables"] = {"is_enabled": False, "data": []}

        # Handle API Call node types
        if node_type in ["api_call", "webhook", "pre_call_api", "post_call_api"]:
            http_request_data = node["data"].get("httpRequest", {})

            print("http_request_data", http_request_data)

            if http_request_data:
                headers = http_request_data.get("headers", {"isEnabled": False})
                if headers.get("isEnabled", False):
                    headers_data = headers.get("data", {})
                    headers = {"is_enabled": True, "data": headers_data}
                else:
                    headers = {"is_enabled": False, "data": {}}

                authorization = http_request_data.get("authorization", {"isEnabled": False})
                if authorization.get("isEnabled", False):
                    authorization_data = authorization.get("data", {})
                    authorization = {"is_enabled": True, "data": authorization_data}
                else:
                    authorization = {"is_enabled": False, "data": None}

                body = http_request_data.get("body", {"isEnabled": False})
                if body.get("isEnabled", False):
                    body_data = body.get("data", {})
                    body = {"is_enabled": True, "data": body_data}
                else:
                    body = {"is_enabled": False, "data": None}

                http_request = {
                    "method": http_request_data.get("method"),
                    "url": http_request_data.get("url"),
                    "timeout": http_request_data.get("timeout", 10),
                    "headers": headers,
                    "authorization": authorization,
                    "body": body,
                }
                new_node["http_request"] = http_request
            else:
                raise RuntimeError("HTTP request data is required for API nodes")

            # Handle response data mapping
            response_mapping = node["data"].get("responseData", {"isEnabled": False}) or {
                "isEnabled": False
            }
            if response_mapping.get("isEnabled", False):
                response_data = response_mapping.get("data", [])
                new_node["response_data"] = {"is_enabled": True, "data": response_data}
            else:
                new_node["response_data"] = {"is_enabled": False, "data": []}

        new_format.append(new_node)

    return new_format


def get_abbreviations() -> set:
    return {
        # Original abbreviations
        "Mr.",
        "Mrs.",
        "Dr.",
        "Ms.",
        "Prof.",
        "Inc.",
        "Ltd.",
        "St.",
        "etc.",
        "e.g.",
        "i.e.",
        # Academic degrees
        "B.",
        "B.A.",
        "B.S.",
        "B.Sc.",
        "B.E.",
        "B.Tech.",
        "B.Ed.",
        "B.Com.",
        "B.B.A.",
        "M.",
        "M.A.",
        "M.S.",
        "M.Sc.",
        "M.E.",
        "M.Tech.",
        "M.Ed.",
        "M.Com.",
        "M.B.A.",
        "M.Phil.",
        "Ph.D.",
        "D.Phil.",
        "LL.B.",
        "LL.M.",
        "J.D.",
        "M.D.",
        # Titles and honorifics
        "Capt.",
        "Col.",
        "Gen.",
        "Lt.",
        "Sgt.",
        "Cmdr.",
        "Adm.",
        "Rev.",
        "Hon.",
        "Sr.",
        "Jr.",
        "Esq.",
        # Common abbreviations in names/places
        "St.",
        "Ave.",
        "Rd.",
        "Blvd.",
        "Dr.",
        "Ct.",
        "Pl.",
        "Ln.",
        "Hwy.",
        # Business abbreviations
        "Co.",
        "Corp.",
        "LLC.",
        "L.L.C.",
        "LLP.",
        "L.L.P.",
        "Assn.",
        "Bros.",
        "Dept.",
        # Time and date abbreviations
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sept.",
        "Oct.",
        "Nov.",
        "Dec.",
        "Mon.",
        "Tue.",
        "Wed.",
        "Thu.",
        "Fri.",
        "Sat.",
        "Sun.",
        "a.m.",
        "p.m.",
        # Geographical abbreviations
        "U.S.",
        "U.S.A.",
        "U.K.",
        "E.U.",
        "N.A.",
        "S.A.",
        "N.Z.",
        # Common Latin abbreviations
        " vs.",
        "etc.",
        " et al.",
        " ibid.",
        " op. cit.",
        " viz.",
        " cf.",
        # Measurement abbreviations
        " ft.",
        " in.",
        " lbs.",
        " oz.",
        " pt.",
        " qt.",
        " gal.",
        # Other common abbreviations
        " fig.",
        " p.",
        " pp.",
        "Vol.",
        "Ch.",
        " approx.",
        " est.",
        # Numbers followed by periods (ordinals)
        "1.",
        "2.",
        "3.",
        "4.",
        "5.",
        "6.",
        "7.",
        "8.",
        "9.",
        "0.",
    }


def get_unallowed_variable_names() -> list:
    return ["call_id", "user_number", "agent_number"]
