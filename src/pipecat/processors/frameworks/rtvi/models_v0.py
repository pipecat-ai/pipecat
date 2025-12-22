#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI pre-1.0 protocol models (deprecated).

All classes here are kept for backward compatibility only. Pipeline configuration
and the actions API were removed in RTVI protocol 1.0.0. Use custom client and
server messages instead.
"""

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import BaseModel, Field, PrivateAttr

import pipecat.processors.frameworks.rtvi.models_v1 as RTVI

ActionResult = Union[bool, int, float, str, list, dict]


class RTVIServiceOption(BaseModel):
    """Configuration option for an RTVI service.

    Defines a configurable option that can be set for an RTVI service,
    including its name, type, and handler function.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    type: Literal["bool", "number", "string", "array", "object"]
    handler: Callable[["RTVIProcessor", str, "RTVIServiceOptionConfig"], Awaitable[None]] = Field(
        exclude=True
    )


class RTVIService(BaseModel):
    """An RTVI service definition.

    Represents a service that can be configured and used within the RTVI protocol,
    containing a name and list of configurable options.

    .. deprecated:: 0.0.75
       Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    name: str
    options: List[RTVIServiceOption]
    _options_dict: Dict[str, RTVIServiceOption] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        """Initialize the options dictionary after model creation."""
        self._options_dict = {}
        for option in self.options:
            self._options_dict[option.name] = option
        return super().model_post_init(__context)


class RTVIActionArgumentData(BaseModel):
    """Data for an RTVI action argument.

    Contains the name and value of an argument passed to an RTVI action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    value: Any


class RTVIActionArgument(BaseModel):
    """Definition of an RTVI action argument.

    Specifies the name and expected type of an argument for an RTVI action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    type: Literal["bool", "number", "string", "array", "object"]


class RTVIAction(BaseModel):
    """An RTVI action definition.

    Represents an action that can be executed within the RTVI protocol,
    including its service, name, arguments, and handler function.

    .. deprecated:: 0.0.75
       Actions have been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    service: str
    action: str
    arguments: List[RTVIActionArgument] = Field(default_factory=list)
    result: Literal["bool", "number", "string", "array", "object"]
    handler: Callable[["RTVIProcessor", str, Dict[str, Any]], Awaitable[ActionResult]] = Field(
        exclude=True
    )
    _arguments_dict: Dict[str, RTVIActionArgument] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        """Initialize the arguments dictionary after model creation."""
        self._arguments_dict = {}
        for arg in self.arguments:
            self._arguments_dict[arg.name] = arg
        return super().model_post_init(__context)


class RTVIServiceOptionConfig(BaseModel):
    """Configuration value for an RTVI service option.

    Contains the name and value to set for a specific service option.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    value: Any


class RTVIServiceConfig(BaseModel):
    """Configuration for an RTVI service.

    Contains the service name and list of option configurations to apply.

    .. deprecated:: 0.0.75
       Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    service: str
    options: List[RTVIServiceOptionConfig]


class RTVIConfig(BaseModel):
    """Complete RTVI configuration.

    Contains the full configuration for all RTVI services.

    .. deprecated:: 0.0.75
       Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    config: List[RTVIServiceConfig]


#
# Client -> Pipecat messages.
#


class RTVIUpdateConfig(BaseModel):
    """Request to update RTVI configuration.

    Contains new configuration settings and whether to interrupt the bot.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    config: List[RTVIServiceConfig]
    interrupt: bool = False


class RTVIActionRunArgument(BaseModel):
    """Argument for running an RTVI action.

    Contains the name and value of an argument to pass to an action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    name: str
    value: Any


class RTVIActionRun(BaseModel):
    """Request to run an RTVI action.

    Contains the service, action name, and optional arguments.

    .. deprecated:: 0.0.75
       Actions have been removed as part of the RTVI protocol 1.0.0.
       Use custom client and server messages instead.
    """

    service: str
    action: str
    arguments: Optional[List[RTVIActionRunArgument]] = None


#
# Pipecat -> Client responses and messages.
#


class RTVIBotReadyDataDeprecated(RTVI.BotReadyData):
    """Data for bot ready notification.

    Contains protocol version and initial configuration.
    """

    # The config field is deprecated and will not be included if
    # the client's rtvi version is 1.0.0 or higher.
    config: Optional[List[RTVIServiceConfig]] = None


class RTVIDescribeConfigData(BaseModel):
    """Data for describing available RTVI configuration.

    Contains the list of available services and their options.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    config: List[RTVIService]


class RTVIDescribeConfig(BaseModel):
    """Message describing available RTVI configuration.

    Sent in response to a describe-config request.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    label: RTVI.MessageLiteral = RTVI.MESSAGE_LABEL
    type: Literal["config-available"] = "config-available"
    id: str
    data: RTVIDescribeConfigData


class RTVIDescribeActionsData(BaseModel):
    """Data for describing available RTVI actions.

    Contains the list of available actions that can be executed.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    actions: List[RTVIAction]


class RTVIDescribeActions(BaseModel):
    """Message describing available RTVI actions.

    Sent in response to a describe-actions request.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    label: RTVI.MessageLiteral = RTVI.MESSAGE_LABEL
    type: Literal["actions-available"] = "actions-available"
    id: str
    data: RTVIDescribeActionsData


class RTVIConfigResponse(BaseModel):
    """Response containing current RTVI configuration.

    Sent in response to a get-config request.

    .. deprecated:: 0.0.75
        Pipeline Configuration has been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    label: RTVI.MessageLiteral = RTVI.MESSAGE_LABEL
    type: Literal["config"] = "config"
    id: str
    data: RTVIConfig


class RTVIActionResponseData(BaseModel):
    """Data for an RTVI action response.

    Contains the result of executing an action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    result: ActionResult


class RTVIActionResponse(BaseModel):
    """Response to an RTVI action execution.

    Sent after successfully executing an action.

    .. deprecated:: 0.0.75
        Actions have been removed as part of the RTVI protocol 1.0.0.
        Use custom client and server messages instead.
    """

    label: RTVI.MessageLiteral = RTVI.MESSAGE_LABEL
    type: Literal["action-response"] = "action-response"
    id: str
    data: RTVIActionResponseData
