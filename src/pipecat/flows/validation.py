#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Validation reports for flow configs.

:func:`validate_flow` runs the checks the runtime runs, loading the config
and constructing a :class:`~pipecat.flows.Flow`, and collects their errors
into a :class:`FlowReport`. It adds graph warnings that a valid config can
still deserve: nodes the flow can never reach, nodes it can never leave, and
branches whose cases and default all go one place.

The ``pipecat flows validate`` command is a thin wrapper over this module;
any other tool that wants to check a config can call it directly.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from pipecat.flows.config import FlowConfig
from pipecat.flows.exceptions import FlowReferenceError
from pipecat.flows.flow import _VARIABLE, Flow

IssueLevel = Literal["error", "warning"]


@dataclass
class FlowIssue:
    """One problem found in a flow config.

    Parameters:
        level: ``error`` for a config the runtime would reject; ``warning``
            for a valid config that probably does not do what its author
            meant.
        code: Stable identifier for the kind of problem, for tooling.
        message: Human-readable description naming the node, function, or
            field involved.
        node: The node the issue is about, when there is one.
        function: The function entry the issue is about, when there is one.
    """

    level: IssueLevel
    code: str
    message: str
    node: str | None = None
    function: str | None = None


@dataclass
class FlowReport:
    """The result of :func:`validate_flow`.

    Parameters:
        issues: Every error and warning found, in the order found.
        tools: Names of every tool the config references, including action
            handlers.
        variables: Names of every ``{{ variable }}`` the config uses.
        config: The parsed config, or ``None`` when it failed to load.
    """

    issues: list[FlowIssue] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    config: FlowConfig | None = None

    @property
    def errors(self) -> list[FlowIssue]:
        """The issues the runtime would reject the config for."""
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[FlowIssue]:
        """The issues that do not stop the config from loading."""
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        """Whether the config has no errors. Warnings do not affect this."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable form of the report, without the parsed config."""
        return {
            "ok": self.ok,
            "issues": [asdict(i) for i in self.issues],
            "tools": list(self.tools),
            "variables": list(self.variables),
        }


def validate_flow(
    source: str | Path | Mapping[str, Any] | FlowConfig,
    *,
    tools: Mapping[str, Callable] | Any | None = None,
    variables: Mapping[str, Any] | None = None,
    base_dir: Path | None = None,
) -> FlowReport:
    """Check a flow config and report everything wrong with it.

    Loading the config reports parse and schema errors. When ``tools`` is
    given, constructing a :class:`~pipecat.flows.Flow` reports every tool,
    handler, and variable reference it cannot resolve. When only
    ``variables`` is given, the variables are checked on their own. Graph
    warnings come from walking the config.

    Args:
        source: A ``Path`` to a YAML or JSON file, YAML text, a mapping, or an
            already-loaded :class:`~pipecat.flows.FlowConfig`.
        tools: Where tool and handler names resolve, as for
            :class:`~pipecat.flows.Flow`. When omitted, references are listed
            in the report but not checked.
        variables: Values for the config's placeholders. When omitted, the
            placeholders are listed in the report but not checked.
        base_dir: Directory that ``!include`` paths resolve against when
            ``source`` is YAML text.

    Returns:
        The report. ``report.ok`` is False when any error was found.
    """
    report = FlowReport()

    config = _load(source, base_dir, report)
    if config is None:
        return report
    report.config = config
    report.tools = _referenced_tools(config)
    report.variables = _used_variables(config)

    _check_graph(config, report)

    if tools is not None:
        # Placeholders stand in for variables not supplied, so the construction
        # checks the tool references without also failing on every variable.
        stand_ins = {name: "{{ " + name + " }}" for name in report.variables}
        try:
            Flow(config, tools=tools, variables={**stand_ins, **dict(variables or {})})
        except FlowReferenceError as e:
            for problem in e.problems:
                report.issues.append(FlowIssue("error", **asdict(problem)))
    if variables is not None:
        for name in report.variables:
            if name not in variables:
                report.issues.append(
                    FlowIssue("error", "missing_variable", f"variable '{name}' has no value")
                )

    return report


def _load(
    source: str | Path | Mapping[str, Any] | FlowConfig, base_dir: Path | None, report: FlowReport
) -> FlowConfig | None:
    try:
        if isinstance(source, FlowConfig):
            return source
        if isinstance(source, Path):
            return FlowConfig.from_file(source)
        if isinstance(source, Mapping):
            return FlowConfig.model_validate(source)
        return FlowConfig.from_yaml(source, base_dir=base_dir)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            where = f"{loc}: " if loc else ""
            msg = err["msg"].removeprefix("Value error, ")
            report.issues.append(FlowIssue("error", "schema", f"{where}{msg}"))
    except yaml.YAMLError as e:
        report.issues.append(FlowIssue("error", "parse", f"invalid YAML: {e}"))
    except (OSError, ValueError) as e:
        report.issues.append(FlowIssue("error", "load", str(e)))
    return None


def _referenced_tools(config: FlowConfig) -> list[str]:
    names: list[str] = []
    for node in config.nodes.values():
        names.extend(f.name for f in node.functions)
        for action in node.pre_actions + node.post_actions:
            if action.handler:
                names.append(action.handler)
    names.extend(f.name for f in config.global_functions)
    return sorted(set(names))


def _used_variables(config: FlowConfig) -> list[str]:
    found: set[str] = set()
    for node in config.nodes.values():
        texts = [m.content for m in node.task_messages]
        if node.role_message:
            texts.append(node.role_message)
        for action in node.pre_actions + node.post_actions:
            text = action.extras().get("text")
            if isinstance(text, str):
                texts.append(text)
        for text in texts:
            found.update(m.group(1) for m in _VARIABLE.finditer(text))
    return sorted(found)


def _check_graph(config: FlowConfig, report: FlowReport) -> None:
    global_targets = {t for f in config.global_functions for t in f.targets()}

    # Every node a function at ``name`` can lead to, counting global functions.
    def exits(name: str) -> set[str]:
        node = config.nodes[name]
        return {t for f in node.functions for t in f.targets()} | global_targets

    reachable = {config.initial_node}
    frontier = [config.initial_node]
    while frontier:
        for target in exits(frontier.pop()):
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    for name in config.nodes:
        if name not in reachable:
            report.issues.append(
                FlowIssue(
                    "warning",
                    "unreachable_node",
                    f"node '{name}' cannot be reached from '{config.initial_node}'",
                    node=name,
                )
            )

    for name, node in config.nodes.items():
        ends = any(a.type == "end_conversation" for a in node.post_actions)
        if not ends and not (exits(name) - {name}):
            report.issues.append(
                FlowIssue(
                    "warning",
                    "dead_end",
                    f"node '{name}' has no function that leaves it "
                    "and does not end the conversation",
                    node=name,
                )
            )

    for name, node in config.nodes.items():
        for func in node.functions:
            branch = func.transition_to
            if isinstance(branch, str) or branch is None:
                continue
            # Without a default, an unmatched value stays on the node, so the
            # branch has two outcomes even when every case names one node.
            if branch.default is None:
                continue
            targets = set(branch.targets())
            if len(targets) == 1:
                (only,) = targets
                report.issues.append(
                    FlowIssue(
                        "warning",
                        "branch_single_target",
                        f"node '{name}' function '{func.name}' branches on "
                        f"'{branch.field}' but every case and the default lead to '{only}'",
                        node=name,
                        function=func.name,
                    )
                )


__all__ = ["FlowIssue", "FlowReport", "validate_flow"]
