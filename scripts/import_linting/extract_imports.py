"""Extract all imports from Python files in a source code tree."""

import ast
import re
import sys
from pathlib import Path


def get_import_paths_from_py_file(file_path: Path) -> set[str]:
    """Find all unique Python import paths in a Python source code file."""
    content = file_path.read_text(encoding="utf-8")
    import_paths: set[str] = set()
    # AST parsing for code blocks
    try:
        tree = ast.parse(content, filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("pipecat"):
                        import_paths.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("pipecat"):
                    module_name = node.module
                    for alias in node.names:
                        import_paths.add(f"{module_name}.{alias.name}")
    except Exception as e:
        print(f"Error parsing {file_path} with AST: {e}", file=sys.stderr)

    # Also search for imports in docstrings and comments via regex
    import_paths.update(get_import_paths_from_text(content))
    return import_paths


def get_import_paths_from_text_file(file_path: Path) -> set[str]:
    """Find and return all unique Python import paths found in a text file."""
    content = file_path.read_text(encoding="utf-8")
    return get_import_paths_from_text(content)


def get_import_paths_from_text(content: str) -> set[str]:
    """Find and return all unique Python import paths found in a string."""
    import_paths: set[str] = set()
    # Regex for multi-line `from ... import (...)`
    multiline_pattern = re.compile(
        r"from\s+(pipecat[\w\.]*)\s+import\s+\(([\s\S]+?)\)", re.MULTILINE
    )
    for match in multiline_pattern.finditer(content):
        module, names_str = match.groups()
        names = [
            name.split(" as ")[0].strip()
            for name in names_str.replace("\n", " ").split(",")
            if name.strip() and not name.strip().startswith("#")
        ]
        for name in names:
            import_paths.add(f"{module}.{name}")

    # Regex for single-line `from ... import ...`
    singleline_from_pattern = re.compile(
        r"from\s+(pipecat[\w\.]*)\s+import\s+([^\(\n].*)", re.MULTILINE
    )
    for match in singleline_from_pattern.finditer(content):
        module, names_str = match.groups()
        names = [
            name.split(" as ")[0].strip()
            for name in names_str.split(",")
            if name.strip() and not name.strip().startswith("#")
        ]
        for name in names:
            import_paths.add(f"{module}.{name}")

    # Regex for `import pipecat.x`
    import_pattern = re.compile(r"import\s+(pipecat[\w\.]*)", re.MULTILINE)
    for match in import_pattern.finditer(content):
        full_import = match.group(1)
        name = full_import.split(" as ")[0].strip()
        import_paths.add(name)

    return import_paths


if __name__ == "__main__":
    all_import_paths: set[str] = set()
    for file in Path(".").rglob("*"):
        if not file.is_file():
            continue
        if any(
            parent.name.startswith(".")
            or parent.name in {"site-packages", "import_linting"}
            for parent in file.parents
        ):
            continue
        if file.suffix == ".py":
            all_import_paths.update(get_import_paths_from_py_file(file))
        elif file.suffix in (".rst", ".md"):
            all_import_paths.update(get_import_paths_from_text_file(file))

    for path in sorted(all_import_paths):
        print(path)
