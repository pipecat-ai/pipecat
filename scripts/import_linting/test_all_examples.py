"""Test imports of examples in the `examples/` directory.

This script automates the process of testing all examples in the `examples/`
directory to ensure they can be imported without dependency errors.

**Usage:**
`python test_all_examples.py [filter1] [filter2] ...`

The script accepts zero or more positional arguments, which act as substring
filters. If filters are provided, the script will only test files where the
relative path (from the project root) or the file's content matches at least
one of the provided filter strings. This is useful for quickly re-testing a
specific example after a fix.

**Goal:**
Ensure all Python files under the `examples/` directory can be imported
successfully without dependency errors.

**Workflow:**
1.  **Planning Phase:**
    a.  The script first traverses the entire `examples/` directory to build a
        comprehensive test plan.
    b.  It identifies all unique testing environments, defined by the presence
        of a `requirements.txt` file.
    c.  For each environment, it gathers all `*.py` files and categorizes them
        into two groups:
        - **New Files:** Files not found in `successful_imports.txt`.
        - **Successful Files:** Files that have been successfully imported in
          previous runs.

2.  **Execution Phase:**
    a.  The script executes the test plan in two distinct waves to prioritize
        finding new errors quickly ("fail-fast").
    b.  **Wave 1 (New Files):** It iterates through each environment and runs
        the import test only for the "new" files. If any import fails, the
        script exits immediately.
    c.  **Wave 2 (Successful Files):** If all new files pass, it proceeds to
        re-test the "successful" files to catch any regressions.
    d.  For each environment, it creates a dedicated virtual environment using
        `uv venv`, installs dependencies, and then runs the import tests.

3.  **Error Handling:**
    a.  If an import fails, the script prints the traceback and exits. The user
        is then expected to:
        - Analyze the traceback to identify the missing module.
        - Consult `pyproject.toml` to find the optional dependency ("extra")
          that provides the missing module (e.g., `[deepgram]`, `[silero]`).
        - Add the required extra to the `requirements.txt` file for the
          corresponding example group.
        - Re-run this script.

4.  **State Management:**
    a.  A list of successfully imported files is maintained in
        `successful_imports.txt`.

5.  **Dependency Mocking:**
    a.  Proprietary or OS-specific dependencies like `[krisp]` and `coremltools`
        cannot be installed in all environments.
    b.  To allow import tests to pass, this script dynamically creates mock
        modules for these dependencies.
    c.  When a mocked dependency is detected in a `requirements.txt` file, the
        script filters it out to prevent a failed installation and creates a
        dummy package structure in the virtual environment to satisfy imports.
"""

import argparse
import re
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import TypedDict, Literal


class TestPlan(TypedDict):
    new_files_to_test: list[Path]
    successful_files_to_retest: list[Path]


def run_command(
    command: list[str],
    check: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"Running: {' '.join(map(str, command))}")
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    result = subprocess.run(command, text=True, cwd=cwd, env=process_env)
    if check and result.returncode != 0:
        print(f"Error running command: {' '.join(map(str, command))}")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    return result


def get_python_executable(example_dir: Path) -> tuple[str, str]:
    python_version = "3.11"
    version_file = example_dir / ".python-version"
    if not version_file.exists():
        for parent in example_dir.parents:
            if (parent / ".python-version").exists():
                version_file = parent / ".python-version"
                break
    if version_file.exists():
        with open(version_file, "r") as f:
            python_version = f.read().strip()
        print(f"--- Found .python-version, using Python {python_version} ---")

    python_executable_name = f"python{python_version}"
    python_path = shutil.which(python_executable_name)
    if not python_path:
        print(f"!!! ERROR: {python_executable_name} not found in PATH !!!")
        sys.exit(1)

    return python_path, str(Path.cwd() / example_dir / ".venv/bin/python")


def setup_environment(example_dir: Path, build_env: dict[str, str]):
    print(f"--- Setting up environment for {example_dir} ---")
    python_path, python_executable = get_python_executable(example_dir)

    _ = run_command(
        ["uv", "venv", "--python", python_path, "--system-site-packages"], cwd=example_dir
    )

    print(f"--- Installing build tools for {example_dir} ---")
    _ = run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python_executable,
            "wheel",
            "setuptools",
            "meson-python",
            "pkgconfig",
            "ninja",
            "cython",
            "pycairo",
        ],
        cwd=example_dir,
        env=build_env,
    )

    # Parse requirements and install pipecat-ai with extras separately
    for requirements_path in example_dir.glob("requirements.txt"):
        print(f"--- Processing dependencies from {requirements_path} ---")

        with open(requirements_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        pipecat_extras = ""
        other_reqs: list[str] = []
        krisp_dependency_found = False
        coreml_dependency_found = False

        for line in lines:
            if line.startswith("pipecat-ai[") or line.startswith("-e"):
                match = re.search(r"\[(.*)\]", line)
                if match:
                    all_extras = match.group(1)
                    if "krisp" in all_extras:
                        krisp_dependency_found = True

                    # Filter out krisp from the extras to be installed
                    extras_list = [e.strip() for e in all_extras.split(',') if e.strip() != "krisp"]
                    pipecat_extras = ",".join(extras_list)
            else:
                # coremltools can't be installed on non-macOS, so we mock it.
                if "coremltools" in line:
                    coreml_dependency_found = True
                    continue
                other_reqs.append(line)

        # Install other dependencies first
        if other_reqs:
            temp_req_path = example_dir / "temp_requirements.txt"
            with open(temp_req_path, "w") as f:
                _ = f.write("\n".join(other_reqs))

            print(f"--- Installing non-pipecat dependencies for {example_dir} ---")
            _ = run_command(
                ["uv", "pip", "install", "--python", python_executable, "-r", str(temp_req_path)],
                cwd=Path("."),  # Run from project root
                env=build_env,
                check=False,
            )
            os.remove(temp_req_path)

        # Install pipecat-ai with extras in editable mode
        install_target = f".[{pipecat_extras}]" if pipecat_extras else "."
        print(f"--- Installing pipecat-ai for {example_dir} with extras: [{pipecat_extras}] ---")
        _ = run_command(
            ["uv", "pip", "install", "--python", python_executable, "-e", install_target],
            cwd=Path("."),  # Run from project root
            env=build_env,
        )

        # If krisp was in the original dependencies, create a mock module for it
        # so the import doesn't fail.
        if krisp_dependency_found:
            print("--- Creating mock krisp module ---")
            venv_path = Path(python_executable).parent.parent
            site_packages_list = list(venv_path.glob("lib/python*/site-packages"))
            if site_packages_list:
                site_packages_path = site_packages_list[0]
                krisp_mock_path = site_packages_path / "pipecat_ai_krisp"
                krisp_mock_path.mkdir(exist_ok=True)
                (krisp_mock_path / "__init__.py").touch()

                # Create the nested audio module
                audio_mock_path = krisp_mock_path / "audio"
                audio_mock_path.mkdir(exist_ok=True)
                (audio_mock_path / "__init__.py").touch()

                # Create a dummy krisp_processor.py with a mock class
                krisp_processor_content = "class KrispAudioProcessor:\n    pass\n"
                with open(audio_mock_path / "krisp_processor.py", "w") as f:
                    _ = f.write(krisp_processor_content)

                print(f"--- Mock krisp module created at {krisp_mock_path} ---")

        # If coremltools was in the original dependencies, create a mock module for it
        # so the import doesn't fail.
        if coreml_dependency_found:
            print("--- Creating mock coremltools module ---")
            venv_path = Path(python_executable).parent.parent
            site_packages_list = list(venv_path.glob("lib/python*/site-packages"))
            if site_packages_list:
                site_packages_path = site_packages_list[0]
                coreml_mock_path = site_packages_path / "coremltools"
                coreml_mock_path.mkdir(exist_ok=True)
                (coreml_mock_path / "__init__.py").touch()
                print(f"--- Mock coremltools module created at {coreml_mock_path} ---")

        # Install pipecat-ai-small-webrtc-prebuilt if it's in the requirements
        if "pipecat-ai-small-webrtc-prebuilt" in other_reqs:
            print("--- Installing pipecat-ai-small-webrtc-prebuilt ---")
            _ = run_command(
                ["uv", "pip", "install", "--python", python_executable, "pipecat-ai-small-webrtc-prebuilt"],
                cwd=Path("."),  # Run from project root
                env=build_env,
            )
            other_reqs.remove("pipecat-ai-small-webrtc-prebuilt")

    # Install local setup.py dependencies (if any)
    for setup_path in example_dir.rglob("setup.py"):
        if ".venv" in str(setup_path):
            continue
        print(f"--- Installing local dependency from {setup_path.parent} ---")
        _ = run_command(
            ["uv", "pip", "install", "--python", python_executable, "-e", "."],
            cwd=setup_path.parent,
            env=build_env,
        )


def plan_tests(
    base_dir: Path, successful_imports: set[str], filters: list[str]
) -> dict[Path, TestPlan]:
    test_plan: dict[Path, TestPlan] = {}
    example_dirs_with_reqs = {req.parent for req in base_dir.rglob("requirements.txt")}

    for example_dir in sorted(list(example_dirs_with_reqs)):
        if example_dir.name in [
            "moondream-chatbot",
            "studypal",
            "local-input-select-stt",
            "p2p-webrtc",
        ]:
            continue

        test_plan[example_dir] = {"new_files_to_test": [], "successful_files_to_retest": []}

        for file_path in example_dir.rglob("*.py"):
            if ".venv" in str(file_path):
                continue

            # Ensure the file belongs to the current example_dir context, not a sub-context
            in_sub_context = False
            for parent in file_path.parents:
                if parent == example_dir:
                    break
                if parent in example_dirs_with_reqs:
                    in_sub_context = True
                    break
            if in_sub_context:
                continue

            if filters:
                file_content = file_path.read_text()
                if not any(f in str(file_path) or f in file_content for f in filters):
                    continue

            if str(file_path) in successful_imports:
                test_plan[example_dir]["successful_files_to_retest"].append(file_path)
            else:
                test_plan[example_dir]["new_files_to_test"].append(file_path)
    return test_plan


def execute_test_wave(
    wave_name: str,
    test_plan: dict[Path, TestPlan],
    file_category: Literal["new_files_to_test", "successful_files_to_retest"],
    build_env: dict[str, str],
    successful_imports: set[str],
):
    print(f"\n{'=' * 20} EXECUTING {wave_name.upper()} WAVE {'=' * 20}")
    for example_dir, plan in test_plan.items():
        files_to_test = plan[file_category]
        if not files_to_test:
            continue

        setup_environment(example_dir, build_env)
        _, python_executable = get_python_executable(example_dir)

        args: list[str] = []
        for file_path in files_to_test:
            module_name = file_path.stem.replace("-", "_")
            args.extend([module_name, str(file_path)])

        command = [python_executable, "test_imports.py"] + args
        result = run_command(command, check=False)

        if result.stdout:
            lines = result.stdout.splitlines()
            in_success_section = False
            for line in lines:
                if line.strip() == "--- SUCCESSFUL IMPORTS ---":
                    in_success_section = True
                    continue
                if line.strip() == "--- FAILED IMPORTS ---":
                    in_success_section = False
                    continue

                if in_success_section:
                    successful_imports.add(line.strip())

        if result.returncode != 0:
            print(f"!!! Import tests failed for {example_dir} !!!")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test imports of examples in the `examples/` directory."
    )
    _ = parser.add_argument(
        "filters",
        nargs="*",
        help="Substring filters to apply to file paths or content.",
    )
    args = parser.parse_args()

    base_dir = Path("examples")
    successful_imports: set[str] = set()

    # Read successful imports
    successful_imports_path = Path("successful_imports.txt")
    if successful_imports_path.exists():
        with open(successful_imports_path, "r") as f:
            successful_imports = set(f.read().splitlines())

    # This environment variable is crucial for NixOS builds
    build_env = {"UV_NO_BUILD_ISOLATION": "1"}

    try:
        # Phase 1: Plan all tests
        filters: list[str] = args.filters or []
        test_plan = plan_tests(base_dir, successful_imports, filters)

        # Phase 2: Execute tests in waves
        execute_test_wave(
            "New Files", test_plan, "new_files_to_test", build_env, successful_imports
        )
        execute_test_wave(
            "Regressions",
            test_plan,
            "successful_files_to_retest",
            build_env,
            successful_imports,
        )

    finally:
        # Write successful imports
        with open(successful_imports_path, "w") as f:
            for item in sorted(list(successful_imports)):
                _ = f.write(f"{item}\n")


if __name__ == "__main__":
    main()
