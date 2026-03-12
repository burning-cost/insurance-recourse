"""
Run insurance-recourse tests on Databricks via Jobs API (serverless compute).
Execute from local machine: python run_tests_databricks.py
"""

import os
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.config/burning-cost/databricks.env"))

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

PROJECT_ROOT = Path(__file__).parent
WORKSPACE_DIR = "/Workspace/insurance-recourse-tests"


def upload_file(local_path: Path, remote_path: str):
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    parent = str(Path(remote_path).parent)
    try:
        w.workspace.mkdirs(parent)
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=ImportFormat.AUTO,
        overwrite=True,
    )


print("Uploading source files...")

for py_file in (PROJECT_ROOT / "src").rglob("*.py"):
    rel = py_file.relative_to(PROJECT_ROOT / "src")
    upload_file(py_file, f"{WORKSPACE_DIR}/src/{rel}")
    print(f"  src/{rel}")

for py_file in (PROJECT_ROOT / "tests").rglob("*.py"):
    rel = py_file.relative_to(PROJECT_ROOT / "tests")
    upload_file(py_file, f"{WORKSPACE_DIR}/tests/{rel}")
    print(f"  tests/{rel}")

upload_file(PROJECT_ROOT / "pyproject.toml", f"{WORKSPACE_DIR}/pyproject.toml")
print("  pyproject.toml")

test_script = r'''import subprocess, sys, os, shutil

WORKSPACE = "/Workspace/insurance-recourse-tests"
TMP = "/tmp/insurance-recourse-tests"

if os.path.exists(TMP):
    shutil.rmtree(TMP)
shutil.copytree(WORKSPACE, TMP)
print(f"Copied to {TMP}")

install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "pytest", "--quiet"],
    capture_output=True, text=True
)
if install.returncode != 0:
    raise RuntimeError("pytest install failed: " + install.stderr)
print("pytest installed")

src_path = os.path.join(TMP, "src")

env = os.environ.copy()
env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")
env["PYTHONDONTWRITEBYTECODE"] = "1"

# Import check
check = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, '" + src_path + "'); "
     "import insurance_recourse; print('v' + insurance_recourse.__version__)"],
    capture_output=True, text=True, env=env
)
print("Import check:", check.stdout.strip(), check.stderr.strip())

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     os.path.join(TMP, "tests"),
     "-v", "--tb=short", "--no-header",
     "-p", "no:cacheprovider",
     "--color=no"],
    capture_output=True, text=True,
    env=env,
    cwd=TMP,
)

output_lines = result.stdout
if result.stderr.strip():
    output_lines += "\nSTDERR: " + result.stderr

print(output_lines)

rc = result.returncode
status = "PASSED" if rc == 0 else "FAILED"
summary_start = max(0, len(output_lines) - 4000)
summary = output_lines[summary_start:]
try:
    dbutils.notebook.exit(f"{status} (rc={rc})\n{summary}")
except NameError:
    pass

if rc != 0:
    raise RuntimeError(f"Tests {status}")
'''

notebook_content = "# Databricks notebook source\n# COMMAND ----------\n" + test_script
encoded = base64.b64encode(notebook_content.encode()).decode()

try:
    w.workspace.delete(f"{WORKSPACE_DIR}/run_tests")
except Exception:
    pass

w.workspace.import_(
    path=f"{WORKSPACE_DIR}/run_tests",
    content=encoded,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"\nTest notebook: {WORKSPACE_DIR}/run_tests")

print("\nSubmitting job (serverless)...")
run = w.jobs.submit(
    run_name="insurance-recourse-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=f"{WORKSPACE_DIR}/run_tests",
            ),
        )
    ],
)

run_id = run.run_id
host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
print(f"Run ID: {run_id}")
print(f"View: {host}/#job/runs/{run_id}")

print("\nPolling (every 20s)...")
start = time.time()
while True:
    state = w.jobs.get_run(run_id=run_id)
    lc = state.state.life_cycle_state.value if state.state and state.state.life_cycle_state else "?"
    rs = state.state.result_state.value if state.state and state.state.result_state else ""
    elapsed = int(time.time() - start)
    print(f"  [{elapsed:3d}s] {lc} {rs}")

    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        success = rs == "SUCCESS"
        task_run_id = None
        for t in (state.tasks or []):
            if t.task_key == "run_tests":
                task_run_id = t.run_id
                break

        if task_run_id:
            try:
                output = w.jobs.get_run_output(run_id=task_run_id)
                if output:
                    if output.notebook_output and output.notebook_output.result:
                        print("\n--- Test output ---")
                        print(output.notebook_output.result)
                    if output.error:
                        print("\n--- Error ---")
                        print(output.error)
                    if output.error_trace:
                        print("\n--- Error trace (last 3000 chars) ---")
                        print(output.error_trace[-3000:])
            except Exception as e:
                print(f"(Could not fetch output: {e})")

        print(f"\n{'TESTS PASSED' if success else 'TESTS FAILED'}: {rs}")
        if not success:
            raise SystemExit(1)
        break
    time.sleep(20)
