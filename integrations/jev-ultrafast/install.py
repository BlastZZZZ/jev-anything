"""Apply the browser adapter to a compatible jev-ultrafast checkout."""

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    target = args.checkout.resolve()
    source = Path(__file__).resolve().parent
    if not (target / "jev_ultrafast/model.py").is_file():
        parser.error("checkout must contain jev_ultrafast/model.py")
    patch = source / "agent.patch"
    check = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=target,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        reverse = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=target,
            capture_output=True,
            text=True,
        )
        if reverse.returncode:
            raise SystemExit(
                "Patch does not match this checkout. Use the upstream commit in upstream.json.\n"
                + check.stderr
            )
    else:
        subprocess.run(["git", "apply", str(patch)], cwd=target, check=True)
    shutil.copyfile(source / "local_model.py", target / "jev_ultrafast/local_model.py")
    print("Installed local backend and browser state/progress fixes.")
    print("Set JEV_BACKEND=local and XIAOJEV_HOME to your xiaojev checkout.")


if __name__ == "__main__":
    main()
