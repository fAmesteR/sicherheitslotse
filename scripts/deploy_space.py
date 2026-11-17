"""Collect the files the Space needs and, with --upload, push them to Hugging Face.

Without --upload it only lists them. The files come from the last commit, not from the working
folder, so neither an unsaved edit nor an untracked file can ship. Log in first with
`hf auth login`, and upload without HF_HUB_OFFLINE set.
"""

import argparse
import io
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

from lotse.config import ROOT, SPACE_REPO

PARTS = ["app", "src", "data/manifest.json", "requirements.txt"]
FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED"}
STARTING_STAGES = {"BUILDING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING"}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def stage(target: Path) -> list[str]:
    """Write the Space's files from HEAD into target and return their names."""
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", "HEAD", *PARTS))) as tar:
        tar.extractall(target, filter="data")
    # A Space reads its settings from the top of its README, which GitHub would show as a table.
    (target / "README.md").write_bytes(git("show", "HEAD:space/README.md"))
    return sorted(
        file.relative_to(target).as_posix() for file in target.rglob("*") if file.is_file()
    )


def wait_until_running(api: HfApi, commit: str, limit: float = 1800) -> None:
    """Until the Space runs this commit. Stops with an error if its build or start fails."""
    start, last, started = time.monotonic(), None, False
    while time.monotonic() - start < limit:
        runtime = api.get_space_runtime(SPACE_REPO)
        current = runtime.raw.get("sha") == commit
        started = started or runtime.stage in STARTING_STAGES
        if (runtime.stage, current) != last:
            print(f"{time.monotonic() - start:5.0f} s  {runtime.stage}  this commit: {current}")
            last = runtime.stage, current
        if current and runtime.stage == "RUNNING":
            return
        # A Space that failed reports no commit, so a failure after the upload's build began is
        # this commit's. One seen before that may still be left from an earlier deploy.
        if (current or started) and runtime.stage in FAILED_STAGES:
            message = runtime.raw.get("errorMessage") or "see its logs"
            raise SystemExit(f"The Space stopped at {runtime.stage}: {message}")
        time.sleep(15)
    raise SystemExit(f"The Space didn't run {commit[:7]} within {limit / 60:.0f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help=f"push the files to {SPACE_REPO}")
    parser.add_argument("--wait", action="store_true", help="then wait until the Space runs them")
    args = parser.parse_args()

    head = git("rev-parse", "--short", "HEAD").decode().strip()
    if git("status", "--porcelain", "--", *PARTS, "space/README.md"):
        print(f"Uncommitted changes are left out: this deploys {head} as committed.")
    with tempfile.TemporaryDirectory() as folder:
        print("\n".join(stage(Path(folder))))
        if not args.upload:
            return
        api = HfApi()
        # Creating a Gradio Space now needs a paid plan, even when it exists, so ask first.
        if not api.repo_exists(SPACE_REPO, repo_type="space"):
            api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio")
        commit = api.upload_folder(
            folder_path=folder,
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message=f"Deploy {head}",
            delete_patterns="*",
        )
        print(f"Uploaded {head} to https://huggingface.co/spaces/{SPACE_REPO}")
    if args.wait:
        wait_until_running(api, commit.oid)


if __name__ == "__main__":
    main()
