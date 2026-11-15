"""Collect the files the Space needs and, with --upload, push them to Hugging Face.

Without --upload it only lists them. Only files that git tracks are taken, so nothing private can
slip in. Log in first with `hf auth login`, and upload without HF_HUB_OFFLINE set.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from lotse.config import ROOT, SPACE_REPO

PARTS = ["app", "src", "data/manifest.json", "requirements.txt"]


def stage(target: Path) -> list[str]:
    """Copy the Space's files into target and return their names."""
    listed = subprocess.run(
        ["git", "ls-files", *PARTS], cwd=ROOT, capture_output=True, text=True, check=True
    )
    for name in listed.stdout.split():
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, target / name)
    # A Space reads its settings from the top of its README, which GitHub would show as a table.
    shutil.copy(ROOT / "space" / "README.md", target / "README.md")
    return sorted(
        file.relative_to(target).as_posix() for file in target.rglob("*") if file.is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help=f"push the files to {SPACE_REPO}")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as folder:
        print("\n".join(stage(Path(folder))))
        if args.upload:
            api = HfApi()
            # Creating a Gradio Space now needs a paid plan, even when it exists, so ask first.
            if not api.repo_exists(SPACE_REPO, repo_type="space"):
                api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio")
            api.upload_folder(
                folder_path=folder,
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message="Deploy the demo",
                delete_patterns="*",
            )
            print(f"Uploaded to https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
