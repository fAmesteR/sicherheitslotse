"""Write the index for the cleared documents to index_public/ and, with --upload, to Hugging Face.

Only documents in API_DOCS leave this machine. The dataset repo is private.
Log in first with `hf auth login`, and upload without HF_HUB_OFFLINE set.
"""

import argparse

from huggingface_hub import HfApi

from lotse.config import API_DOCS, INDEX_REPO, PUBLIC_INDEX_DIR
from lotse.retrieval.index import load_index, save_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="push index_public/ to the dataset")
    args = parser.parse_args()

    save_subset(load_index(), API_DOCS, PUBLIC_INDEX_DIR)
    saved = load_index(PUBLIC_INDEX_DIR)  # read it back: this is what would be uploaded
    docs = {chunk["doc"] for chunk in saved.chunks}
    if not docs <= API_DOCS:
        raise SystemExit(f"Not cleared for publishing: {sorted(docs - API_DOCS)}")
    print(f"{len(saved.chunks)} chunks from {len(docs)} documents in {PUBLIC_INDEX_DIR}")

    if args.upload:
        api = HfApi()
        api.create_repo(INDEX_REPO, repo_type="dataset", private=True, exist_ok=True)
        commit = api.upload_folder(
            folder_path=PUBLIC_INDEX_DIR,
            repo_id=INDEX_REPO,
            repo_type="dataset",
            commit_message=f"Index of {len(saved.chunks)} chunks from {len(docs)} documents",
            delete_patterns="*",
        )
        print(f'Uploaded. Now set INDEX_REVISION = "{commit.oid}" in src/lotse/config.py')


if __name__ == "__main__":
    main()
