"""Fetch the laws into data/raw, then record them and the saved PDFs in data/manifest.json."""

import hashlib
import json
import urllib.request
from datetime import date

from lotse.config import BAUA_ASR_URL, GII_URL, LAWS, MANIFEST, PDF_DIR, PDFS, RAW_DIR
from lotse.parsing.laws_xml import load_root, parse_xml, version_date


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "sicherheitslotse/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    toc = parse_xml(fetch(f"{GII_URL}/gii-toc.xml"))
    links = {link.text.replace("http://", "https://") for link in toc.iter("link")}

    manifest = []
    for doc, slug in LAWS.items():
        url = f"{GII_URL}/{slug}/xml.zip"
        if url not in links:
            raise SystemExit(f"{doc}: {url} is not in gii-toc.xml, the slug has changed")
        data = fetch(url)
        path = RAW_DIR / f"{slug}.zip"
        path.write_bytes(data)
        manifest.append(
            {
                "doc": doc,
                "url": url,
                "landing_page": f"{GII_URL}/{slug}/",
                "downloaded": date.today().isoformat(),
                "version_date": version_date(load_root(path)),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        print(f"{doc:24} {manifest[-1]['version_date']}  {len(data):>7} bytes")

    for doc, (stem, _, version) in PDFS.items():
        path = PDF_DIR / f"{stem}.pdf"
        url = f"{BAUA_ASR_URL}/pdf/{stem}.pdf?__blob=publicationFile"
        if not path.exists():
            raise SystemExit(f"{doc}: open {url} in a browser and save it as {path}")
        data = path.read_bytes()
        manifest.append(
            {
                "doc": doc,
                "url": url,
                "landing_page": f"{BAUA_ASR_URL}/{stem}",
                "downloaded": date.fromtimestamp(path.stat().st_mtime).isoformat(),
                "version_date": version,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        print(f"{doc:24} {version}  {len(data):>7} bytes")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
