"""Prépare les partitions de TinyStories pour les deux modèles."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download
from pyarrow.parquet import ParquetFile


DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
LICENSE = "CDLA-Sharing-1.0"
ROOT = Path(__file__).resolve().parent
VALIDATION_PERCENT = 5
SPLITS = ("train", "validation", "test")


def read_stories(paths):
    """Une ligne Parquet donne un récit. Un texte vide donne None."""
    for path in paths:
        with ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(columns=["text"], batch_size=4096):
                for text in batch.column(0).to_pylist():
                    yield text if text and text.strip() else None


def fingerprint(text):
    return sha256(text.encode("utf-8")).digest()


def write_story(stream, text, digest):
    record = {"sha256": digest.hex(), "text": text}
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_test(paths, destination):
    """Conserve le split officiel de validation comme test final."""
    test_hashes = set()
    counts = {"blank_rows": 0, "exact_duplicates_kept": 0}
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for text in read_stories(paths):
            if text is None:
                counts["blank_rows"] += 1
                continue
            digest = fingerprint(text)
            counts["exact_duplicates_kept"] += int(digest in test_hashes)
            test_hashes.add(digest)
            write_story(stream, text, digest)
    return test_hashes, counts


def collect_train_candidates(paths, test_hashes):
    seen = set()
    counts = {"blank_rows": 0, "test_overlap_rows_removed": 0, "exact_duplicates_removed": 0}
    for text in read_stories(paths):
        if text is None:
            counts["blank_rows"] += 1
            continue
        digest = fingerprint(text)
        if digest in test_hashes:
            counts["test_overlap_rows_removed"] += 1
        elif digest in seen:
            counts["exact_duplicates_removed"] += 1
        else:
            seen.add(digest)
    return seen, counts


def write_train_and_validation(paths, destination, candidates, validation_hashes):
    # Second passage : les textes restent sur disque et seules les empreintes
    # sont gardées en mémoire. On écrit la première occurrence de chaque récit.
    with (
        (destination / "train.jsonl").open("w", encoding="utf-8", newline="\n") as train,
        (destination / "validation.jsonl").open("w", encoding="utf-8", newline="\n") as validation,
    ):
        for text in read_stories(paths):
            if text is None:
                continue
            digest = fingerprint(text)
            if digest in candidates:
                candidates.remove(digest)
                stream = validation if digest in validation_hashes else train
                write_story(stream, text, digest)
    if candidates:
        raise ValueError("Des récits attendus manquent au second passage sur le train.")


def verify_partitions(directory):
    """Relit les fichiers, vérifie leurs empreintes et mesure leur taille."""
    summaries = {}
    hashes_by_split = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        file_hash = sha256()
        story_hashes = set()
        stories = utf8_bytes = 0
        with path.open("rb") as stream:
            for line in stream:
                file_hash.update(line)
                record = json.loads(line)
                raw = record["text"].encode("utf-8")
                digest = sha256(raw).digest()
                if digest.hex() != record["sha256"]:
                    raise ValueError(f"Empreinte de récit incorrecte dans {path}.")
                story_hashes.add(digest)
                stories += 1
                utf8_bytes += len(raw)
        if split != "test" and len(story_hashes) != stories:
            raise ValueError(f"Doublon exact dans {path}.")
        hashes_by_split[split] = story_hashes
        summaries[split] = {
            "file": path.name,
            "stories": stories,
            "unique_story_hashes": len(story_hashes),
            "text_utf8_bytes": utf8_bytes,
            "file_bytes": path.stat().st_size,
            "file_sha256": file_hash.hexdigest(),
        }
    for left, right in combinations(SPLITS, 2):
        if not hashes_by_split[left].isdisjoint(hashes_by_split[right]):
            raise ValueError(f"Des récits sont communs à {left} et {right}.")
    return summaries


def prepare_partitions(train_paths, test_paths, destination):
    print("Lecture du test officiel…", flush=True)
    test_hashes, test_counts = write_test(test_paths, destination / "test.jsonl")
    print("Calcul des empreintes du train officiel…", flush=True)
    candidates, train_counts = collect_train_candidates(train_paths, test_hashes)
    validation_size = len(candidates) * VALIDATION_PERCENT // 100
    validation_hashes = set(sorted(candidates)[:validation_size])
    print("Écriture du train et de la validation…", flush=True)
    write_train_and_validation(train_paths, destination, candidates, validation_hashes)
    print("Vérification des trois fichiers…", flush=True)
    summaries = verify_partitions(destination)
    if summaries["validation"]["stories"] != validation_size:
        raise ValueError("Effectif de validation différent de l'effectif prévu.")
    return {
        "official_train": train_counts,
        "official_validation": test_counts,
    }, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "tinystories",
        help="Répertoire de sortie à créer.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"{output} existe déjà. Choisir un autre répertoire avec --output.")

    print(f"Récupération de {DATASET} à la révision {REVISION}…", flush=True)
    raw = Path(snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns="data/*.parquet",
        local_dir=ROOT / "data" / "raw" / f"tinystories-{REVISION[:7]}",
    ))
    train_paths = sorted((raw / "data").glob("train-*.parquet"))
    test_paths = sorted((raw / "data").glob("validation-*.parquet"))
    if len(train_paths) != 4 or len(test_paths) != 1:
        raise ValueError("La révision attendue fournit quatre fichiers train et un fichier validation.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".prepare-tinystories-") as temporary:
        staging = Path(temporary) / "corpus"
        staging.mkdir()
        filtering, summaries = prepare_partitions(train_paths, test_paths, staging)
        manifest = {
            "source": {
                "dataset": DATASET,
                "revision": REVISION,
                "license": LICENSE,
                "official_train_files": [path.relative_to(raw).as_posix() for path in train_paths],
                "official_validation_files": [path.relative_to(raw).as_posix() for path in test_paths],
            },
            "rules": {
                "story_fingerprint": "sha256 of original UTF-8 text",
                "blank_rows": "exclude null, empty or whitespace-only text",
                "test": "official validation, preserving nonblank rows in source order",
                "train_candidates": "first nonblank row per fingerprint, excluding all test fingerprints",
                "validation": f"lowest floor({VALIDATION_PERCENT}% of candidates) fingerprints",
                "train": "all remaining candidates",
                "output_order": "source file name, then source row order within each split",
            },
            "filtering": filtering,
            "splits": summaries,
            "story_hashes_pairwise_disjoint": True,
            "limitation": "Exact matching does not detect near-duplicates.",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n",
        )
        staging.rename(output)

    for split, summary in summaries.items():
        print(f"{split}: {summary['stories']:,} récits, {summary['text_utf8_bytes']:,} octets de texte")
    print(f"Manifeste vérifié : {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
