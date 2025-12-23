#!/usr/bin/env python
"""
build_hf_audio_dataset.py

Liest ein LJSpeech-artiges Dataset (z.B. Thorsten-Voice) ein und baut daraus
ein Hugging-Face-Dataset im Parquet-Format mit Spalten:

- id    : Sample-ID (z.B. "LJ001-0001")
- text  : Transkription (aus metadata.csv)
- audio : HF Audio-Feature, verweist auf die WAV-Datei

Anschließend wird das Dataset in ein Hugging-Face-Dataset-Repo gepusht.
"""

import os
import argparse
from typing import List, Dict, Any, Optional

from datasets import Dataset, Features, Value, Audio
from huggingface_hub import HfApi


# ---------------------------------------------------------------------------
# 1. LJSpeech-artige Metadaten einlesen
# ---------------------------------------------------------------------------

def load_ljs_metadata(
    root_dir: str,
    metadata_filename: str = "metadata.csv",
    wav_subdir: str = "wavs",
    audio_ext: str = ".wav",
    text_column_index: int = -1,
) -> List[Dict[str, Any]]:
    """
    Erwartet eine metadata.csv im (klassischen) LJSpeech-Format:

        <id>|<text>|<optional_norm_text>

    Beispiel:
        LJ001-0001|Mary had a little lamb.|MARY HAD A LITTLE LAMB.

    Parameter:
    ----------
    root_dir: Wurzelverzeichnis des Datasets
    metadata_filename: Name der Metadatendatei (Default: metadata.csv)
    wav_subdir: Unterordner, in dem die WAV-Dateien liegen (Default: "wavs")
    audio_ext: Dateiendung der Audiodateien (Default: ".wav")
    text_column_index: Index der Spalte, aus der der Text genommen wird.
                       -1 bedeutet „letzte Spalte“ (Default).

    Rückgabe:
    ---------
    Liste von Dicts mit:
      - "id"
      - "text"
      - "audio_path"
    """
    meta_path = os.path.join(root_dir, metadata_filename)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata file not found: {meta_path}")

    data: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 2:
                # Zeile ist wahrscheinlich kaputt → skip
                print(f"[WARN] malformed metadata line (too few columns): {line}")
                continue

            utt_id = parts[0].strip()

            # Textspalte wählen (Standard: letzte Spalte)
            try:
                text = parts[text_column_index].strip()
            except IndexError:
                print(f"[WARN] text_column_index {text_column_index} invalid for line: {line}")
                continue

            wav_rel = os.path.join(wav_subdir, utt_id + audio_ext)
            wav_path = os.path.join(root_dir, wav_rel)

            if not os.path.isfile(wav_path):
                print(f"[WARN] audio file not found, skipping: {wav_path}")
                continue

            data.append(
                {
                    "id": utt_id,
                    "text": text,
                    "audio_path": wav_path,
                }
            )

    if not data:
        raise RuntimeError("No valid rows found in metadata file.")
    return data


# ---------------------------------------------------------------------------
# 2. Hugging-Face-Dataset bauen (Audio-Feature)
# ---------------------------------------------------------------------------

def build_hf_audio_dataset(
    examples: List[Dict[str, Any]],
    sampling_rate: int = 24000,
) -> Dataset:
    """
    Baut ein HF-Dataset mit Spalten:
      - id: string
      - text: string
      - audio: Audio-Feature (pfad-basiert, lazy geladen)

    sampling_rate dient hier vor allem als Metadatum für das Audio-Feature;
    die WAV-Dateien selbst werden in Original-SR eingelesen, wenn sie
    später von HF geladen werden.
    """
    features = Features(
        {
            "id": Value("string"),
            "text": Value("string"),
            "audio": Audio(sampling_rate=sampling_rate),
        }
    )

    rows = []
    for ex in examples:
        rows.append(
            {
                "id": ex["id"],
                "text": ex["text"],
                # HF Dataset Audio-Feature interpretiert Pfad automatisch
                "audio": ex["audio_path"],
            }
        )

    ds = Dataset.from_list(rows, features=features)
    return ds


# ---------------------------------------------------------------------------
# 3. Dataset auf Hugging Face Hub hochladen
# ---------------------------------------------------------------------------

def push_dataset_to_hub(
    ds: Dataset,
    repo_id: str,
    hf_token: Optional[str] = None,
    private: bool = False,
):
    """
    Lädt ein Dataset nach Hugging Face hoch.

    Parameter:
    ----------
    ds:     das zu pushende Dataset
    repo_id: z.B. "Thorsten-Voice/TV-24kHz-Full"
    hf_token: HF-Token (optional, sonst HF_TOKEN / Cache)
    private: True → privates Dataset
    """
    if hf_token is None:
      hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError(
            "No HF token found. Pass --hf-token or set HF_TOKEN env variable."
        )

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=hf_token,
    )

    # schreibt automatisch Arrow/Parquet + Dateien ins Repo
    ds.push_to_hub(repo_id=repo_id, token=hf_token)


# ---------------------------------------------------------------------------
# 4. CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and upload a Hugging Face audio dataset from a LJSpeech-style corpus."
    )

    # Basisdaten
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory of the LJSpeech-style dataset",
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="metadata.csv",
        help="Name of the metadata file (default: metadata.csv)",
    )
    parser.add_argument(
        "--wav-subdir",
        type=str,
        default="wavs",
        help="Subdirectory under root that contains wav files (default: wavs).",
    )
    parser.add_argument(
        "--audio-ext",
        type=str,
        default=".wav",
        help="Audio file extension (default: .wav)",
    )
    parser.add_argument(
        "--text-column-index",
        type=int,
        default=-1,
        help="Index of text column in metadata (default: -1 → last column).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="Sampling rate metadata for HF Audio feature (default: 24000).",
    )

    # Hugging Face
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help='HF dataset repo id, e.g. "Thorsten-Voice/TV-24kHz-Full".',
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted, HF_TOKEN env or cached token is used.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create private dataset on Hugging Face Hub.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Metadaten einlesen
    print("[INFO] Loading metadata...")
    examples = load_ljs_metadata(
        root_dir=args.root_dir,
        metadata_filename=args.metadata_filename,
        wav_subdir=args.wav_subdir,
        audio_ext=args.audio_ext,
        text_column_index=args.text_column_index,
    )
    print(f"[INFO] Found {len(examples)} valid examples")

    # 2) HF-Dataset bauen
    print("[INFO] Building HF audio dataset...")
    ds = build_hf_audio_dataset(
        examples,
        sampling_rate=args.sampling_rate,
    )
    print(ds)

    # 3) Push to Hub
    print(f"[INFO] Pushing dataset to hub as {args.hf_repo} ...")
    push_dataset_to_hub(
        ds,
        repo_id=args.hf_repo,
        hf_token=args.hf_token,
        private=args.hf_private,
    )
    print("[INFO] Done. Dataset uploaded.")


if __name__ == "__main__":
    main()
