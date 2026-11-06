#!/usr/bin/env python3
"""
Turn a MathJobs ratings export into an anonymized training file.

After a search is over, export ratings.csv from MathJobs and run, from the
folder that contains this script:

    python anonymize_training.py ratings.csv -o training/2026

Several exports from the same year (for example, two positions) go into one
file:

    python anonymize_training.py pott_ratings.csv instr_ratings.csv -o training/2024

The output, training/<year>/ratings.csv, has three columns and nothing else:

    FacultyUid   the rater's MathJobs ID (used to learn each rater's habits)
    ApplicantId  a random ID (A0001, A0002, ...) standing in for the applicant
    Rating       the score

Applicant names, emails, and comments, faculty names, and every other column
are dropped. Every rated applicant gets an ID, whether or not they appear in
appl.csv. IDs are assigned in random order, so they don't follow the
alphabetical order of the export.

Keep training data out of the repository. .gitignore blocks training/ and
*.csv, but the files belong in a private location.

Requires pandas and numpy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ["Uid", "FacultyUid", "Rating"]


def anonymize_training_data(ratings_paths: list, output_dir: str, force: bool = False) -> pd.DataFrame:
    """
    Read one or more MathJobs ratings exports and write an anonymized
    training/<year>/ratings.csv with columns FacultyUid, ApplicantId, Rating.
    """
    frames = []
    for path in ratings_paths:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}; expected a MathJobs ratings export.")
        frames.append(df[REQUIRED])

    df = pd.concat(frames, ignore_index=True).dropna(subset=REQUIRED)
    # One rating per rater and applicant (someone rated in two exports counts once)
    df = df.groupby(["FacultyUid", "Uid"], as_index=False)["Rating"].mean()

    uids = df["Uid"].unique()
    order = np.random.default_rng().permutation(len(uids))
    id_map = {uid: f"A{i + 1:04d}" for uid, i in zip(uids, order)}

    out = pd.DataFrame({
        "FacultyUid": df["FacultyUid"].astype(int),
        "ApplicantId": df["Uid"].map(id_map),
        "Rating": df["Rating"].astype(float),
    }).sort_values(["ApplicantId", "FacultyUid"])

    out_dir = Path(output_dir)
    out_path = out_dir / "ratings.csv"
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; use --force to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved {out_path}: {len(out)} ratings of {len(uids)} applicants "
          f"by {out['FacultyUid'].nunique()} raters.")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Turn MathJobs ratings exports into an anonymized training file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    python anonymize_training.py ratings.csv -o training/2026
    python anonymize_training.py pott_ratings.csv instr_ratings.csv -o training/2024
""",
    )
    parser.add_argument("ratings_csv", nargs="+", help="MathJobs ratings export(s) from one year")
    parser.add_argument("--output", "-o", required=True, help="Output folder, e.g. training/2026")
    parser.add_argument("--force", action="store_true", help="Replace an existing ratings.csv")
    args = parser.parse_args()

    try:
        anonymize_training_data(args.ratings_csv, args.output, force=args.force)
    except (ValueError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
