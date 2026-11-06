#!/usr/bin/env python3
"""
Anonymize MathJobs training data for sharing.

QUICK START (PowerShell, from the repo folder):
-----------
    # After a search is complete, run:
    pip install pandas   # if not already installed
    python anonymize_training.py appl.csv ratings.csv -o training/2024/

DETAILED INSTRUCTIONS:
---------------------
1. Open PowerShell
2. Go to the folder that contains this script:
       cd path/to/mathjobs-rankings
3. Make sure pandas is installed:
       pip install pandas
4. Run the script:
       python anonymize_training.py appl.csv ratings.csv -o training/2024/

   Replace "2024" with the current year's folder name.

WHAT IT DOES:
    - Replaces candidate names/emails/URLs with anonymous IDs (Applicant_0001, etc.)
    - Keeps faculty names (needed for collaborative filtering)
    - Keeps ratings (the core training data)
    - Removes comments, research areas, organizations (potentially identifying)

OUTPUT FILES (in the specified output folder):
    - appl_anon.csv: Anonymized applicant data
    - ratings_anon.csv: Anonymized ratings data
"""

import argparse
import pandas as pd
from pathlib import Path


def anonymize_training_data(appl_path: str, ratings_path: str, output_dir: str):
    """
    Anonymize training data for sharing.

    Keeps:
    - Applicant Uid (mapped to anonymous ID)
    - Faculty names/IDs (raters - needed for collaborative filtering)
    - Ratings (the core training signal)
    - Job type (POTT, INST)

    Removes:
    - Candidate names, emails, URLs
    - Comments (subjective, potentially embarrassing)
    - Research areas (potentially identifying)
    - Organization, DegreeCountry (potentially identifying)
    """
    # Read input files
    appl_df = pd.read_csv(appl_path)
    ratings_df = pd.read_csv(ratings_path)

    # Create anonymous applicant ID mapping
    unique_uids = appl_df['Uid'].unique()
    uid_map = {uid: f"Applicant_{i:04d}" for i, uid in enumerate(unique_uids, 1)}

    print(f"Anonymizing {len(unique_uids)} applicants...")

    # Anonymize appl.csv - keep only what's needed for training
    anon_appl_columns = {
        'AnonId': appl_df['Uid'].map(uid_map),
        'AppliedFor': appl_df['AppliedFor'],
        'MyRating': appl_df['MyRating'],
        'Avg': appl_df['Avg'],
        'Ratings': appl_df['Ratings'],
    }
    anon_appl = pd.DataFrame(anon_appl_columns)

    # Anonymize ratings.csv - keep faculty info and ratings
    anon_ratings_columns = {
        'AnonId': ratings_df['Uid'].map(uid_map),
        'JobTag': ratings_df['JobTag'],
        'FacultyUid': ratings_df['FacultyUid'],
        'Faculty': ratings_df['Faculty'],
        'Rating': ratings_df['Rating'],
    }
    anon_ratings = pd.DataFrame(anon_ratings_columns)

    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save anonymized files
    appl_out = output_path / 'appl_anon.csv'
    ratings_out = output_path / 'ratings_anon.csv'

    anon_appl.to_csv(appl_out, index=False)
    anon_ratings.to_csv(ratings_out, index=False)

    print(f"Saved: {appl_out}")
    print(f"Saved: {ratings_out}")
    print(f"\nAnonymized {len(anon_appl)} applicants, {len(anon_ratings)} ratings.")

    return anon_appl, anon_ratings


def main():
    parser = argparse.ArgumentParser(
        description='Anonymize MathJobs training data for sharing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE:
    python anonymize_training.py appl.csv ratings.csv --output training/2024/

This will create:
    training/2024/appl_anon.csv
    training/2024/ratings_anon.csv
"""
    )
    parser.add_argument('appl_csv', help='Path to appl.csv (applicant data)')
    parser.add_argument('ratings_csv', help='Path to ratings.csv (faculty ratings)')
    parser.add_argument('--output', '-o', default='.',
                        help='Output directory (default: current directory)')

    args = parser.parse_args()

    anonymize_training_data(args.appl_csv, args.ratings_csv, args.output)


if __name__ == '__main__':
    main()
