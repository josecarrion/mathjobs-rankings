#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathJobs Rankings - Faculty Hiring Application Processor

Processes MathJobs export files to:
1. Load faculty ratings of job candidates
2. Use collaborative filtering (SVD++) to predict missing ratings
3. Normalize ratings across faculty (so tough/easy graders are comparable),
   using each rater's actual ratings, shrunk toward the pooled values
4. Repeat on resampled ratings (fixed seed) to get, for each candidate, a
   score, the range covering the middle half of runs, and the chance of
   landing in the top K

Input files (from MathJobs export):
- ratings.csv: Faculty ratings of candidates
- appl.csv: Applicant information (filtered to PageUp candidates)

Optional training data (helps most early in a search, when few ratings are in):
- A folder of year subfolders, each with an anonymized ratings.csv written by
  anonymize_training.py:
  training/
    2022/ratings.csv
    2024/ratings.csv

Output:
- mathjobs ranking YYYY-MM-DD.xlsx

Based on script used for SRS by Ken Richardson.

Author: José Carrión (j.carrion@tcu.edu)
Original: 2022-09-22
Refactored: 2025-11-26
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill

# Surprise library for collaborative filtering
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import KFold, cross_validate


# =============================================================================
# Constants
# =============================================================================

MIN_RATINGS_PER_FACULTY = 5  # Exclude faculty who rated fewer than this many candidates
RATING_SCALE = (1, 10)  # MathJobs rating scale
CV_FOLDS = 4  # Cross-validation folds for algorithm evaluation
N_RUNS = 50  # Resampled runs used for the score range and the top-K chance
TOP_K = 15  # "Chance of top K" column (roughly the Zoom line)
BASE_SEED = 2026  # Fixed seed so the same data always gives the same output
SHRINK_K = 5  # Pseudo-ratings pulling each rater's mean/SD toward the pooled values
RANGE_PCT = (25, 75)  # Percentiles shown as the score range (middle half of runs)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CandidateResult:
    """Results for a single candidate."""
    name: str
    avg_rating: float
    max_rating: float
    min_rating: float
    num_ratings: int
    ratings_by_faculty: dict = field(default_factory=dict)
    # avg_rating is the median score over the resampled runs; min/max_rating
    # hold the score range (RANGE_PCT percentiles) over those runs.
    top_k_chance: float = 0.0  # Fraction of runs in which the candidate ranked in the top K


@dataclass
class ProcessingResults:
    """Container for all processing results."""
    candidates: list  # List of CandidateResult
    faculty_names: list  # List of faculty names with rating counts
    predictions_matrix: list  # Full matrix of actual + predicted ratings
    algorithm_metrics: dict  # RMSE, MAE from cross-validation
    training_years: list = field(default_factory=list)  # Years of training data used
    n_runs: int = 0
    top_k: int = 0
    seed: int = 0


@dataclass
class TrainingDataset:
    """A single year's training data."""
    year: str
    ratings_path: Path
    applicants_path: Optional[Path] = None  # not used; kept for compatibility
    num_ratings: int = 0


# Columns of an anonymized training file (written by anonymize_training.py)
TRAINING_COLUMNS = ["FacultyUid", "ApplicantId", "Rating"]


# =============================================================================
# Data Loading
# =============================================================================

def build_full_name(row: pd.Series) -> str:
    """
    Build full name from first, middle, last name fields.

    MathJobs format: "LastName, FirstName MiddleName"
    Handles missing middle names and trailing spaces.
    """
    parts = []
    if pd.notna(row.get('LastName')):
        parts.append(str(row['LastName']).strip())
    if pd.notna(row.get('FirstName')):
        if parts:
            parts[0] += ','
        parts.append(str(row['FirstName']).strip())
    if pd.notna(row.get('MiddleName')):
        parts.append(str(row['MiddleName']).strip())

    return ' '.join(parts).strip()


def load_ratings(filepath: Path, min_ratings: int = MIN_RATINGS_PER_FACULTY) -> pd.DataFrame:
    """
    Load and preprocess ratings CSV from MathJobs.

    Args:
        filepath: Path to ratings.csv
        min_ratings: Minimum ratings a faculty must have made to be included

    Returns:
        DataFrame with ratings and computed Name column
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Ratings file not found: {filepath}")

    df = pd.read_csv(filepath, header=0, sep=",")

    if df.empty:
        raise ValueError(f"Ratings file is empty: {filepath}")

    # Filter out faculty who rated too few candidates
    df = df[df.groupby("FacultyUid").FacultyUid.transform("count") > min_ratings]

    if df.empty:
        raise ValueError(f"No faculty with more than {min_ratings} ratings")

    # Build full names
    df['Name'] = df.apply(build_full_name, axis=1)

    return df


def load_applicants(filepath: Path, pageup_only: bool = True) -> pd.DataFrame:
    """
    Load and preprocess applicants CSV from MathJobs.

    Args:
        filepath: Path to appl.csv
        pageup_only: If True, filter to only PageUp candidates

    Returns:
        DataFrame with applicant info and computed Name column
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Applicants file not found: {filepath}")

    df = pd.read_csv(filepath, header=0, sep=",")

    if df.empty:
        raise ValueError(f"Applicants file is empty: {filepath}")

    # Build full names
    df['Name'] = df.apply(build_full_name, axis=1)

    # Filter to PageUp candidates if requested
    if pageup_only and 'PageUp' in df.columns:
        df = df[df.PageUp == "yes"]

    if df.empty:
        raise ValueError("No candidates found (check PageUp filter)")

    return df


def discover_training_data(training_folder: Path) -> list[TrainingDataset]:
    """
    Discover training data from a folder with year subfolders.

    Expected structure (each ratings.csv written by anonymize_training.py):
        training_folder/
            2022/
                ratings.csv
            2024/
                ratings.csv
            ...

    Args:
        training_folder: Path to the training data folder

    Returns:
        List of TrainingDataset objects, sorted by year
    """
    if not training_folder.exists():
        return []

    if not training_folder.is_dir():
        return []

    datasets = []

    # Look for year subfolders
    for subfolder in sorted(training_folder.iterdir()):
        if not subfolder.is_dir():
            continue

        ratings_path = subfolder / "ratings.csv"
        if ratings_path.exists():
            datasets.append(TrainingDataset(year=subfolder.name, ratings_path=ratings_path))

    return datasets


def load_training_data(
    datasets: list[TrainingDataset],
    verbose: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and combine multiple years of anonymized training data.

    Each year's applicants get IDs prefixed with the folder name, so they can't
    collide with another year's or with this year's MathJobs Uids. A year that
    can't be read stops the run with an error rather than being skipped.

    Args:
        datasets: List of TrainingDataset objects
        verbose: Print progress information

    Returns:
        Tuple of (combined_ratings_df, list_of_years_loaded)
    """
    if not datasets:
        return pd.DataFrame(columns=["FacultyUid", "Uid", "Rating"]), []

    all_ratings = []
    years_loaded = []

    for dataset in datasets:
        df = pd.read_csv(dataset.ratings_path)
        missing = [c for c in TRAINING_COLUMNS if c not in df.columns]
        if missing:
            hint = ""
            if {"Uid", "LastName"} <= set(df.columns):
                hint = " This looks like a raw MathJobs export; run anonymize_training.py on it first."
            raise ValueError(
                f"Training file {dataset.ratings_path} is missing columns {missing}.{hint}"
            )

        df = df.dropna(subset=TRAINING_COLUMNS)
        ratings_for_surprise = pd.DataFrame({
            "FacultyUid": df["FacultyUid"].astype(int),
            "Uid": dataset.year + ":" + df["ApplicantId"].astype(str),
            "Rating": df["Rating"].astype(float),
        })
        if ratings_for_surprise.empty:
            raise ValueError(f"Training file {dataset.ratings_path} has no ratings.")

        all_ratings.append(ratings_for_surprise)
        dataset.num_ratings = len(ratings_for_surprise)
        years_loaded.append(dataset.year)

        if verbose:
            print(f"  Loaded {dataset.year}: {len(ratings_for_surprise)} ratings")

    if all_ratings:
        combined = pd.concat(all_ratings, axis=0, ignore_index=True)
        return combined, years_loaded
    else:
        return pd.DataFrame(columns=["FacultyUid", "Uid", "Rating"]), []


def prepare_ratings_for_surprise(
    ratings_df: pd.DataFrame,
    candidate_names: list
) -> tuple[pd.DataFrame, list, list]:
    """
    Prepare ratings data for the Surprise library.

    Args:
        ratings_df: Raw ratings DataFrame
        candidate_names: List of candidate names to include

    Returns:
        Tuple of (filtered_df, candidate_uids, faculty_uids)
    """
    # Filter to only candidates in our list
    filtered = ratings_df[ratings_df.Name.isin(candidate_names)].copy()

    # Get unique IDs
    candidate_uids = list(set(filtered["Uid"].values.tolist()))
    faculty_uids = list(set(filtered["FacultyUid"].values.tolist()))

    # Build list of (faculty, candidate, rating) tuples, excluding NaN ratings
    ratings_list = []
    for _, row in filtered.iterrows():
        if pd.notna(row['Rating']):
            ratings_list.append([
                int(row['FacultyUid']),
                int(row['Uid']),
                int(row['Rating'])
            ])

    ratings_for_surprise = pd.DataFrame(
        ratings_list,
        columns=["FacultyUid", "Uid", "Rating"]
    )

    return ratings_for_surprise, candidate_uids, faculty_uids


# =============================================================================
# Collaborative Filtering
# =============================================================================

def train_recommender(
    training_df: pd.DataFrame,
    verbose: bool = False,
    seed: Optional[int] = None
) -> tuple[SVDpp, dict]:
    """
    Train the SVD++ collaborative filtering algorithm.

    Args:
        training_df: DataFrame with FacultyUid, Uid, Rating columns
        verbose: Print cross-validation results

    Returns:
        Tuple of (trained_algorithm, metrics_dict)
    """
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(training_df, reader)

    algo = SVDpp(random_state=seed)

    # Cross-validate to get performance metrics
    cv_results = cross_validate(
        algo, data,
        measures=["RMSE", "MAE"],
        cv=KFold(n_splits=CV_FOLDS, random_state=seed),
        verbose=verbose
    )

    metrics = {
        'rmse': float(np.mean(cv_results['test_rmse'])),
        'mae': float(np.mean(cv_results['test_mae']))
    }

    # Train on full dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    return algo, metrics


def build_rating_matrix(
    algo: SVDpp,
    ratings_list: list,
    candidate_uids: list,
    faculty_uids: list
) -> list[list[float]]:
    """
    Build complete rating matrix, filling in predictions for missing ratings.

    Args:
        algo: Trained SVD++ algorithm
        ratings_list: List of [faculty_uid, candidate_uid, rating] lists
        candidate_uids: List of candidate UIDs
        faculty_uids: List of faculty UIDs

    Returns:
        2D matrix [candidates x faculty] with actual and predicted ratings
    """
    # Build lookup dict for O(1) access
    ratings_lookup = {}
    for faculty_uid, candidate_uid, rating in ratings_list:
        ratings_lookup[(faculty_uid, candidate_uid)] = rating

    matrix = []
    for candidate_uid in candidate_uids:
        row = []
        for faculty_uid in faculty_uids:
            key = (faculty_uid, candidate_uid)
            if key in ratings_lookup:
                # Use actual rating
                row.append(float(ratings_lookup[key]))
            else:
                # Predict missing rating
                prediction = algo.predict(uid=faculty_uid, iid=candidate_uid)
                row.append(float(prediction.est))
        matrix.append(row)

    return matrix


# =============================================================================
# Normalization and Statistics
# =============================================================================

def normalize_ratings(matrix: list[list[float]]) -> list[list[float]]:
    """
    Normalize ratings per faculty member (z-score normalization).

    This adjusts for faculty who rate consistently high or low.

    Args:
        matrix: [candidates x faculty] rating matrix

    Returns:
        Normalized matrix (same dimensions)
    """
    if not matrix or not matrix[0]:
        return matrix

    n_candidates = len(matrix)
    n_faculty = len(matrix[0])

    normalized = [[0.0] * n_candidates for _ in range(n_faculty)]

    for j in range(n_faculty):
        # Get this faculty member's ratings
        faculty_ratings = [matrix[i][j] for i in range(n_candidates)]

        mean = sum(faculty_ratings) / len(faculty_ratings)
        variance = sum((r - mean) ** 2 for r in faculty_ratings) / len(faculty_ratings)
        std = math.sqrt(variance) if variance > 0 else 0

        for i in range(n_candidates):
            if std > 0:
                normalized[j][i] = (matrix[i][j] - mean) / std
            else:
                normalized[j][i] = matrix[i][j]

    return normalized


def rescale_ratings(
    normalized: list[list[float]],
    target_min: float = 1.0,
    target_max: float = 10.0
) -> list[list[float]]:
    """
    Rescale normalized ratings to target range.

    Args:
        normalized: [faculty x candidates] normalized matrix
        target_min: Minimum value in output range
        target_max: Maximum value in output range

    Returns:
        Rescaled matrix (same dimensions)
    """
    if not normalized or not normalized[0]:
        return normalized

    n_faculty = len(normalized)
    n_candidates = len(normalized[0])

    rescaled = [[0.0] * n_candidates for _ in range(n_faculty)]

    for i in range(n_faculty):
        row = normalized[i]
        min_val = min(row)
        max_val = max(row)
        range_val = max_val - min_val

        for j in range(n_candidates):
            if range_val > 0:
                rescaled[i][j] = ((row[j] - min_val) / range_val) * (target_max - target_min) + target_min
            else:
                rescaled[i][j] = row[j]

    return rescaled


def compute_candidate_stats(
    rescaled: list[list[float]],
    candidate_uids: list,
    ratings_df: pd.DataFrame,
    actual_ratings: list
) -> list[CandidateResult]:
    """
    Compute statistics for each candidate.

    Args:
        rescaled: [faculty x candidates] rescaled rating matrix
        candidate_uids: List of candidate UIDs
        ratings_df: Original ratings DataFrame (for names)
        actual_ratings: List of actual (non-predicted) ratings

    Returns:
        List of CandidateResult objects, sorted by average rating
    """
    n_candidates = len(candidate_uids)
    n_faculty = len(rescaled)

    # Count actual ratings per candidate
    rating_counts = {}
    for faculty_uid, candidate_uid, rating in actual_ratings:
        if candidate_uid not in rating_counts:
            rating_counts[candidate_uid] = 0
        rating_counts[candidate_uid] += 1

    results = []
    for i, uid in enumerate(candidate_uids):
        # Get this candidate's rescaled ratings from all faculty
        ratings = [rescaled[j][i] for j in range(n_faculty)]

        # Get name from DataFrame
        name_rows = ratings_df.loc[ratings_df["Uid"] == uid]
        if len(name_rows) > 0:
            row = name_rows.iloc[0]
            name = f"{row['LastName']}, {row['FirstName']}"
        else:
            name = f"Unknown (UID: {uid})"

        results.append(CandidateResult(
            name=name,
            avg_rating=round(sum(ratings) / len(ratings), 2),
            max_rating=round(max(ratings), 2),
            min_rating=round(min(ratings), 2),
            num_ratings=rating_counts.get(uid, 0)
        ))

    # Sort by average rating (descending)
    results.sort(key=lambda x: -x.avg_rating)

    return results


def candidate_scores(
    matrix: list[list[float]],
    actual_ratings: list,
    candidate_uids: list,
    faculty_uids: list,
    shrink_k: float = SHRINK_K
) -> np.ndarray:
    """
    One score per candidate from a filled [candidates x faculty] matrix.

    Each rater's leniency (mean) and spread (SD) are estimated from that rater's
    actual ratings only, pulled toward the pooled mean/SD by shrink_k
    pseudo-ratings so raters with few ratings aren't over-corrected. Every entry
    (actual or predicted) is converted to a z-score with its rater's values, the
    z-scores are averaged over raters, and the average is mapped back to the
    rating scale with the pooled mean/SD, so scores read like ordinary ratings.
    """
    m = np.asarray(matrix, dtype=float)
    actual = np.asarray([[f, c, r] for f, c, r in actual_ratings], dtype=float)
    pooled_mean = float(actual[:, 2].mean())
    pooled_sd = float(actual[:, 2].std()) or 1.0

    z = np.zeros_like(m)
    for j, fuid in enumerate(faculty_uids):
        own = actual[actual[:, 0] == fuid, 2]
        n = len(own)
        mean_j = (own.sum() + shrink_k * pooled_mean) / (n + shrink_k)
        var_own = own.var() if n > 1 else 0.0
        var_j = (n * var_own + shrink_k * pooled_sd ** 2) / (n + shrink_k)
        z[:, j] = (m[:, j] - mean_j) / math.sqrt(var_j)

    scores = pooled_mean + pooled_sd * z.mean(axis=1)
    return np.clip(scores, RATING_SCALE[0], RATING_SCALE[1])


def score_one_run(
    current_ratings: pd.DataFrame,
    history: Optional[pd.DataFrame],
    candidate_uids: list,
    faculty_uids: list,
    seed: int,
    resample: bool = True
) -> np.ndarray:
    """
    Score all candidates once.

    With resample=True, this year's ratings are drawn with replacement (a
    bootstrap: some ratings drop out, others count twice), which mimics a
    slightly different set of readers. Historical ratings are kept as they are;
    they only help the model learn how each rater tends to score.
    """
    rng = np.random.default_rng(seed)
    cur = current_ratings
    if resample:
        cur = cur.iloc[rng.integers(0, len(cur), len(cur))]
    # A pair drawn twice counts once, at its rating
    cur = cur.groupby(["FacultyUid", "Uid"], as_index=False)["Rating"].mean()

    fit_df = pd.concat([cur, history], ignore_index=True) if history is not None and len(history) else cur
    data = Dataset.load_from_df(fit_df[["FacultyUid", "Uid", "Rating"]], Reader(rating_scale=RATING_SCALE))
    algo = SVDpp(random_state=seed)
    algo.fit(data.build_full_trainset())

    actual = [[int(f), int(c), float(r)] for f, c, r in cur[["FacultyUid", "Uid", "Rating"]].itertuples(index=False)]
    matrix = build_rating_matrix(algo, actual, candidate_uids, faculty_uids)
    return candidate_scores(matrix, actual, candidate_uids, faculty_uids)


def summarize_runs(
    runs: np.ndarray,
    top_k: int = TOP_K,
    pct: tuple = RANGE_PCT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From a [runs x candidates] score array, return per candidate the median
    score, the score range (pct percentiles), and the fraction of runs in which
    the candidate ranked in the top_k.
    """
    median = np.median(runs, axis=0)
    low = np.percentile(runs, pct[0], axis=0)
    high = np.percentile(runs, pct[1], axis=0)
    # rank 1 = best in each run
    ranks = (-runs).argsort(axis=1).argsort(axis=1) + 1
    chance = (ranks <= top_k).mean(axis=0)
    return median, low, high, chance


# =============================================================================
# Excel Export
# =============================================================================

def create_results_excel(
    results: ProcessingResults,
    output_path: Path
) -> None:
    """
    Create Excel file with ranking results.

    Sheets:
    - Ranking: Candidates sorted by score, with the score range and top-K chance
    - How to read: Three-line explanation for a meeting
    - Predictions: Full matrix of actual and predicted ratings
    - Metrics: Algorithm performance metrics
    """
    k = results.top_k or TOP_K
    chance_header = f"Chance of top {k}"
    lo, hi = RANGE_PCT
    range_header = "Middle half of runs" if hi - lo == 50 else f"Middle {hi - lo}% of runs"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Rankings sheet
        rankings_data = [
            [i, c.name, c.avg_rating, f"{c.min_rating:.1f}–{c.max_rating:.1f}",
             round(c.top_k_chance, 2), c.num_ratings]
            for i, c in enumerate(results.candidates, start=1)
        ]
        rankings_df = pd.DataFrame(rankings_data)
        rankings_df.to_excel(
            writer,
            sheet_name="Ranking",
            header=["Rank", "Candidate", "Score", range_header, chance_header, "# Ratings"],
            index=False
        )

        sheet = writer.sheets["Ranking"]
        sheet.freeze_panes = "A2"
        for col, width in zip("ABCDEF", (6, 34, 8, 13, 16, 10)):
            sheet.column_dimensions[col].width = width
        green = PatternFill("solid", start_color="c6efce")
        yellow = PatternFill("solid", start_color="ffeb9c")
        red = PatternFill("solid", start_color="ffcccc")
        for row_idx, candidate in enumerate(results.candidates, start=2):
            cell = sheet.cell(row=row_idx, column=5)
            cell.number_format = "0%"
            if candidate.top_k_chance >= 0.75:
                cell.fill = green
            elif candidate.top_k_chance >= 0.25:
                cell.fill = yellow
            # Few actual ratings: the score is mostly predicted
            if candidate.num_ratings <= 2:
                sheet.cell(row=row_idx, column=6).fill = red

        # How to read
        notes = [
            ["How to read the Ranking sheet"],
            [f"Score: the candidate's typical committee rating on the 1-10 scale, after adjusting for tough and easy raters and filling in raters who didn't rate them."],
            [f"{range_header}: we re-scored everyone {results.n_runs} times, each time on a resample of the ratings (as if slightly different people had read the files). The middle {hi - lo}% of those re-scorings gave a score in this range."],
            [f"{chance_header}: how often the candidate landed in the top {k} across those runs. Green = 75% or more (clearly above the line), yellow = 25-74% (on the bubble)."],
            ["# Ratings in red: two or fewer actual ratings, so the score rests mostly on predictions."],
        ]
        pd.DataFrame(notes).to_excel(writer, sheet_name="How to read", header=None, index=False)
        writer.sheets["How to read"].column_dimensions["A"].width = 140

        # Predictions sheet
        predictions_df = pd.DataFrame(results.predictions_matrix)
        predictions_df.to_excel(
            writer,
            sheet_name="Predictions",
            header=None,
            index=False
        )

        # Metrics sheet
        metrics_data = [
            ["Metric", "Value"],
            ["RMSE (Root Mean Square Error)", round(results.algorithm_metrics['rmse'], 4)],
            ["MAE (Mean Absolute Error)", round(results.algorithm_metrics['mae'], 4)],
            ["", ""],
            ["Note", "Lower values indicate better prediction accuracy"],
        ]

        # Add training years info if available
        if results.training_years:
            metrics_data.append(["", ""])
            metrics_data.append(["Training Data Years", ", ".join(results.training_years)])

        metrics_data.append(["", ""])
        metrics_data.append(["Resampled runs", results.n_runs])
        metrics_data.append(["Top-K line", k])
        metrics_data.append(["Random seed", results.seed])

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(
            writer,
            sheet_name="Metrics",
            header=None,
            index=False
        )


# =============================================================================
# Main Processing
# =============================================================================

def process_mathjobs_data(
    ratings_path: Path,
    applicants_path: Path,
    training_folder: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    n_runs: int = N_RUNS,
    top_k: int = TOP_K,
    seed: int = BASE_SEED
) -> ProcessingResults:
    """
    Process MathJobs data and produce rankings.

    Args:
        ratings_path: Path to current ratings.csv
        applicants_path: Path to current appl.csv
        training_folder: Path to folder with year subfolders containing historical data
        output_dir: Directory for output file (default: same as ratings_path)
        verbose: Print progress information
        n_runs: Number of resampled runs for the score range and top-K chance
        top_k: Cutoff for the "chance of top K" column
        seed: Base random seed (same data + same seed = same output)

    Returns:
        ProcessingResults object
    """
    if output_dir is None:
        output_dir = ratings_path.parent

    training_years = []

    # Load current data
    if verbose:
        print("Loading current application data...")

    ratings_df = load_ratings(ratings_path)
    applicants_df = load_applicants(applicants_path)

    candidates = [name.strip() for name in applicants_df["Name"].values.tolist()]

    if verbose:
        print(f"  Found {len(candidates)} PageUp candidates")

    # Prepare for Surprise
    current_ratings, candidate_uids, faculty_uids = prepare_ratings_for_surprise(
        ratings_df, candidates
    )

    if verbose:
        print(f"  Found {len(faculty_uids)} faculty with ratings")
        print(f"  Total ratings: {len(current_ratings)}")

    # Load training data if folder provided
    history = None
    if training_folder:
        if verbose:
            print(f"Discovering training data in {training_folder}...")

        datasets = discover_training_data(training_folder)

        if datasets:
            if verbose:
                print(f"  Found {len(datasets)} year(s) of historical data")

            training_data, training_years = load_training_data(datasets, verbose=verbose)

            if len(training_data) > 0:
                # Combine current and training data
                history = training_data
                combined_training = pd.concat([current_ratings, training_data], axis=0, ignore_index=True)

                if verbose:
                    print(f"  Combined training set: {len(combined_training)} ratings")
            else:
                combined_training = current_ratings
        else:
            if verbose:
                print("  No valid training data found")
            combined_training = current_ratings
    else:
        combined_training = current_ratings

    # Train recommender
    if verbose:
        print("Training collaborative filtering model...")

    algo, metrics = train_recommender(combined_training, verbose=verbose, seed=seed)

    if verbose:
        print(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

    # Full-data rating matrix (actual + predicted), shown on the Predictions sheet
    if verbose:
        print("Building rating matrix with predictions...")

    actual_ratings_list = current_ratings.values.tolist()
    rating_matrix = build_rating_matrix(algo, actual_ratings_list, candidate_uids, faculty_uids)

    # Resampled runs: score, score range, and top-K chance
    if verbose:
        print(f"Scoring {n_runs} resampled runs...")

    runs = np.vstack([
        score_one_run(current_ratings, history, candidate_uids, faculty_uids, seed=seed + b, resample=True)
        for b in range(n_runs)
    ])
    median, low, high, chance = summarize_runs(runs, top_k=top_k)

    rating_counts = current_ratings.groupby("Uid").size().to_dict()
    candidate_results = []
    for i, uid in enumerate(candidate_uids):
        name_rows = ratings_df.loc[ratings_df["Uid"] == uid]
        if len(name_rows) > 0:
            row = name_rows.iloc[0]
            name = f"{row['LastName']}, {row['FirstName']}"
        else:
            name = f"Unknown (UID: {uid})"
        candidate_results.append(CandidateResult(
            name=name,
            avg_rating=round(float(median[i]), 2),
            max_rating=round(float(high[i]), 2),
            min_rating=round(float(low[i]), 2),
            num_ratings=int(rating_counts.get(uid, 0)),
            top_k_chance=float(chance[i]),
        ))
    candidate_results.sort(key=lambda c: (-c.avg_rating, -c.top_k_chance))

    if verbose:
        print(f"Ranked {len(candidate_results)} candidates")

    # Build predictions matrix for export
    # Format: First row is faculty names, subsequent rows are candidates
    faculty_names = []
    for fuid in faculty_uids:
        faculty_rows = ratings_df.loc[ratings_df["FacultyUid"] == fuid]
        if len(faculty_rows) > 0:
            name = faculty_rows.iloc[0]["Faculty"]
            count = len(faculty_rows)
            faculty_names.append(f"{name} ({count})")
        else:
            faculty_names.append(f"Faculty {fuid}")

    predictions_matrix = [[""] + faculty_names]
    for i, uid in enumerate(candidate_uids):
        name_rows = ratings_df.loc[ratings_df["Uid"] == uid]
        if len(name_rows) > 0:
            row = name_rows.iloc[0]
            name = f"{row['LastName']}, {row['FirstName']}"
        else:
            name = f"Unknown (UID: {uid})"

        ratings_row = [name] + [round(rating_matrix[i][j], 1) for j in range(len(faculty_uids))]
        predictions_matrix.append(ratings_row)

    # Sort predictions by candidate name
    predictions_matrix[1:] = sorted(predictions_matrix[1:], key=lambda x: x[0])

    # Create results object
    results = ProcessingResults(
        candidates=candidate_results,
        faculty_names=faculty_names,
        predictions_matrix=predictions_matrix,
        algorithm_metrics=metrics,
        training_years=training_years,
        n_runs=n_runs,
        top_k=top_k,
        seed=seed
    )

    # Export to Excel
    date_str = time.strftime("%Y-%m-%d")
    output_path = output_dir / f"mathjobs ranking {date_str}.xlsx"

    if verbose:
        print(f"Saving results to {output_path}...")

    create_results_excel(results, output_path)

    if verbose:
        print("Done!")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process MathJobs ratings and produce candidate rankings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python mathjobs_rankings_new.py
    python mathjobs_rankings_new.py --ratings my_ratings.csv --applicants my_appl.csv
    python mathjobs_rankings_new.py --verbose
    python mathjobs_rankings_new.py --training-folder ./training
    python mathjobs_rankings_new.py --no-training

Training folder structure (each file written by anonymize_training.py):
    training/
        2022/ratings.csv
        2024/ratings.csv
        ...
        """
    )

    parser.add_argument(
        "--ratings", "-r",
        type=Path,
        default=Path("ratings.csv"),
        help="Path to ratings CSV (default: ratings.csv)"
    )

    parser.add_argument(
        "--applicants", "-a",
        type=Path,
        default=Path("appl.csv"),
        help="Path to applicants CSV (default: appl.csv)"
    )

    parser.add_argument(
        "--training-folder", "-t",
        type=Path,
        default=Path("training"),
        help="Path to folder with year subfolders containing historical data (default: training)"
    )

    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Skip loading historical training data"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: same as ratings file)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=N_RUNS,
        help=f"Resampled runs for the score range and top-K chance (default: {N_RUNS})"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Cutoff for the 'chance of top K' column (default: {TOP_K})"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help=f"Random seed; same data and seed give the same output (default: {BASE_SEED})"
    )

    args = parser.parse_args()

    try:
        training_folder = None if args.no_training else args.training_folder

        results = process_mathjobs_data(
            ratings_path=args.ratings,
            applicants_path=args.applicants,
            training_folder=training_folder,
            output_dir=args.output,
            verbose=args.verbose,
            n_runs=args.runs,
            top_k=args.top_k,
            seed=args.seed
        )

        # Print summary
        print("\n" + "=" * 60)
        if results.training_years:
            print(f"Training data: {', '.join(results.training_years)}")
        else:
            print("Training data: none (this year's ratings only)")
        print(f"Model accuracy: RMSE={results.algorithm_metrics['rmse']:.3f}, MAE={results.algorithm_metrics['mae']:.3f}")
        print("=" * 60)

        # Print top candidates
        print(f"\nTOP CANDIDATES (score, middle half of runs, chance of top {results.top_k})\n")

        for i, candidate in enumerate(results.candidates[:15], start=1):
            ratings_note = f"({candidate.num_ratings} ratings)"
            if candidate.num_ratings <= 2:
                ratings_note += " *"
            print(f"{i:3}. {candidate.name:36} {candidate.avg_rating:5.2f}  "
                  f"{candidate.min_rating:4.1f}–{candidate.max_rating:4.1f}  "
                  f"{candidate.top_k_chance:4.0%}  {ratings_note}")

        print("\n  * two or fewer actual ratings; the score rests mostly on predictions")

        if len(results.candidates) > 15:
            print(f"\n... and {len(results.candidates) - 15} more candidates")

        print("\n" + "=" * 60)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
