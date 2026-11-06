#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathJobs Rankings - Faculty Hiring Application Processor

Ranks candidates by an adjusted mean of their faculty ratings:

1. Load this year's faculty ratings of PageUp applicants.
2. Fit the standard additive model for ratings from several raters,
       rating = candidate level + rater offset + noise,
   with candidate levels as fixed effects and rater offsets as random
   effects (a linear mixed model). The variances are estimated by REML,
   using the EM algorithm on Henderson's mixed model equations.
3. Score each candidate by their estimated level: the average of their
   ratings after subtracting each rater's estimated offset. Offsets are
   measured against the average rater, so scores can sit above or below the
   plain means; only differences between candidates matter.
4. Mark the top places (default 20), which require a minimum number of
   ratings (default 3), and how many more reads would let a candidate with
   fewer ratings compete for one.

Input files (from MathJobs export):
- ratings.csv: Faculty ratings of candidates
- appl.csv: Applicant information (filtered to PageUp candidates)

Output:
- mathjobs ranking YYYY-MM-DD.xlsx

Based on script used for SRS by Ken Richardson.

Author: José Carrión (j.carrion@tcu.edu)
Original: 2022-09-22
Refactored: 2025-11-26
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from openpyxl.styles import Border, PatternFill, Side


# =============================================================================
# Constants
# =============================================================================

RATING_SCALE = (1, 10)  # MathJobs rating scale
MIN_RATINGS_TOP = 3  # Ratings needed to hold a top place
TOP_N = 20  # Number of top places
CV_FOLDS = 5  # Folds for the accuracy check
CV_SEED = 2026  # Fixed split for the accuracy check
REML_MAX_ITER = 20000  # EM iterations for the variance estimates
REML_TOL = 1e-8  # Relative change at which the EM iterations stop
VAR_FLOOR = 1e-6  # Smallest variance allowed, as a fraction of the variance of the ratings
BOUNDARY = 1e-3  # Rater variance below this fraction of the noise variance counts as zero

RATINGS_COLUMNS = ["Uid", "LastName", "FirstName", "FacultyUid", "Faculty", "Rating"]
APPLICANTS_COLUMNS = ["Uid", "LastName", "FirstName"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CandidateResult:
    """Results for a single scored candidate."""
    name: str
    uid: int
    score: float  # adjusted mean
    mean_rating: float  # plain mean of actual ratings (as in MathJobs)
    num_ratings: int  # actual ratings received (as in MathJobs)
    ratings: list  # the ratings themselves, highest first
    score_se: float  # standard error of the score
    top_place: bool = False  # holds one of the top places
    reads_needed: int = 0  # more reads needed to compete for a top place


@dataclass
class ModelFit:
    """Fitted additive model: rating = candidate level + rater offset + noise."""
    level: pd.Series  # candidate levels (the scores), by applicant Uid
    rater_offset: pd.Series  # by FacultyUid
    var_noise: float
    var_rater: float
    score_se: pd.Series  # standard error of each level, by Uid
    iterations: int
    converged: bool
    adjusted: bool = True  # False when no candidate has two ratings, so offsets can't be estimated


@dataclass
class ProcessingResults:
    """Container for all processing results."""
    candidates: list  # List of CandidateResult, best first
    not_scored: list  # Names of PageUp applicants with no ratings
    fit: ModelFit
    algorithm_metrics: dict  # accuracy check
    precision: dict  # {number of ratings: typical 95% half-width of the score}
    min_ratings: int = MIN_RATINGS_TOP
    top_n: int = TOP_N
    line_score: float = float("nan")  # lowest score holding a top place
    num_raters: int = 0
    num_ratings: int = 0
    pageup_filter: str = ""
    output_path: Optional[Path] = None


# =============================================================================
# Data Loading
# =============================================================================

def build_full_name(row: pd.Series) -> str:
    """
    Build full name from first, middle, last name fields.

    MathJobs format: "LastName, FirstName MiddleName"
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


def short_name(row: pd.Series) -> str:
    """'LastName, FirstName' for display."""
    return f"{str(row['LastName']).strip()}, {str(row['FirstName']).strip()}"


def read_csv_checked(filepath: Path, required: list, label: str) -> pd.DataFrame:
    """Read a MathJobs export and check that it has the columns we need."""
    if not filepath.exists():
        raise FileNotFoundError(f"{label} file not found: {filepath}")

    df = pd.read_csv(filepath, header=0, sep=",")
    if df.empty:
        raise ValueError(f"{label} file is empty: {filepath}")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} file {filepath} is missing columns {missing}")
    return df


def load_ratings(filepath: Path) -> pd.DataFrame:
    """Load ratings.csv from MathJobs (one row per rater and applicant; Rating may be blank)."""
    df = read_csv_checked(filepath, RATINGS_COLUMNS, "Ratings")
    df = df.dropna(subset=["Uid", "FacultyUid"])
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    bad = df["Rating"].notna() & ~df["Rating"].between(*RATING_SCALE)
    if bad.any():
        raise ValueError(f"Ratings file {filepath} has {int(bad.sum())} rating(s) outside "
                         f"{RATING_SCALE[0]}-{RATING_SCALE[1]}")
    return df


def load_applicants(filepath: Path, pageup_only: bool = True) -> pd.DataFrame:
    """
    Load appl.csv from MathJobs.

    Args:
        filepath: Path to appl.csv
        pageup_only: If True, keep only applicants marked PageUp == "yes"

    Returns:
        DataFrame with one row per applicant Uid and a computed Name column
    """
    df = read_csv_checked(filepath, APPLICANTS_COLUMNS, "Applicants")
    df = df.dropna(subset=["Uid"])
    df["Name"] = df.apply(build_full_name, axis=1)

    if pageup_only and 'PageUp' in df.columns:
        df = df[df["PageUp"].astype(str).str.strip().str.lower() == "yes"]
        df.attrs["pageup_filter"] = f"applied ({len(df)} PageUp applicants)"
    else:
        df.attrs["pageup_filter"] = "not applied: appl.csv has no PageUp column, so all applicants are included"

    if df.empty:
        raise ValueError("No candidates found (check PageUp filter)")

    return df.drop_duplicates("Uid")


def select_ratings(ratings_df: pd.DataFrame, candidate_uids: set) -> pd.DataFrame:
    """
    Non-blank ratings of the given applicants, one per rater and applicant
    (duplicate rows are averaged), in a fixed order.

    Returns:
        DataFrame with columns FacultyUid, Uid, Rating
    """
    df = ratings_df[ratings_df["Uid"].isin(candidate_uids) & ratings_df["Rating"].notna()]
    df = pd.DataFrame({
        "FacultyUid": df["FacultyUid"].astype(int),
        "Uid": df["Uid"].astype(int),
        "Rating": df["Rating"].astype(float),
    })
    df = df.groupby(["FacultyUid", "Uid"], as_index=False)["Rating"].mean()
    return df.sort_values(["FacultyUid", "Uid"]).reset_index(drop=True)


# =============================================================================
# The model
# =============================================================================

def fit_model(df: pd.DataFrame, max_iter: int = REML_MAX_ITER, tol: float = REML_TOL) -> ModelFit:
    """
    Fit rating = candidate level + rater offset + noise, with candidate levels
    as fixed effects, rater offsets ~ N(0, var_rater) and noise ~ N(0, var_noise).

    The variances are estimated by REML with the EM algorithm on Henderson's
    mixed model equations. A candidate's level is the average of their
    ratings after subtracting each rater's estimated offset; the offsets come
    from comparing raters on the candidates they both rated, and an offset
    based on few ratings is pulled toward zero.

    Args:
        df: Ratings with columns FacultyUid, Uid, Rating (one per rater and applicant)
    """
    if len(df) < 1:
        raise ValueError("No ratings to fit.")

    cands = np.sort(df["Uid"].unique())
    raters = np.sort(df["FacultyUid"].unique())
    ci = np.searchsorted(cands, df["Uid"].to_numpy())
    ri = np.searchsorted(raters, df["FacultyUid"].to_numpy())
    y = df["Rating"].to_numpy(dtype=float)
    n_obs, n_c, n_r = len(y), len(cands), len(raters)
    resid_df = n_obs - n_c  # degrees of freedom within candidates

    if resid_df < 1 or n_r < 2:
        # No candidate has two ratings (or there is one rater): raters can't
        # be compared, so no adjustment
        return ModelFit(
            level=pd.Series(df.groupby("Uid")["Rating"].mean().reindex(cands).to_numpy(), index=cands),
            rater_offset=pd.Series(0.0, index=raters),
            var_noise=float("nan"), var_rater=0.0,
            score_se=pd.Series(float("nan"), index=cands),
            iterations=0, converged=True, adjusted=False,
        )

    # Henderson's mixed model equations, unknowns [candidate levels, rater offsets]:
    #   [ D   B          ] [b]   [sum_c]
    #   [ B'  diag(m)+kI ] [l] = [sum_r]
    # with D = diag(ratings per candidate), B = candidate x rater incidence,
    # m = ratings per rater, k = var_noise / var_rater. D is diagonal, so the
    # system is solved through the R x R Schur complement S = diag(m)+kI - B'D^-1 B,
    # whose inverse is the rater block of the inverse.
    cnt_c = np.bincount(ci, minlength=n_c).astype(float)
    cnt_r = np.bincount(ri, minlength=n_r).astype(float)
    B = np.zeros((n_c, n_r))
    np.add.at(B, (ci, ri), 1.0)
    sum_c = np.bincount(ci, weights=y, minlength=n_c)
    sum_r = np.bincount(ri, weights=y, minlength=n_r)
    yty = float(y @ y)
    U = B / cnt_c[:, None]  # D^-1 B
    BtDinvB = B.T @ U
    rhs_l = sum_r - U.T @ sum_c

    total_var = float(np.var(y)) or 1.0
    floor = VAR_FLOOR * total_var
    var_e, var_l = total_var / 2, total_var / 10

    def solve(var_e, var_l):
        Sinv = np.linalg.inv(np.diag(cnt_r + var_e / var_l) - BtDinvB)
        l = Sinv @ rhs_l
        b = (sum_c - B @ l) / cnt_c
        return Sinv, b, l

    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        Sinv, b, l = solve(var_e, var_l)
        new_e = max((yty - float(b @ sum_c) - float(l @ sum_r)) / resid_df, floor)
        new_l = max((float(l @ l) + var_e * float(np.trace(Sinv))) / n_r, floor)
        change = max(abs(new_e - var_e) / var_e, abs(new_l - var_l) / var_l)
        var_e, var_l = new_e, new_l
        if change < tol:
            converged = True
            break
        if var_l < BOUNDARY * var_e:
            # The estimate is heading to zero (raters don't differ), where EM
            # only creeps; offsets this small are negligible, so set it to zero.
            var_l = floor
            converged = True
            break

    Sinv, b, l = solve(var_e, var_l)
    # Prediction error variance of each level: var_noise * diag(D^-1 + U S^-1 U')
    pev = var_e * (1.0 / cnt_c + np.einsum("ij,jk,ik->i", U, Sinv, U))

    return ModelFit(
        level=pd.Series(b, index=cands),
        rater_offset=pd.Series(l, index=raters),
        var_noise=var_e,
        var_rater=var_l,
        score_se=pd.Series(np.sqrt(np.maximum(pev, 0.0)), index=cands),
        iterations=iterations,
        converged=converged,
    )


def predict(fit: ModelFit, df: pd.DataFrame) -> np.ndarray:
    """Predicted ratings for (FacultyUid, Uid) pairs of candidates in the fit; unknown raters count as average."""
    return (df["Uid"].map(fit.level).to_numpy()
            + df["FacultyUid"].map(fit.rater_offset).fillna(0.0).to_numpy())


def evaluate(df: pd.DataFrame, folds: int = CV_FOLDS, seed: int = CV_SEED) -> dict:
    """
    Cross-validated accuracy: hide part of the ratings, predict them from the
    rest, and compare with the plain mean of the candidate's other ratings.
    Only hidden ratings of candidates with another rating are counted, since
    for the others the plain mean has nothing to go on.
    """
    order = np.random.default_rng(seed).permutation(len(df))
    model_err, mean_err = [], []
    for k in range(folds):
        test = df.iloc[order[k::folds]]
        train = df.drop(index=test.index)
        if len(train) < 2:
            continue
        test = test[test["Uid"].isin(train["Uid"])]
        if test.empty:
            continue
        fit = fit_model(train)
        y = test["Rating"].to_numpy()
        model_err.append(predict(fit, test) - y)
        mean_err.append(test["Uid"].map(train.groupby("Uid")["Rating"].mean()).to_numpy() - y)
    if not model_err:
        return {}
    m, b = np.concatenate(model_err), np.concatenate(mean_err)
    return {
        "mae": float(np.mean(np.abs(m))), "rmse": float(np.sqrt(np.mean(m ** 2))),
        "mean_mae": float(np.mean(np.abs(b))), "mean_rmse": float(np.sqrt(np.mean(b ** 2))),
        "n": int(len(m)),
    }


# =============================================================================
# Excel Export
# =============================================================================

def unique_output_path(output_dir: Path, stem: str) -> Path:
    """mathjobs ranking DATE.xlsx, or ... (2).xlsx etc. if that name is taken."""
    path = output_dir / f"{stem}.xlsx"
    n = 2
    while path.exists():
        path = output_dir / f"{stem} ({n}).xlsx"
        n += 1
    return path


def format_ratings(ratings: list) -> str:
    """'9, 8, 5' (one decimal only where a rating isn't whole)."""
    return ", ".join(f"{r:g}" if float(r).is_integer() else f"{r:.1f}" for r in ratings)


def precision_text(precision: dict) -> str:
    """'±1.6 with 1 rating, ±1.2 with 2, ...'"""
    return ", ".join(f"±{w:.1f} with {k}" + (" rating" if i == 0 else "")
                     for i, (k, w) in enumerate(sorted(precision.items())))


def create_results_excel(results: ProcessingResults, output_path: Path) -> None:
    """
    Create Excel file with ranking results.

    Sheets:
    - Ranking: Scored candidates, best first
    - Not scored: PageUp applicants with no ratings yet
    - How to read: Short explanation for a meeting
    - Metrics: Accuracy and settings
    """
    red = PatternFill("solid", start_color="ffcccc")
    green = PatternFill("solid", start_color="c6efce")
    k, top = results.min_ratings, results.top_n

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Ranking
        rankings_df = pd.DataFrame(
            [[i, c.name, round(c.score, 2), round(c.mean_rating, 2), c.num_ratings, format_ratings(c.ratings)]
             for i, c in enumerate(results.candidates, start=1)],
            columns=["Rank", "Candidate", "Score", "Mean rating", "# Ratings", "Ratings"],
        )
        rankings_df.to_excel(writer, sheet_name="Ranking", index=False)
        sheet = writer.sheets["Ranking"]
        sheet.freeze_panes = "A2"
        for col, width in zip("ABCDEF", (6, 34, 8, 12, 10, 18)):
            sheet.column_dimensions[col].width = width
        for row_idx, c in enumerate(results.candidates, start=2):
            if c.top_place:
                for col in (1, 2, 3):
                    sheet.cell(row=row_idx, column=col).fill = green
            if c.num_ratings < k:
                sheet.cell(row=row_idx, column=5).fill = red
        # A line under the last row at or above the top-place cutoff
        at_or_above = [i for i, c in enumerate(results.candidates) if c.score >= results.line_score - 1e-9]
        if at_or_above:
            cut_row = at_or_above[-1] + 2
            for col in range(1, 7):
                sheet.cell(row=cut_row, column=col).border = Border(bottom=Side(style="medium"))

        # Not scored
        pd.DataFrame({"Candidate": results.not_scored}).to_excel(writer, sheet_name="Not scored", index=False)
        writer.sheets["Not scored"].column_dimensions["A"].width = 34

        # How to read
        notes = [
            ["How to read the Ranking sheet"],
            ["Score: the average of the candidate's ratings, each adjusted for how high or low that rater tends to rate. "
             "Only differences between scores matter."],
            ["Mean rating: the plain average, as in MathJobs."],
            [f"# Ratings, Ratings: red means fewer than {k}."],
            [f"Green: top-{top} places, which need at least {k} ratings. The line marks the cutoff; "
             "a red count above it means the candidate needs more ratings to compete for a place."],
            [f"Precision (95%): {precision_text(results.precision)}."],
            ["Not scored: PageUp applicants with no ratings yet."],
        ]
        pd.DataFrame(notes).to_excel(writer, sheet_name="How to read", header=None, index=False)
        writer.sheets["How to read"].column_dimensions["A"].width = 140

        # Metrics
        m, fit = results.algorithm_metrics, results.fit
        metrics_data = [["Metric", "Value"]]
        if m:
            metrics_data += [
                ["Typical error predicting a hidden rating (MAE), adjusted mean", round(m["mae"], 3)],
                ["Typical error predicting a hidden rating (MAE), plain mean", round(m["mean_mae"], 3)],
                ["Root-mean-square error, adjusted mean", round(m["rmse"], 3)],
                ["Root-mean-square error, plain mean", round(m["mean_rmse"], 3)],
                ["Note", f"{CV_FOLDS}-fold check on this year's ratings; counts the {m['n']} hidden ratings "
                         "of candidates with another rating. Lower is better."],
            ]
        metrics_data += [
            ["", ""],
            ["Rater disagreement (SD of one rating around the candidate's level)", round(float(np.sqrt(fit.var_noise)), 3)],
            ["Spread of rater tendencies (SD)", round(float(np.sqrt(fit.var_rater)), 3)],
            ["Ratings adjusted for rater tendencies", "yes" if fit.adjusted else "NO: no candidate has two ratings yet"],
            ["Model fit converged", f"{'yes' if fit.converged else 'NO'} ({fit.iterations} iterations)"],
            ["", ""],
            ["Ratings used", results.num_ratings],
            ["Raters", results.num_raters],
            ["Candidates scored", len(results.candidates)],
            ["Ratings needed for a top place", k],
            ["Top places", top],
            ["PageUp filter", results.pageup_filter],
        ]
        pd.DataFrame(metrics_data).to_excel(writer, sheet_name="Metrics", header=None, index=False)
        writer.sheets["Metrics"].column_dimensions["A"].width = 66
        writer.sheets["Metrics"].column_dimensions["B"].width = 90


# =============================================================================
# Main Processing
# =============================================================================

def process_mathjobs_data(
    ratings_path: Path,
    applicants_path: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    min_ratings: int = MIN_RATINGS_TOP,
    top_n: int = TOP_N,
) -> ProcessingResults:
    """
    Process MathJobs data and produce rankings.

    Args:
        ratings_path: Path to current ratings.csv
        applicants_path: Path to current appl.csv
        output_dir: Directory for output file (default: same as ratings_path)
        verbose: Print progress information
        min_ratings: Ratings needed to hold a top place
        top_n: Number of top places

    Returns:
        ProcessingResults object
    """
    if output_dir is None:
        output_dir = ratings_path.parent
    if min_ratings < 1 or top_n < 1:
        raise ValueError("The minimum number of ratings and the number of top places must be at least 1.")

    if verbose:
        print("Loading data...")

    ratings_df = load_ratings(ratings_path)
    applicants_df = load_applicants(applicants_path)
    pageup_filter = applicants_df.attrs.get("pageup_filter", "")
    if verbose or "not applied" in pageup_filter:
        print(f"PageUp filter: {pageup_filter}")
    pageup_uids = set(int(u) for u in applicants_df["Uid"])
    names = {int(row["Uid"]): short_name(row) for _, row in applicants_df.iterrows()}
    taken = pd.Series(list(names.values())).value_counts()
    names = {u: (f"{nm} (ID {u})" if taken[nm] > 1 else nm) for u, nm in names.items()}

    data = select_ratings(ratings_df, pageup_uids)
    if len(data) < 2:
        raise ValueError("Fewer than two ratings of PageUp applicants in this export.")

    if verbose:
        print(f"  {len(pageup_uids)} applicants; {data['Uid'].nunique()} rated; "
              f"{data['FacultyUid'].nunique()} raters; {len(data)} ratings")
        print("Fitting the model...")

    fit = fit_model(data)
    if verbose:
        print(f"  {'Converged' if fit.converged else 'Did NOT converge'} after {fit.iterations} iterations")
        print("Checking accuracy...")
    metrics = evaluate(data)

    by_cand = data.groupby("Uid")["Rating"]
    counts = by_cand.size()
    means = by_cand.mean()
    rating_lists = {uid: sorted(s.tolist(), reverse=True) for uid, s in by_cand}
    score = fit.level

    candidates = [
        CandidateResult(
            name=names.get(int(uid), f"Unknown (UID: {uid})"),
            uid=int(uid),
            score=float(score[uid]),
            mean_rating=float(means[uid]),
            num_ratings=int(counts[uid]),
            ratings=rating_lists[uid],
            score_se=float(fit.score_se[uid]),
        )
        for uid in score.index
    ]
    candidates.sort(key=lambda c: (-round(c.score, 9), -c.mean_rating, c.name))

    # Top places: the top_n highest scores among candidates with at least
    # min_ratings ratings (all of them if several tie for the last place).
    # While fewer than top_n candidates qualify, the line is the top_n-th
    # highest score overall.
    qualified = [c for c in candidates if c.num_ratings >= min_ratings]
    if len(qualified) >= top_n:
        line = qualified[top_n - 1].score
    else:
        line = candidates[min(top_n, len(candidates)) - 1].score
    eps = 1e-9
    for c in qualified:
        c.top_place = c.score >= line - eps
    for c in candidates:
        if c.num_ratings < min_ratings and c.score >= line - eps:
            c.reads_needed = min_ratings - c.num_ratings

    # Typical precision by number of ratings
    precision = {}
    for k in range(1, 6):
        se = [c.score_se for c in candidates if c.num_ratings == k and np.isfinite(c.score_se)]
        if se:
            precision[k] = 1.96 * float(np.median(se))

    not_scored = sorted(names[u] for u in pageup_uids - set(int(u) for u in score.index))

    results = ProcessingResults(
        candidates=candidates,
        not_scored=not_scored,
        fit=fit,
        algorithm_metrics=metrics,
        precision=precision,
        min_ratings=min_ratings,
        top_n=top_n,
        line_score=line,
        num_raters=int(data["FacultyUid"].nunique()),
        num_ratings=len(data),
        pageup_filter=pageup_filter,
    )

    output_path = unique_output_path(output_dir, f"mathjobs ranking {time.strftime('%Y-%m-%d')}")
    if verbose:
        print(f"Saving results to {output_path}...")
    create_results_excel(results, output_path)
    results.output_path = output_path

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
    python mathjobs_rankings_new.py --top 15 --min-ratings 3
        """
    )
    parser.add_argument("--ratings", "-r", type=Path, default=Path("ratings.csv"),
                        help="Path to ratings CSV (default: ratings.csv)")
    parser.add_argument("--applicants", "-a", type=Path, default=Path("appl.csv"),
                        help="Path to applicants CSV (default: appl.csv)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: same as ratings file)")
    parser.add_argument("--top", type=int, default=TOP_N,
                        help=f"Number of top places (default: {TOP_N})")
    parser.add_argument("--min-ratings", type=int, default=MIN_RATINGS_TOP,
                        help=f"Ratings needed to hold a top place (default: {MIN_RATINGS_TOP})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress information")
    args = parser.parse_args()

    try:
        results = process_mathjobs_data(
            ratings_path=args.ratings,
            applicants_path=args.applicants,
            output_dir=args.output,
            verbose=args.verbose,
            min_ratings=args.min_ratings,
            top_n=args.top,
        )

        m = results.algorithm_metrics
        print("\n" + "=" * 70)
        if m:
            print(f"Typical error predicting a hidden rating: {m['mae']:.2f} (plain mean: {m['mean_mae']:.2f})")
        if not results.fit.converged:
            print("WARNING: the model fit did not converge; treat the scores with caution.")
        if not results.fit.adjusted:
            print("WARNING: no candidate has two ratings yet, so scores are plain means (no rater adjustment).")
        print(f"Saved: {results.output_path}")
        print("=" * 70)

        top, k = results.top_n, results.min_ratings
        print(f"\nRANKING (score, mean rating, ratings). # = holds a top-{top} place; "
              f"+n = needs n more reads to compete for one\n")
        last = max([i for i, c in enumerate(results.candidates) if c.top_place or c.reads_needed] + [top - 1])
        for i, c in enumerate(results.candidates[:last + 1], start=1):
            mark = "#" if c.top_place else (f"+{c.reads_needed}" if c.reads_needed else "")
            print(f"{i:3}. {mark:>3} {c.name:36} {c.score:5.2f}  {c.mean_rating:5.2f}  ({format_ratings(c.ratings)})")

        needing = [c for c in results.candidates if c.reads_needed]
        print(f"\n{len(needing)} candidates with fewer than {k} ratings score above the top-{top} line "
              f"({sum(c.reads_needed for c in needing)} reads in all)")
        print(f"Precision (95%): {precision_text(results.precision)}")
        if results.not_scored:
            print(f"{len(results.not_scored)} PageUp applicants not rated yet (see the 'Not scored' sheet)")
        print("=" * 70)

    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
