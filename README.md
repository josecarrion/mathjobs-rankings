# MathJobs Rankings

Rank faculty job candidates from MathJobs ratings, adjusting for raters who tend to rate high or low.

## What It Does

1. **Loads faculty ratings** of PageUp applicants from the MathJobs export files
2. **Adjusts for tough and easy raters**: each rater's tendency is estimated by comparing their ratings with other raters' ratings of the same candidates
3. **Scores each candidate** by the average of their adjusted ratings, next to the plain mean rating that MathJobs shows
4. **Marks the top 20 places**, which need at least 3 ratings

## Quick Start

### Using the Desktop App

1. Export `ratings.csv` and `appl.csv` from MathJobs
2. Run `MathJobsRankings.exe` (or `python mathjobs_app.py`)
3. Select both files
4. Click "Process Ratings"
5. Results are saved as `mathjobs ranking YYYY-MM-DD.xlsx` next to `ratings.csv`

### Using the Command Line

```bash
python mathjobs_rankings_new.py --verbose
python mathjobs_rankings_new.py --ratings my_ratings.csv --applicants my_appl.csv
python mathjobs_rankings_new.py --top 15 --min-ratings 4   # 15 top places, 4 ratings needed
```

The same data always gives the same output. A second run on the same day saves `mathjobs ranking YYYY-MM-DD (2).xlsx` instead of overwriting the first.

## Input Files

From the current MathJobs export:
- `ratings.csv` - Faculty ratings of candidates
- `appl.csv` - Applicant information. Only applicants marked PageUp are ranked.

## Output

The Excel file has four sheets.

### Ranking Sheet

| Rank | Candidate | Score | Mean rating | # Ratings | Ratings |
|------|-----------|-------|-------------|-----------|---------|
| 1 | Smith, Jane | 9.39 | 9.00 | 5 | 10, 9, 9, 9, 8 |
| 2 | Doe, John | 9.31 | 9.00 | 2 | 9, 9 |

- **Score**: the average of the candidate's ratings, each adjusted for how high or low that rater tends to rate. Only differences between scores matter.
- **Mean rating**: the plain average, as in MathJobs.
- **# Ratings** and **Ratings**: how many ratings the candidate received, and the ratings themselves. Red means fewer than 3.
- **Green**: top-20 places, which need at least 3 ratings. A line marks the cutoff; a red count above it means the candidate needs more ratings to compete for a place.

### Not scored Sheet
PageUp applicants with no ratings yet.

### How to read Sheet
A short explanation of the columns for sharing at a meeting, including how precise a score is with 1, 2, 3, ... ratings.

### Metrics Sheet
How well the scores predict ratings they didn't see, compared with the plain mean; how much raters disagree and how much their tendencies differ; and the settings used.

## How It Works

Each rating is treated as

    rating = candidate's level + rater's offset + noise

This is the standard additive model for scores from several judges: a linear mixed model with candidates as fixed effects and raters as random effects. A rater's offset is estimated by comparing their ratings with other raters' ratings of the same candidates; an offset based on few ratings is pulled toward zero. The variances are estimated by REML. A candidate's score is their estimated level: the average of their ratings after subtracting each rater's offset.

A score based on one or two ratings can be off by more than a point, so a top place needs a minimum number of ratings. The How to read sheet gives the precision for the current data.

## Requirements

**For the executable:** None (everything bundled)

**For Python scripts:**
- Python 3.9+
- pandas, numpy, openpyxl
- flet (for GUI)

Install with:
```bash
pip install pandas numpy openpyxl flet
```

## Files

- `mathjobs_rankings_new.py` - Core processing logic
- `mathjobs_app.py` - Desktop GUI
- `build_exe.bat` - Build Windows executable

## Building the Executable

```bash
# On Windows, double-click build_exe.bat
# Or run manually:
pip install pyinstaller flet pandas numpy openpyxl
python -m PyInstaller --onefile --windowed --name "MathJobsRankings" ...
```

## License

MIT. See `LICENSE`.
