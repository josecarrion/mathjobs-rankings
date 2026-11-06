# MathJobs Rankings

Process faculty ratings from MathJobs to produce normalized candidate rankings using collaborative filtering.

## What It Does

1. **Loads faculty ratings** from MathJobs export files
2. **Predicts missing ratings** using SVD++ collaborative filtering (faculty don't rate every candidate, so the algorithm predicts what they would have rated based on patterns)
3. **Normalizes ratings** across faculty (adjusts for tough/easy graders), using each rater's actual ratings
4. **Re-scores on resampled ratings** to give each candidate a score, the range covering the middle half of runs, and the chance of landing in the top 15

## Quick Start

### Using the Desktop App

1. Export `ratings.csv` and `appl.csv` from MathJobs
2. Run `MathJobsRankings.exe` (or `python mathjobs_app.py`)
3. Select both required files
4. (Optional) Select a training folder with historical data
5. Click "Process Ratings"
6. Results are saved as `mathjobs ranking YYYY-MM-DD.xlsx`

### Using the Command Line

```bash
python mathjobs_rankings_new.py --verbose
python mathjobs_rankings_new.py --ratings my_ratings.csv --applicants my_appl.csv
python mathjobs_rankings_new.py --training-folder ./training
python mathjobs_rankings_new.py --no-training  # Skip historical data
python mathjobs_rankings_new.py --top-k 20   # move the "chance of top K" line
python mathjobs_rankings_new.py --runs 100   # more resampled runs
python mathjobs_rankings_new.py --seed 7     # different seed (same data + same seed = same output)
```

## Input Files

**Required (from current MathJobs export):**
- `ratings.csv` - Faculty ratings of candidates
- `appl.csv` - Applicant information

**Optional (past searches, for training):**

A `training` folder with one subfolder per past search, each holding an anonymized `ratings.csv`:

```
training/
    2022/
        ratings.csv
    2024/
        ratings.csv
```

Make each file from that search's final MathJobs ratings export:

```bash
python anonymize_training.py ratings.csv -o training/2026
```

The anonymized file keeps only the rater's MathJobs ID, a random applicant ID, and the rating. Past searches help the model learn how each rater tends to score, which matters most early in a search, when few ratings are in. Keep training data private; don't commit it.

## Output

The Excel file contains four sheets:

### Ranking Sheet
| Rank | Candidate | Score | Middle half of runs | Chance of top 15 | # Ratings |
|------|-----------|-------|---------------------|------------------|-----------|
| 1 | Smith, Jane | 8.07 | 7.7–8.3 | 96% | 5 |

- **Score**: the candidate's typical committee rating on the 1–10 scale, adjusted for tough and easy raters.
- **Middle half of runs**: half of the resampled runs gave a score in this range.
- **Chance of top 15**: how often the candidate ranked in the top 15 over those runs. Green is 75% or more; yellow is 25–74%.
- **# Ratings**: how many faculty actually rated the candidate. Red means two or fewer, so the score rests mostly on predictions.

### How to read Sheet
A short explanation of the columns, for sharing at a meeting.

### Predictions Sheet
Full matrix showing actual and predicted ratings for each candidate by each faculty member. Predicted ratings are used to fill gaps where a faculty member didn't rate a candidate.

### Metrics Sheet
Algorithm accuracy metrics:
- **RMSE**: Root Mean Square Error (lower = better)
- **MAE**: Mean Absolute Error (lower = better)
- **Training Data Years**: Which years of historical data were used (if any)
- Also lists the number of resampled runs, the top-K line, and the random seed.

## How the Algorithm Works

### Collaborative Filtering (SVD++)

Not every faculty member rates every candidate. Rather than just averaging available ratings (which could be biased), we use collaborative filtering:

1. Find patterns in the rating matrix (e.g., "Faculty A and B tend to agree")
2. Use these patterns to predict what Faculty A would have rated Candidate X, even if they didn't
3. Include predicted ratings when computing averages

### Normalization

Faculty members have different rating styles: some rate everyone 8–10, others 4–7. For each rater we estimate their usual rating and their spread from the ratings they actually gave. For raters with only a few ratings, those estimates are pulled toward the committee's overall values. Every rating, actual or predicted, is converted to "points above or below this rater's usual rating." These are averaged across raters and mapped back to the 1–10 scale, so a score reads like an ordinary rating.

### Score Ranges and Chance of Top K

One run gives one ranking, but small changes in who rated whom can reshuffle candidates with few ratings. So we re-score everyone 50 times. Each time we use a random resample of this year's ratings: some ratings drop out and others count twice, as if a slightly different set of people had read the files. The **score** is the median over those runs, the **middle half of runs** is the range holding half of them, and the **chance of top K** is how often the candidate made the top K. A fixed random seed means the same data always gives the same output.

## Requirements

**For the executable:** None (everything bundled)

**For Python scripts:**
- Python 3.9+
- pandas, numpy, scikit-surprise, openpyxl
- flet (for GUI)

Install with:
```bash
pip install pandas numpy scikit-surprise openpyxl flet
```

## Files

- `mathjobs_rankings_new.py` - Core processing logic
- `mathjobs_app.py` - Desktop GUI
- `build_exe.bat` - Build Windows executable
- `anonymize_training.py` - Makes anonymized training files from past searches

## Building the Executable

```bash
# On Windows, double-click build_exe.bat
# Or run manually:
pip install pyinstaller flet pandas numpy scikit-surprise openpyxl
python -m PyInstaller --onefile --windowed --name "MathJobsRankings" ...
```

## License

MIT. See `LICENSE`.
