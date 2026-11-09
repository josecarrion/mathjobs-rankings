#!/usr/bin/env python3
"""
MathJobs Rankings - Desktop Application

A graphical interface that ranks MathJobs candidates by an adjusted
mean of the faculty ratings (each rating adjusted for how high or low
that rater tends to rate).

Usage:
    python mathjobs_app.py

Or double-click the file on Windows.

Requirements:
    pip install "flet==1.1.2" pandas numpy openpyxl
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import flet as ft

# Import the processing logic from our refactored script
from mathjobs_rankings_new import process_mathjobs_data, format_ratings


def main(page: ft.Page):
    """Main application entry point."""

    # Page configuration
    page.title = "MathJobs Rankings"
    page.window.width = 850
    page.window.height = 750
    page.padding = 30
    page.theme_mode = ft.ThemeMode.LIGHT

    # State
    ratings_file = None
    applicants_file = None
    output_dir = None

    # --- UI Components ---

    # Title
    title = ft.Text(
        "MathJobs Rankings",
        size=32,
        weight=ft.FontWeight.BOLD,
    )

    subtitle = ft.Text(
        "Rank candidates from faculty ratings",
        size=16,
        color=ft.Colors.GREY_700,
    )

    # File picker (a service in Flet 1.x: it registers itself when created;
    # keep the reference so it stays registered)
    file_picker = ft.FilePicker()

    # File path displays
    ratings_path_text = ft.Text("No file selected", size=12, color=ft.Colors.GREY_600, width=450)
    applicants_path_text = ft.Text("No file selected", size=12, color=ft.Colors.GREY_600, width=450)

    async def pick_csv(title: str):
        """Open the file dialog for a CSV file; return its path, or None if cancelled."""
        files = await file_picker.pick_files(
            dialog_title=title,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["csv"],
        )
        return files[0].path if files and files[0].path else None

    async def pick_ratings(_):
        nonlocal ratings_file, output_dir
        path = await pick_csv("Select ratings.csv")
        if path:
            ratings_file = path
            output_dir = str(Path(path).parent)
            ratings_path_text.value = path
            ratings_path_text.color = ft.Colors.BLACK
            update_run_button()
            page.update()

    async def pick_applicants(_):
        nonlocal applicants_file
        path = await pick_csv("Select appl.csv")
        if path:
            applicants_file = path
            applicants_path_text.value = path
            applicants_path_text.color = ft.Colors.BLACK
            update_run_button()
            page.update()

    def update_run_button():
        run_button.disabled = not (ratings_file and applicants_file)

    # File selection buttons
    ratings_button = ft.Button(
        content="Select ratings.csv",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=pick_ratings,
        width=200,
    )

    applicants_button = ft.Button(
        content="Select appl.csv",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=pick_applicants,
        width=200,
    )

    # Progress/status area
    status_text = ft.Text(
        "",
        size=14,
        color=ft.Colors.GREY_700,
    )

    progress_bar = ft.ProgressBar(
        visible=False,
        width=400,
    )

    # Results area
    results_container = ft.Container(
        visible=False,
        padding=20,
        border_radius=10,
        bgcolor=ft.Colors.GREY_100,
        content=ft.Column([]),
    )

    async def process_ratings(_=None):
        """Run the processing in a background thread and update the UI."""
        nonlocal ratings_file, applicants_file, output_dir

        if not ratings_file or not applicants_file:
            return

        # Show progress
        progress_bar.visible = True
        progress_bar.value = None  # Indeterminate
        status_text.value = "Loading data..."
        status_text.color = ft.Colors.BLUE_700
        run_button.disabled = True
        results_container.visible = False
        page.update()

        try:
            # Process
            status_text.value = "Fitting the model..."
            page.update()

            # Off the event loop, so the window keeps responding
            results = await asyncio.to_thread(
                process_mathjobs_data,
                ratings_path=Path(ratings_file),
                applicants_path=Path(applicants_file),
                output_dir=Path(output_dir),
                verbose=False,
            )
            top, k = results.top_n, results.min_ratings

            # Show results
            progress_bar.visible = False

            # Build results display
            results_rows = []

            results_rows.append(
                ft.Container(
                    padding=10,
                    border_radius=5,
                    bgcolor=ft.Colors.GREEN_100,
                    content=ft.Row([
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN_700),
                        ft.Text(
                            f"Ranked {len(results.candidates)} candidates",
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.GREEN_900,
                        ),
                    ]),
                )
            )

            # Warnings
            warnings = []
            if "not applied" in results.pageup_filter:
                warnings.append(f"PageUp filter {results.pageup_filter}")
            if not results.fit.converged:
                warnings.append("the model fit did not converge; treat the scores with caution")
            if not results.fit.adjusted:
                warnings.append("no candidate has two ratings yet, so scores are plain means (no rater adjustment)")
            for w in warnings:
                results_rows.append(ft.Text(f"Warning: {w}", size=12, color=ft.Colors.ORANGE_700))

            # Accuracy
            m = results.algorithm_metrics
            if m:
                results_rows.append(
                    ft.Text(
                        f"Typical error predicting a hidden rating: {m['mae']:.2f} (plain mean: {m['mean_mae']:.2f})",
                        size=12,
                        color=ft.Colors.GREY_600,
                    )
                )

            results_rows.append(ft.Divider())
            results_rows.append(ft.Text("Top of the ranking:", weight=ft.FontWeight.BOLD, size=16))
            results_rows.append(
                ft.Row([
                    ft.Text("", width=35),
                    ft.Text("", width=30),
                    ft.Text("", width=280),
                    ft.Text("Score", width=55, color=ft.Colors.GREY_600),
                    ft.Text("Mean", width=55, color=ft.Colors.GREY_600),
                    ft.Text("Ratings", color=ft.Colors.GREY_600),
                ])
            )

            # Rows down to the last top place or candidate needing reads
            marked = [i for i, c in enumerate(results.candidates) if c.top_place or c.reads_needed]
            last = max(marked + [min(top, len(results.candidates)) - 1])
            for i, c in enumerate(results.candidates[:last + 1], start=1):
                mark = "#" if c.top_place else (f"+{c.reads_needed}" if c.reads_needed else "")
                results_rows.append(
                    ft.Row([
                        ft.Text(f"{i}.", width=35, weight=ft.FontWeight.BOLD),
                        ft.Text(mark, width=30, color=ft.Colors.GREEN_800 if c.top_place else ft.Colors.ORANGE_700),
                        ft.Text(c.name, width=280),
                        ft.Text(f"{c.score:.2f}", width=55),
                        ft.Text(f"{c.mean_rating:.2f}", width=55),
                        ft.Text(format_ratings(c.ratings), color=ft.Colors.GREY_600),
                    ])
                )

            needing = [c for c in results.candidates if c.reads_needed]
            summary = [
                f"{len(needing)} candidates with fewer than {k} ratings score above the top-{top} line "
                f"({sum(c.reads_needed for c in needing)} reads in all)",
            ]
            if results.not_scored:
                summary.append(f"{len(results.not_scored)} PageUp applicants not rated yet (see the 'Not scored' sheet)")
            for line in summary:
                results_rows.append(ft.Text(line, size=12, color=ft.Colors.GREY_600, italic=True))

            results_rows.append(ft.Divider())

            # Output file
            excel_name = results.output_path.name
            results_rows.append(ft.Text("Output file:", weight=ft.FontWeight.BOLD))
            results_rows.append(ft.Text(f"  {excel_name}", size=12))

            # Open folder button
            def open_output_folder(_):
                if sys.platform == "win32":
                    os.startfile(output_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", output_dir])
                else:
                    subprocess.run(["xdg-open", output_dir])

            results_rows.append(ft.Container(height=10))
            results_rows.append(
                ft.Button(
                    content="Open Output Folder",
                    icon=ft.Icons.FOLDER,
                    on_click=open_output_folder,
                )
            )

            results_container.content = ft.Column(results_rows, spacing=5)
            results_container.visible = True

            status_text.value = "Done!"
            status_text.color = ft.Colors.GREEN_700

        except Exception as e:
            progress_bar.visible = False
            status_text.value = f"Error: {str(e)}"
            status_text.color = ft.Colors.RED_700
            results_container.visible = False

        finally:
            run_button.disabled = False
            page.update()

    run_button = ft.Button(
        content="Process Ratings",
        icon=ft.Icons.PLAY_ARROW,
        disabled=True,
        style=ft.ButtonStyle(
            bgcolor={
                ft.ControlState.DEFAULT: ft.Colors.BLUE_700,
                ft.ControlState.DISABLED: ft.Colors.GREY_400,
            },
            color={
                ft.ControlState.DEFAULT: ft.Colors.WHITE,
                ft.ControlState.DISABLED: ft.Colors.GREY_600,
            },
        ),
        on_click=process_ratings,
    )

    # Help text
    help_text = ft.Container(
        padding=15,
        border_radius=10,
        bgcolor=ft.Colors.BLUE_50,
        content=ft.Column([
            ft.Text("How to use:", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_900),
            ft.Text(
                "1. Export ratings.csv and appl.csv from MathJobs\n"
                "2. Select both files above\n"
                "3. Click 'Process Ratings' to generate rankings\n"
                "4. Results are saved as an Excel file (see its 'How to read' sheet)",
                size=13,
                color=ft.Colors.BLUE_800,
            ),
            ft.Container(height=5),
            ft.Text(
                "# = holds a top-20 place (needs at least 3 ratings); +n = needs n more reads to compete for one",
                size=11,
                color=ft.Colors.ORANGE_700,
            ),
        ], spacing=5),
    )

    # --- Layout ---

    page.add(
        ft.Column([
            title,
            subtitle,
            ft.Container(height=20),

            # Required files section
            ft.Text("Required Files:", weight=ft.FontWeight.BOLD, size=14),
            ft.Row([ratings_button, ratings_path_text], spacing=15),
            ft.Row([applicants_button, applicants_path_text], spacing=15),

            ft.Container(height=20),

            # Run button and status
            ft.Row([
                run_button,
                status_text,
            ], spacing=20),

            progress_bar,

            ft.Container(height=20),

            # Results
            results_container,

            ft.Container(height=20),

            # Help
            help_text,

        ], spacing=8, scroll=ft.ScrollMode.AUTO, expand=True)
    )


if __name__ == "__main__":
    ft.run(main)
