#!/usr/bin/env python3
"""
MathJobs Rankings - Desktop Application

A graphical interface for processing MathJobs faculty ratings
using collaborative filtering to predict missing ratings and
produce normalized candidate rankings.

Usage:
    python mathjobs_app.py

Or double-click the file on Windows.

Requirements:
    pip install flet pandas numpy scikit-surprise openpyxl
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import flet as ft

# Import the processing logic from our refactored script
from mathjobs_rankings_new import (
    process_mathjobs_data,
    ProcessingResults,
    discover_training_data,
)


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
    training_folder = None
    output_dir = None

    # --- UI Components ---

    # Title
    title = ft.Text(
        "MathJobs Rankings",
        size=32,
        weight=ft.FontWeight.BOLD,
    )

    subtitle = ft.Text(
        "Process faculty ratings with collaborative filtering",
        size=16,
        color=ft.Colors.GREY_700,
    )

    # File selection components
    def create_file_picker(on_result, text_ref):
        def handle_result(e: ft.FilePickerResultEvent):
            on_result(e, text_ref)

        picker = ft.FilePicker(on_result=handle_result)
        page.overlay.append(picker)
        return picker

    # File path displays
    ratings_path_text = ft.Text("No file selected", size=12, color=ft.Colors.GREY_600, width=450)
    applicants_path_text = ft.Text("No file selected", size=12, color=ft.Colors.GREY_600, width=450)
    training_folder_text = ft.Text("(Optional) No folder selected", size=12, color=ft.Colors.GREY_500, width=450)

    def on_ratings_picked(e: ft.FilePickerResultEvent, text_ref):
        nonlocal ratings_file, output_dir
        if e.files and len(e.files) > 0:
            ratings_file = e.files[0].path
            output_dir = str(Path(ratings_file).parent)
            text_ref.value = ratings_file
            text_ref.color = ft.Colors.BLACK
            update_run_button()
        page.update()

    def on_applicants_picked(e: ft.FilePickerResultEvent, text_ref):
        nonlocal applicants_file
        if e.files and len(e.files) > 0:
            applicants_file = e.files[0].path
            text_ref.value = applicants_file
            text_ref.color = ft.Colors.BLACK
            update_run_button()
        page.update()

    def on_training_folder_picked(e: ft.FilePickerResultEvent, text_ref):
        nonlocal training_folder
        if e.path:
            training_folder = e.path
            # Check what years are available
            datasets = discover_training_data(Path(training_folder))
            if datasets:
                years = [d.year for d in datasets]
                text_ref.value = f"{training_folder} ({len(datasets)} years: {', '.join(years)})"
            else:
                text_ref.value = f"{training_folder} (no valid data found)"
            text_ref.color = ft.Colors.BLACK
        page.update()

    # Create file/folder pickers
    ratings_picker = create_file_picker(on_ratings_picked, ratings_path_text)
    applicants_picker = create_file_picker(on_applicants_picked, applicants_path_text)

    # Folder picker for training data
    def handle_folder_result(e: ft.FilePickerResultEvent):
        on_training_folder_picked(e, training_folder_text)

    training_folder_picker = ft.FilePicker(on_result=handle_folder_result)
    page.overlay.append(training_folder_picker)

    def update_run_button():
        run_button.disabled = not (ratings_file and applicants_file)

    # File selection buttons
    def pick_csv(picker):
        return lambda _: picker.pick_files(
            allowed_extensions=["csv"],
            dialog_title="Select CSV File",
        )

    ratings_button = ft.ElevatedButton(
        "Select ratings.csv",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=pick_csv(ratings_picker),
        width=200,
    )

    applicants_button = ft.ElevatedButton(
        "Select appl.csv",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=pick_csv(applicants_picker),
        width=200,
    )

    training_folder_button = ft.OutlinedButton(
        "Select training folder",
        icon=ft.Icons.FOLDER,
        on_click=lambda _: training_folder_picker.get_directory_path(
            dialog_title="Select Training Data Folder"
        ),
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

    def process_ratings():
        """Run the processing and update the UI."""
        nonlocal ratings_file, applicants_file, training_folder, output_dir

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
            status_text.value = "Training collaborative filtering model (this may take a moment)..."
            page.update()

            training_f = Path(training_folder) if training_folder else None

            results = process_mathjobs_data(
                ratings_path=Path(ratings_file),
                applicants_path=Path(applicants_file),
                training_folder=training_f,
                output_dir=Path(output_dir),
                verbose=False
            )

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
                            f"Processed {len(results.candidates)} candidates",
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.GREEN_900,
                        ),
                    ]),
                )
            )

            # Training data info
            if results.training_years:
                results_rows.append(
                    ft.Text(
                        f"Training data: {', '.join(results.training_years)}",
                        size=12,
                        color=ft.Colors.GREY_600,
                    )
                )

            # Algorithm metrics
            results_rows.append(
                ft.Text(
                    f"Model accuracy: RMSE={results.algorithm_metrics['rmse']:.3f}, MAE={results.algorithm_metrics['mae']:.3f}",
                    size=12,
                    color=ft.Colors.GREY_600,
                )
            )

            results_rows.append(ft.Divider())
            results_rows.append(ft.Text("Top Candidates:", weight=ft.FontWeight.BOLD, size=16))

            # Top 10 candidates
            for i, candidate in enumerate(results.candidates[:10], start=1):
                warning = " *" if candidate.num_ratings <= 2 else ""
                results_rows.append(
                    ft.Row([
                        ft.Text(f"{i}.", width=30, weight=ft.FontWeight.BOLD),
                        ft.Text(candidate.name, width=300),
                        ft.Text(f"{candidate.avg_rating:.2f}", width=50),
                        ft.Text(f"({candidate.num_ratings} ratings){warning}", color=ft.Colors.GREY_600),
                    ])
                )

            if len(results.candidates) > 10:
                results_rows.append(
                    ft.Text(
                        f"... and {len(results.candidates) - 10} more (see Excel file)",
                        size=12,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    )
                )

            results_rows.append(ft.Divider())

            # Output file
            date_str = time.strftime("%Y-%m-%d")
            excel_name = f"mathjobs ranking {date_str}.xlsx"
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
                ft.ElevatedButton(
                    "Open Output Folder",
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

    run_button = ft.ElevatedButton(
        "Process Ratings",
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
        on_click=lambda _: process_ratings(),
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
                "2. Select both required files above\n"
                "3. (Optional) Select folder with historical data for better predictions\n"
                "4. Click 'Process Ratings' to generate rankings\n"
                "5. Results are saved as an Excel file",
                size=13,
                color=ft.Colors.BLUE_800,
            ),
            ft.Container(height=5),
            ft.Text(
                "Training folder structure: training/2020/, training/2021/, etc.\n"
                "Each year folder should contain a ratings.csv made by anonymize_training.py",
                size=11,
                color=ft.Colors.BLUE_600,
            ),
            ft.Container(height=5),
            ft.Text(
                "* = two or fewer actual ratings (the score rests mostly on predictions)",
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

            ft.Container(height=15),

            # Optional training folder section
            ft.Text("Training Data (Optional - improves predictions):", weight=ft.FontWeight.BOLD, size=14, color=ft.Colors.GREY_700),
            ft.Row([training_folder_button, training_folder_text], spacing=15),

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
    ft.app(target=main)
