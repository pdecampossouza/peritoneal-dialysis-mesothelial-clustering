"""
build_data_image.py

This script scans the folder "HPMCs_DialisePeritoneal" and creates an Excel file
called "data_image.xlsx" with one row per image.

It is designed to be easy to read and understand for non-programmers.
The goal is to have a clean "manifest" of all images, linked to:
- patient ID
- well number
- zone inside the well
- image mode (RGB or BW)
- file name and relative path
"""

from pathlib import Path
import re
import pandas as pd


def main():
    print("=" * 70)
    print("BUILD DATA IMAGE – START")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Define main folders
    # ------------------------------------------------------------------
    # Root folder = the folder where this script file is located
    root = Path(__file__).resolve().parent
    print(f"[INFO] Project root folder: {root}")

    # Folder that contains the patients and their images
    data_dir = root / "HPMCs_DialisePeritoneal"
    print(f"[INFO] Images folder: {data_dir}")

    if not data_dir.exists():
        print("[ERROR] The folder 'HPMCs_DialisePeritoneal' was not found.")
        print("        Please make sure this script is in the correct project folder.")
        return

    # This list will store one dictionary per image
    records = []

    # Valid image extensions (can be extended if needed)
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    # ------------------------------------------------------------------
    # 2. Loop over patient folders
    # ------------------------------------------------------------------
    print("\n[STEP] Scanning patient folders...\n")
    patient_folders = sorted(p for p in data_dir.iterdir() if p.is_dir())

    if not patient_folders:
        print("[WARNING] No patient folders found inside 'HPMCs_DialisePeritoneal'.")
        return

    for patient_folder in patient_folders:
        folder_name = patient_folder.name

        # We expect patient folders to be numeric (e.g., 1056, 1059, ...)
        try:
            patient_id = int(folder_name)
        except ValueError:
            print(f"[WARNING] Skipping folder (not a patient ID): {folder_name}")
            continue

        print(f"  -> Patient {patient_id} (folder: {folder_name})")

        # ------------------------------------------------------------------
        # 3. Loop over well folders inside each patient
        #    Example folder names: 'Poço 1', 'Poço 2', ...
        # ------------------------------------------------------------------
        well_folders = sorted(w for w in patient_folder.iterdir() if w.is_dir())
        if not well_folders:
            print(f"     [WARNING] No well folders found for patient {patient_id}.")
            continue

        for well_folder in well_folders:
            well_folder_name = well_folder.name

            # Extract the well number (any digits in the folder name)
            # e.g. "Poço 1" -> 1
            match_well = re.search(r"(\d+)", well_folder_name)
            if not match_well:
                print(
                    f"     [WARNING] Could not detect well number in folder: "
                    f"{well_folder_name}"
                )
                continue

            well = int(match_well.group(1))
            print(f"     -> Well {well} (folder: {well_folder_name})")

            # ------------------------------------------------------------------
            # 4. Loop over image files inside each well
            # ------------------------------------------------------------------
            img_files = sorted(f for f in well_folder.iterdir() if f.is_file())

            if not img_files:
                print(f"        [WARNING] No image files in {well_folder_name}.")
                continue

            for img_path in img_files:
                if img_path.suffix.lower() not in valid_exts:
                    # Skip non-image files
                    continue

                # Example file name: '1.A.1.jpg'
                filename = img_path.name
                stem = img_path.stem  # '1.A.1' (without extension)

                parts = stem.split(".")
                if len(parts) != 3:
                    print(
                        f"        [WARNING] Unexpected file name format: {filename} "
                        "(expected something like '1.A.1.jpg')"
                    )
                    continue

                # parts[0] should be the well number inside the file name (1..5)
                try:
                    well_from_name = int(parts[0])
                except ValueError:
                    print(
                        f"        [WARNING] Could not read well number in file name: "
                        f"{filename}"
                    )
                    continue

                zone = parts[1].upper().strip()  # A, B, C, D, E

                # parts[2] should be the mode code: 1 (RGB) or 2 (BW)
                try:
                    mode_code = int(parts[2])
                except ValueError:
                    print(
                        f"        [WARNING] Could not read mode code in file name: "
                        f"{filename}"
                    )
                    continue

                if mode_code == 1:
                    mode_label = "RGB"
                elif mode_code == 2:
                    mode_label = "BW"
                else:
                    mode_label = "UNKNOWN"

                # Optional consistency check: does the well in the file name
                # match the well in the folder name?
                if well_from_name != well:
                    print(
                        f"        [NOTE] Well number mismatch: file name says "
                        f"{well_from_name}, folder says {well}. "
                        f"File: {filename}"
                    )

                # Relative path from the project root
                relative_path = img_path.relative_to(root)

                # Store this image as one record
                records.append(
                    {
                        "patient_id": patient_id,
                        "well": well,
                        "zone": zone,
                        "mode_code": mode_code,
                        "mode_label": mode_label,
                        "filename": filename,
                        "relative_path": str(relative_path),
                    }
                )

    # ----------------------------------------------------------------------
    # 5. Create DataFrame and save to Excel
    # ----------------------------------------------------------------------
    print("\n[STEP] Creating Excel file 'data_image.xlsx'...\n")

    if not records:
        print("[ERROR] No image records were found. Excel file will not be created.")
        return

    df = pd.DataFrame(records)

    # Sort the table for nicer reading
    df = df.sort_values(by=["patient_id", "well", "zone", "mode_code"]).reset_index(
        drop=True
    )

    output_path = root / "data_image.xlsx"
    df.to_excel(output_path, index=False)

    # ----------------------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------------------
    n_patients = df["patient_id"].nunique()
    n_images = len(df)

    print("=" * 70)
    print("BUILD DATA IMAGE – DONE")
    print("=" * 70)
    print(f"[SUMMARY] Number of patients found: {n_patients}")
    print(f"[SUMMARY] Number of images listed:  {n_images}")
    print(f"[SUMMARY] Excel file saved as:      {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
