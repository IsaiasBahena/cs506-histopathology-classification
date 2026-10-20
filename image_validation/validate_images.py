
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "full" / "image_metadata.csv"

REPORTS_DIR = PROJECT_ROOT / "image_validation" / "reports"
SUMMARY_PATH = REPORTS_DIR / "image_quality_summary.json"
ISSUES_PATH = REPORTS_DIR / "image_quality_issues.csv"
SUMMARY_PATH = REPORTS_DIR / "image_quality_summary.json"
ISSUES_PATH = REPORTS_DIR / "image_quality_issues.csv"

VALID_LABELS = {
    "ADI", "BACK", "DEB", "LYM", "MUC",
    "MUS", "NORM", "STR", "TUM"
}

VALID_DATASETS = {
    "NCT-CRC-HE-100K",
    "CRC-VAL-HE-7K"
}

REQUIRED_COLUMNS = (
    "id",
    "path",
    "label",
    "dataset",
    "width",
    "height",
    "color_mode",
)

EXPECTED_SIZE = (224, 224)
EXPECTED_MODE = "RGB"

ISSUE_TYPES = (
    "metadata",
    "missing_metadata",
    "duplicate_metadata",
    "unreadable_image",
    "file_access_error",
    "invalid_dimensions",
    "invalid_color_mode",
    "exact_duplicate",
    "pixel_duplicate",
)


# ============================================================
# Issue tracking
# ============================================================

def create_issue_tracker():
    """Create an empty collection of validation issues."""

    return {
        issue_type: []
        for issue_type in ISSUE_TYPES
    }


def add_issue(issues, issue_type, path, **details):
    """Record a validation issue using a consistent structure."""

    issues[issue_type].append({
        "path": str(path),
        **details
    })


# ============================================================
# Path utilities
# ============================================================

def normalize_path(path):
    """Convert metadata paths into platform-independent Path objects."""

    return Path(str(path).replace("\\", "/"))


def resolve_image_path(relative_path):
    """Resolve an image path relative to the data directory."""

    return DATA_DIR / normalize_path(relative_path)


def has_missing_values(row, columns):
    """Check whether a metadata row has missing required values."""

    return any(
        pd.isna(getattr(row, column))
        for column in columns
    )


# ============================================================
# Metadata validation
# ============================================================

def validate_metadata_completeness(df, issues):
    """Validate required columns and detect missing metadata values."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required metadata columns: {missing_columns}"
        )

    for column in REQUIRED_COLUMNS:
        missing_rows = df[df[column].isna()]

        for index in missing_rows.index:
            add_issue(
                issues,
                "missing_metadata",
                df.at[index, "path"],
                row_index=int(index),
                column=column
            )


def validate_metadata_duplicates(df, issues):
    """Detect duplicate image paths and dataset-specific IDs."""

    duplicate_checks = {
        "Duplicate image path": ["path"],
        "Duplicate dataset/image ID": ["dataset", "id"]
    }

    for issue_name, columns in duplicate_checks.items():
        duplicates = df.duplicated(
            subset=columns,
            keep="first"
        )

        for index in df.index[duplicates]:
            add_issue(
                issues,
                "duplicate_metadata",
                df.at[index, "path"],
                row_index=int(index),
                issue=issue_name
            )


def validate_metadata_row(row, issues):
    """Validate a single metadata record."""

    if has_missing_values(
        row,
        ("id", "path", "label", "dataset")
    ):
        return

    path = normalize_path(row.path)

    # Validate tissue label.
    if row.label not in VALID_LABELS:
        add_issue(
            issues,
            "metadata",
            row.path,
            issue="Invalid tissue label",
            value=row.label
        )

    # Validate dataset name.
    if row.dataset not in VALID_DATASETS:
        add_issue(
            issues,
            "metadata",
            row.path,
            issue="Invalid dataset",
            value=row.dataset
        )

    # Validate expected directory structure.
    expected_path = (
        Path("full") / row.dataset / row.label / row.id
    )

    if path != expected_path:
        add_issue(
            issues,
            "metadata",
            row.path,
            issue="Path mismatch",
            expected=str(expected_path)
        )

    # Validate filename against metadata ID.
    if path.name != row.id:
        add_issue(
            issues,
            "metadata",
            row.path,
            issue="Filename mismatch",
            expected=row.id
        )

    # Validate filename prefix against tissue label.
    if not row.id.startswith(f"{row.label}-"):
        add_issue(
            issues,
            "metadata",
            row.path,
            issue="Filename label mismatch",
            value=row.id
        )


def validate_metadata(df, issues):
    """Run all metadata validation checks."""

    validate_metadata_completeness(df, issues)
    validate_metadata_duplicates(df, issues)

    for row in df.itertuples(index=False):
        validate_metadata_row(row, issues)


# ============================================================
# Image property validation
# ============================================================

def validate_dimensions(image, row, issues):
    """Validate actual image dimensions and metadata agreement."""

    actual_size = image.size

    if has_missing_values(row, ("width", "height")):
        return

    metadata_size = (row.width, row.height)

    if (
        actual_size != EXPECTED_SIZE
        or actual_size != metadata_size
    ):
        add_issue(
            issues,
            "invalid_dimensions",
            row.path,
            actual_size=actual_size,
            metadata_size=metadata_size
        )


def validate_color_mode(image, row, issues):
    """Validate image color mode and metadata agreement."""

    if pd.isna(row.color_mode):
        return

    if (
        image.mode != EXPECTED_MODE
        or image.mode != row.color_mode
    ):
        add_issue(
            issues,
            "invalid_color_mode",
            row.path,
            actual_mode=image.mode,
            metadata_mode=row.color_mode
        )


# ============================================================
# Hashing and duplicate detection
# ============================================================

def calculate_file_hash(image_path):
    """Calculate SHA-256 hash of the original image file."""

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_pixel_hash(image):
    """Calculate SHA-256 hash of decoded image content."""

    sha256 = hashlib.sha256()

    sha256.update(image.mode.encode("utf-8"))
    sha256.update(str(image.size).encode("utf-8"))
    sha256.update(image.tobytes())

    return sha256.hexdigest()


def check_duplicate(
    image_hash,
    image_path,
    seen_hashes,
    issues,
    issue_type
):
    """Detect duplicate hashes and record the first occurrence."""

    if image_hash in seen_hashes:
        add_issue(
            issues,
            issue_type,
            image_path,
            original=seen_hashes[image_hash],
            sha256=image_hash
        )
    else:
        seen_hashes[image_hash] = str(image_path)


# ============================================================
# Image validation
# ============================================================

def validate_single_image(row, issues, file_hashes, pixel_hashes):
    """Validate one image and detect duplicate content."""

    # Skip records without an image path.
    if pd.isna(row.path):
        return

    image_path = resolve_image_path(row.path)

    # Step 1: Open and decode the image.
    try:
        with Image.open(image_path) as image:
            image.load()

            # Step 2: Validate image properties.
            validate_dimensions(image, row, issues)
            validate_color_mode(image, row, issues)

            # Step 3: Calculate decoded pixel hash.
            pixel_hash = calculate_pixel_hash(image)

    except (OSError, ValueError) as error:
        add_issue(
            issues,
            "unreadable_image",
            row.path,
            error=str(error)
        )
        return

    # Step 4: Detect pixel-identical duplicates.
    check_duplicate(
        pixel_hash,
        row.path,
        pixel_hashes,
        issues,
        "pixel_duplicate"
    )

    # Step 5: Calculate file-level hash.
    try:
        file_hash = calculate_file_hash(image_path)

    except OSError as error:
        add_issue(
            issues,
            "file_access_error",
            row.path,
            error=str(error)
        )
        return

    # Step 6: Detect exact file duplicates.
    check_duplicate(
        file_hash,
        row.path,
        file_hashes,
        issues,
        "exact_duplicate"
    )


def validate_images(df, issues):
    """Validate all images referenced in the metadata."""

    file_hashes = {}
    pixel_hashes = {}

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Validating images"
    ):
        validate_single_image(
            row,
            issues,
            file_hashes,
            pixel_hashes
        )


# ============================================================
# Report generation
# ============================================================

def build_summary(df, issues):
    """Create the validation summary dictionary."""

    return {
        "validation_timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "total_images": len(df),
        "images_per_dataset": (
            df["dataset"].value_counts().to_dict()
        ),
        "validation_results": {
            "metadata_issues": len(issues["metadata"]),
            "missing_metadata": len(issues["missing_metadata"]),
            "duplicate_metadata": len(issues["duplicate_metadata"]),
            "unreadable_images": len(issues["unreadable_image"]),
            "file_access_errors": len(issues["file_access_error"]),
            "invalid_dimensions": len(issues["invalid_dimensions"]),
            "invalid_color_modes": len(issues["invalid_color_mode"]),
            "exact_duplicate_images": len(issues["exact_duplicate"]),
            "pixel_identical_duplicate_images": len(issues["pixel_duplicate"])
        }
    }


def build_issues_dataframe(issues):
    """Convert validation issues into a CSV-compatible DataFrame."""

    records = []

    for issue_type, issue_list in issues.items():
        for issue in issue_list:
            records.append({
                "issue_type": issue_type,
                "path": issue["path"],
                "details": json.dumps(issue, default=str)
            })

    return pd.DataFrame(
        records,
        columns=["issue_type", "path", "details"]
    )


def generate_reports(df, issues):
    """Save the JSON summary and CSV issue report."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary(df, issues)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    issues_df = build_issues_dataframe(issues)
    issues_df.to_csv(ISSUES_PATH, index=False)

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(f"Issues saved to: {ISSUES_PATH}")


# ============================================================
# Console output
# ============================================================

def print_summary(df, issues):
    """Display validation statistics and example issues."""

    print()
    print(f"Total images: {len(df):,}")

    labels = {
        "metadata": "Metadata issues",
        "missing_metadata": "Missing metadata values",
        "duplicate_metadata": "Duplicate metadata records",
        "unreadable_image": "Unreadable images",
        "file_access_error": "File access errors",
        "invalid_dimensions": "Invalid dimensions",
        "invalid_color_mode": "Invalid color modes",
        "exact_duplicate": "Exact duplicate images",
        "pixel_duplicate": "Pixel-identical duplicate images"
    }

    for issue_type, label in labels.items():
        records = issues[issue_type]

        print(f"{label}: {len(records):,}")

        if records:
            print(f"\nExamples of {label.lower()}:")

            for record in records[:10]:
                print(record)


# ============================================================
# Main
# ============================================================

def main():
    """Run the complete image-quality validation pipeline."""

    # Load metadata.
    df = pd.read_csv(METADATA_PATH)

    print(f"Total images to validate: {len(df):,}")

    # Step 1: Validate metadata.
    issues = create_issue_tracker()
    validate_metadata(df, issues)

    # Step 2: Validate image files and detect duplicates.
    validate_images(df, issues)

    # Step 3: Display results.
    print_summary(df, issues)

    # Step 4: Generate reports.
    generate_reports(df, issues)


if __name__ == "__main__":
    main()
