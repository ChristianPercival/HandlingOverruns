import os
import time
import cv2
import pandas as pd

IMAGE_DIRS = {
    "easy": "images/easy",
    "medium": "images/medium",
    "hard": "images/hard",
    "missing": "images/missing",
}

RESULTS_FILE = "results/runtime_log.csv"

# Predicted center area
CENTER_X = 320
CENTER_Y = 240

# Search regions
SMALL_WINDOW = 100
MEDIUM_WINDOW = 260

# Timing assumptions in milliseconds.
# We will update these after measuring real results.
NORMAL_BUDGET_MS = 2.0
DEADLINE_MS = 10.0


def crop_window(image, center_x, center_y, size):
    half = size // 2

    x1 = max(center_x - half, 0)
    y1 = max(center_y - half, 0)
    x2 = min(center_x + half, image.shape[1])
    y2 = min(center_y + half, image.shape[0])

    return image[y1:y2, x1:x2]


def find_red_object(image_region):
    """
    Detects a red object using simple color thresholding.
    Returns True if enough red pixels are found.
    """

    hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)

    # Red wraps around the hue range, so use two masks.
    lower_red_1 = (0, 100, 100)
    upper_red_1 = (10, 255, 255)

    lower_red_2 = (160, 100, 100)
    upper_red_2 = (179, 255, 255)

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = mask1 + mask2

    red_pixels = cv2.countNonZero(mask)

    return red_pixels > 50


def process_image(image_path):
    """
    Search order:
    1. Small predicted window
    2. Medium window
    3. Full image
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    start = time.perf_counter()

    small_region = crop_window(image, CENTER_X, CENTER_Y, SMALL_WINDOW)
    found = find_red_object(small_region)

    if found:
        search_mode = "small"
    else:
        medium_region = crop_window(image, CENTER_X, CENTER_Y, MEDIUM_WINDOW)
        found = find_red_object(medium_region)

        if found:
            search_mode = "medium"
        else:
            found = find_red_object(image)
            search_mode = "full"

    finish = time.perf_counter()

    execution_time_ms = (finish - start) * 1000.0

    return found, search_mode, execution_time_ms


def collect_image_paths():
    image_entries = []

    for difficulty, folder in IMAGE_DIRS.items():
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(".png"):
                image_entries.append({
                    "difficulty": difficulty,
                    "path": os.path.join(folder, filename),
                    "filename": filename,
                })

    return image_entries


def main():
    os.makedirs("results", exist_ok=True)

    rows = []
    image_entries = collect_image_paths()

    for frame_id, entry in enumerate(image_entries):
        found, search_mode, execution_time_ms = process_image(entry["path"])

        overrun = execution_time_ms > NORMAL_BUDGET_MS
        deadline_missed = execution_time_ms > DEADLINE_MS

        rows.append({
            "frame_id": frame_id,
            "filename": entry["filename"],
            "difficulty": entry["difficulty"],
            "found": found,
            "search_mode": search_mode,
            "execution_time_ms": execution_time_ms,
            "normal_budget_ms": NORMAL_BUDGET_MS,
            "deadline_ms": DEADLINE_MS,
            "overrun": overrun,
            "deadline_missed": deadline_missed,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_FILE, index=False)

    print(f"Processed {len(df)} images.")
    print(f"Saved results to {RESULTS_FILE}")

    print()
    print("Average execution time by difficulty:")
    print(df.groupby("difficulty")["execution_time_ms"].mean())

    print()
    print("Overruns by difficulty:")
    print(df.groupby("difficulty")["overrun"].sum())


if __name__ == "__main__":
    main()