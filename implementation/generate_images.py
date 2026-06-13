import os
import cv2
import numpy as np
import random

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

OUTPUT_DIRS = {
    "easy": "images/easy",
    "medium": "images/medium",
    "hard": "images/hard",
    "missing": "images/missing",
}

RED = (0, 0, 255)  # OpenCV uses BGR, not RGB


def ensure_dirs():
    for folder in OUTPUT_DIRS.values():
        os.makedirs(folder, exist_ok=True)


def create_blank_image():
    return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)


def draw_red_circle(image, x, y, radius=20):
    cv2.circle(image, (x, y), radius, RED, -1)


def generate_easy_image(index):
    """
    Easy case:
    Object is inside the predicted small search window.
    """
    image = create_blank_image()

    # Predicted center is around (320, 240)
    x = random.randint(290, 350)
    y = random.randint(210, 270)

    draw_red_circle(image, x, y)
    filename = f"images/easy/easy_{index:03d}.png"
    cv2.imwrite(filename, image)


def generate_medium_image(index):
    """
    Medium case:
    Object is outside the small window, but inside a medium search area.
    """
    image = create_blank_image()

    x = random.randint(220, 420)
    y = random.randint(140, 340)

    # Avoid placing it too close to the center
    while 280 <= x <= 360 and 200 <= y <= 280:
        x = random.randint(220, 420)
        y = random.randint(140, 340)

    draw_red_circle(image, x, y)
    filename = f"images/medium/medium_{index:03d}.png"
    cv2.imwrite(filename, image)


def generate_hard_image(index):
    """
    Hard case:
    Object is far away from the predicted area.
    Full image search is likely required.
    """
    image = create_blank_image()

    corners = [
        (random.randint(20, 120), random.randint(20, 120)),
        (random.randint(520, 620), random.randint(20, 120)),
        (random.randint(20, 120), random.randint(360, 460)),
        (random.randint(520, 620), random.randint(360, 460)),
    ]

    x, y = random.choice(corners)
    draw_red_circle(image, x, y)

    filename = f"images/hard/hard_{index:03d}.png"
    cv2.imwrite(filename, image)


def generate_missing_image(index):
    """
    Missing case:
    No object exists. This forces full search and represents the worst case.
    """
    image = create_blank_image()
    filename = f"images/missing/missing_{index:03d}.png"
    cv2.imwrite(filename, image)


def main():
    ensure_dirs()

    number_per_case = 30

    for i in range(number_per_case):
        generate_easy_image(i)
        generate_medium_image(i)
        generate_hard_image(i)
        generate_missing_image(i)

    print(f"Generated {number_per_case} images for each difficulty level.")


if __name__ == "__main__":
    main()