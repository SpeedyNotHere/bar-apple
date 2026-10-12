
import cv2
import numpy as np
import barcode
from barcode.writer import ImageWriter
from PIL import Image
import os

INPUT_VIDEO = "bad_apple_for_comparison.mp4"
OUTPUT_VIDEO = "bad_apple_barcodes.mp4"

# output settings
OUTPUT_WIDTH = 960
OUTPUT_HEIGHT = 720

# image sampling settings
# every other vertical pixel is scanned before resizing
SAMPLE_WIDTH = 12
SAMPLE_HEIGHT = 12

# black-and-white contrast
THRESHOLD = 128

# barcode rendering settings
BARCODE_MODULE_WIDTH = 0.5
BARCODE_MODULE_HEIGHT = 30
BARCODE_DPI = 150

# 0.0 = no smoothing; higher values create more smoothing
TEMPORAL_SMOOTHING = 0.10


def frame_to_code128(frame, previous_samples):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scan every other vertical pixel
    gray = gray[::2, :]

    # resize to a fixed-size image representation
    small = cv2.resize(
        gray,
        (SAMPLE_WIDTH, SAMPLE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # smooth changes between consecutive frames
    if previous_samples is not None and TEMPORAL_SMOOTHING > 0:
        small = cv2.addWeighted(
            small,
            1.0 - TEMPORAL_SMOOTHING,
            previous_samples,
            TEMPORAL_SMOOTHING,
            0
        )

    previous_samples = small.copy()

    # force the sampled pixels to pure black or white
    _, binary = cv2.threshold(
        small,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # encode the binary pixels as 0s and 1s
    bits = (binary == 0).astype(np.uint8).flatten()

    # pack six pixels into each printable character
    # each character represents six sampled black/white pixels
    payload = []

    for start in range(0, len(bits), 6):
        chunk = bits[start:start + 6]
        value = 0

        for bit in chunk:
            value = (value << 1) | int(bit)

        payload.append(chr(32 + value))

    return "".join(payload), previous_samples


def render_barcode(payload):
    code128 = barcode.get(
        "code128",
        payload,
        writer=ImageWriter()
    )

    options = {
        "module_width": BARCODE_MODULE_WIDTH,
        "module_height": BARCODE_MODULE_HEIGHT,
        "quiet_zone": 2,
        "font_size": 1,
        "text_distance": 1,
        "write_text": False,
        "dpi": BARCODE_DPI,
        "background": "white",
        "foreground": "black"
    }

    return code128.render(options).convert("RGB")


def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: could not find {INPUT_VIDEO}")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("Error: could not open the input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )

    if not writer.isOpened():
        print("Error: could not create the output video.")
        cap.release()
        return

    previous_samples = None
    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            payload, previous_samples = frame_to_code128(
                frame,
                previous_samples
            )

            barcode_image = render_barcode(payload)

            # keep the barcode inside the output frame
            barcode_image.thumbnail(
                (OUTPUT_WIDTH - 20, OUTPUT_HEIGHT - 20),
                Image.Resampling.LANCZOS
            )

            # white background
            canvas = Image.new(
                "RGB",
                (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                "white"
            )

            # center the single continuous barcode
            x = (OUTPUT_WIDTH - barcode_image.width) // 2
            y = (OUTPUT_HEIGHT - barcode_image.height) // 2
            canvas.paste(barcode_image, (x, y))

            writer.write(
                cv2.cvtColor(
                    np.array(canvas),
                    cv2.COLOR_RGB2BGR
                )
            )

            frame_number += 1

            if frame_number % 30 == 0:
                if total_frames > 0:
                    percent = frame_number / total_frames * 100
                    print(
                        f"\rProgress: {percent:.1f}% "
                        f"({frame_number}/{total_frames} frames)",
                        end="",
                        flush=True
                    )
                else:
                    print(
                        f"\rProcessed {frame_number} frames",
                        end="",
                        flush=True
                    )

    finally:
        cap.release()
        writer.release()

    print(f"\nDone! Saved as {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
