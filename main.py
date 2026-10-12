from PIL import Image
import cv2
import barcode
from barcode.writer import ImageWriter
import io
import numpy as np

print("hello")

video = cv2.VideoCapture("bad_apple_for_comparison.mp4")

if not video.isOpened():
    print("failed to open bad_apple_for_comparison.mp4")
    raise SystemExit

# settings
fps = 60
start_frame = 81
max_frames = fps * 15
output_size = (960, 300)
bar_count = 180
section_count = 12
module_width = 0.25

# start at frame 81
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

success, frame = video.read()
if not success:
    print("failed to read the starting frame")
    video.release()
    raise SystemExit

print(f"source frame shape: {frame.shape}")

# start processing from frame 81
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

writer = cv2.VideoWriter(
    "bad_apple_barcodes.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    output_size
)

if not writer.isOpened():
    print("failed to create output video")
    video.release()
    raise SystemExit

frame_count = 0

while frame_count < max_frames:
    success, frame = video.read()
    if not success:
        break

    # convert source frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # improve local contrast
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    # otsu thresholding
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # adaptive thresholding helps capture smaller local details
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5
    )

    # combine global and local detail
    combined = cv2.bitwise_and(binary, adaptive)

    # reduce the frame into a grid while retaining vertical detail
    small_binary = cv2.resize(
        combined,
        (bar_count, section_count),
        interpolation=cv2.INTER_AREA
    )

    _, small_binary = cv2.threshold(
        small_binary,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # encode each grid row as a string of 0s and 1s
    # use every row so vertical details affect the encoded data
    pixel_data = ""

    for y in range(section_count):
        for x in range(bar_count):
            pixel_data += "1" if small_binary[y, x] < 128 else "0"

    # Code 128 supports text data, not arbitrary image pixels.
    # Encoding the binary grid as text gives the barcode a changing pattern.
    code128 = barcode.get(
        "code128",
        pixel_data,
        writer=ImageWriter()
    )

    buffer = io.BytesIO()

    code128.write(buffer, options={
        "write_text": False,
        "module_width": module_width,
        "module_height": 30,
        "quiet_zone": 2,
        "font_size": 0,
        "text_distance": 0
    })

    buffer.seek(0)
    barcode_image = Image.open(buffer).convert("RGB")

    # invert barcode colors
    barcode_array = 255 - np.array(barcode_image)

    # create barcode-only output canvas
    canvas = np.full(
        (output_size[1], output_size[0], 3),
        255,
        dtype=np.uint8
    )

    # fit barcode into output without cropping
    h, w = barcode_array.shape[:2]

    scale = min(
        output_size[1] / h,
        output_size[0] / w
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    barcode_array = cv2.resize(
        barcode_array,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    x_offset = (output_size[0] - new_w) // 2
    y_offset = (output_size[1] - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = cv2.cvtColor(
        barcode_array,
        cv2.COLOR_RGB2BGR
    )

    # write only the barcode frame; never overlay the source video
    writer.write(canvas)

    buffer.close()
    frame_count += 1

    print(f"processed {frame_count}/{max_frames}")

writer.release()
video.release()
cv2.destroyAllWindows()

print("saved bad_apple_barcodes.mp4")