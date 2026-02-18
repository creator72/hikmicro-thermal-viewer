import cv2
import numpy as np
import glob
import time
import fcntl
import os

# USB vendor:product for HikMicro thermal camera
THERMAL_USB_VID = "2bdf"
THERMAL_USB_PID = "0102"


def usb_reset_thermal():
    """Reset the thermal camera's USB port to force re-enumeration."""
    for devpath in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            vid = open(devpath).read().strip()
            if vid != THERMAL_USB_VID:
                continue
            devdir = os.path.dirname(devpath)
            pid = open(os.path.join(devdir, "idProduct")).read().strip()
            if pid != THERMAL_USB_PID:
                continue
            busnum = int(open(os.path.join(devdir, "busnum")).read().strip())
            devnum = int(open(os.path.join(devdir, "devnum")).read().strip())
            usbfs = f"/dev/bus/usb/{busnum:03d}/{devnum:03d}"
            print(f"Resetting USB device at {usbfs}...")
            fd = os.open(usbfs, os.O_WRONLY)
            fcntl.ioctl(fd, 21780, 0)
            os.close(fd)
            time.sleep(2)
            return True
        except Exception as e:
            print(f"USB reset failed: {e}")
    return False


def find_thermal_usb_sysfs():
    """Find the thermal camera's sysfs path by USB VID:PID."""
    for devpath in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            vid = open(devpath).read().strip()
            if vid != THERMAL_USB_VID:
                continue
            devdir = os.path.dirname(devpath)
            pid = open(os.path.join(devdir, "idProduct")).read().strip()
            if pid == THERMAL_USB_PID:
                return devdir
        except Exception:
            continue
    return None


def find_thermal_video_device(sysfs_dir):
    """Find the /dev/videoN node for a USB device."""
    usb_basename = os.path.basename(sysfs_dir)
    for vdev in sorted(glob.glob("/sys/class/video4linux/video*")):
        real = os.path.realpath(vdev)
        if usb_basename in real:
            return "/dev/" + os.path.basename(vdev)
    return None


# Auto-detect thermal camera, reset USB if needed
print("Searching for HikMicro thermal camera...")
sysfs = find_thermal_usb_sysfs()

if not sysfs:
    print("HikMicro thermal camera not found on USB. Is it connected?")
    exit(1)

print(f"Found USB device at {sysfs}")
device = find_thermal_video_device(sysfs)

if not device:
    print("No video device bound, resetting USB to re-enumerate...")
    usb_reset_thermal()
    for attempt in range(5):
        time.sleep(1)
        sysfs = find_thermal_usb_sysfs()
        if sysfs:
            device = find_thermal_video_device(sysfs)
            if device:
                break
        print(f"  Waiting for device... ({attempt + 1}/5)")

if not device:
    print("Cannot find HikMicro thermal camera. Is it connected?")
    exit(1)

print(f"Found thermal camera at {device}")

# Open camera in 256x192 mode (native thermal resolution)
cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

if not cap.isOpened():
    print(f"Cannot open {device}")
    exit(1)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Capture mode: {actual_w}x{actual_h}")

# Output window
OUT_W, OUT_H = 768, 576

# Temporal smoothing
SMOOTH_FRAMES = 3
frame_buffer = []

# CLAHE for local contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Colormaps (press C to cycle)
COLORMAPS = [
    (cv2.COLORMAP_INFERNO, "Inferno"),
    (cv2.COLORMAP_JET, "Jet"),
    (cv2.COLORMAP_HOT, "Hot"),
    (cv2.COLORMAP_MAGMA, "Magma"),
    (cv2.COLORMAP_TURBO, "Turbo"),
    (cv2.COLORMAP_PLASMA, "Plasma"),
]
cmap_idx = 0

cv2.namedWindow("HikMicro Thermal", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HikMicro Thermal", OUT_W + 90, OUT_H)

print("Controls: Q=quit, S=snapshot, C=colormap, +/-=contrast")

contrast_boost = 1.0


def draw_scale_bar(image, colormap):
    """Draw a vertical relative intensity scale bar."""
    h = image.shape[0]
    bar_w = 25
    margin = 8

    gradient = np.linspace(255, 0, h).astype(np.uint8)
    bar = np.tile(gradient.reshape(-1, 1), (1, bar_w))
    bar_colored = cv2.applyColorMap(bar, colormap)
    cv2.rectangle(bar_colored, (0, 0), (bar_w - 1, h - 1), (180, 180, 180), 1)

    label_w = 45
    label_area = np.zeros((h, label_w, 3), dtype=np.uint8)

    labels = ["HOT", "", "", "", "", "", "COLD"]
    for i, label in enumerate(labels):
        if not label:
            continue
        y = int(h * i / (len(labels) - 1))
        y = max(12, min(y, h - 5))
        cv2.line(bar_colored, (bar_w - 4, y), (bar_w - 1, y), (255, 255, 255), 1)
        cv2.putText(label_area, label, (4, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    spacer = np.zeros((h, margin, 3), dtype=np.uint8)
    return np.hstack((image, spacer, bar_colored, label_area))


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    raw_bytes = frame.flatten()
    h, w = 192, 256

    if len(raw_bytes) < h * w * 2:
        continue

    # Extract Y channel from YUYV
    frame_data = raw_bytes[:h * w * 2].reshape(h, w * 2)
    gray = frame_data[:, 0::2].copy().astype(np.float32)

    # Temporal averaging
    frame_buffer.append(gray)
    if len(frame_buffer) > SMOOTH_FRAMES:
        frame_buffer.pop(0)
    thermal = np.mean(frame_buffer, axis=0).astype(np.float32)

    # Find hot/cold spots on raw data
    min_val, max_val, minLoc, maxLoc = cv2.minMaxLoc(thermal)

    # Normalize with contrast boost
    t_min = thermal.min()
    t_max = thermal.max()
    t_range = max(t_max - t_min, 1)
    t_mid = (t_min + t_max) / 2
    b_min = t_mid - (t_range / 2) / contrast_boost
    b_max = t_mid + (t_range / 2) / contrast_boost

    thermal_norm = np.clip((thermal - b_min) / (b_max - b_min), 0, 1)
    thermal_8bit = (thermal_norm * 255).astype(np.uint8)

    # CLAHE for local contrast
    thermal_enhanced = clahe.apply(thermal_8bit)

    # Upscale with Lanczos
    thermal_big = cv2.resize(thermal_enhanced, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)

    # Bilateral filter - smooth blocks, preserve edges
    thermal_smooth = cv2.bilateralFilter(thermal_big, 7, 50, 50)

    # Apply colormap
    colormap, cmap_name = COLORMAPS[cmap_idx]
    thermal_color = cv2.applyColorMap(thermal_smooth, colormap)

    # Light sharpen
    blurred = cv2.GaussianBlur(thermal_color, (0, 0), 2)
    thermal_color = cv2.addWeighted(thermal_color, 1.3, blurred, -0.3, 0)

    # Scale hotspot to display coordinates
    sx = OUT_W / w
    sy = OUT_H / h
    hot_pt = (int(maxLoc[0] * sx), int(maxLoc[1] * sy))
    cold_pt = (int(minLoc[0] * sx), int(minLoc[1] * sy))

    # Hotspot crosshair (no numbers)
    cx, cy = hot_pt
    cv2.line(thermal_color, (cx - 12, cy), (cx - 4, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(thermal_color, (cx + 4, cy), (cx + 12, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(thermal_color, (cx, cy - 12), (cx, cy - 4), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(thermal_color, (cx, cy + 4), (cx, cy + 12), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(thermal_color, hot_pt, 14, (255, 255, 255), 1, cv2.LINE_AA)

    # Cold spot marker
    cv2.drawMarker(thermal_color, cold_pt, (255, 200, 0), cv2.MARKER_TRIANGLE_DOWN, 8, 1, cv2.LINE_AA)

    # Centre crosshair
    ctr = (OUT_W // 2, OUT_H // 2)
    cv2.drawMarker(thermal_color, ctr, (200, 200, 200), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

    # Minimal HUD - just colormap name
    cv2.putText(thermal_color, cmap_name, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    # Scale bar
    display = draw_scale_bar(thermal_color, colormap)

    cv2.imshow("HikMicro Thermal", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"thermal_{int(time.time())}.png"
        cv2.imwrite(fname, display)
        print(f"Saved {fname}")
    elif key == ord('c'):
        cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
        print(f"Colormap: {COLORMAPS[cmap_idx][1]}")
    elif key == ord('+') or key == ord('='):
        contrast_boost = min(contrast_boost + 0.2, 5.0)
        print(f"Contrast: {contrast_boost:.1f}x")
    elif key == ord('-'):
        contrast_boost = max(contrast_boost - 0.2, 0.4)
        print(f"Contrast: {contrast_boost:.1f}x")

cap.release()
cv2.destroyAllWindows()
