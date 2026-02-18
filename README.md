# HikMicro Thermal Camera Viewer

Real-time thermal imaging viewer for HikMicro USB thermal cameras on Linux.

## Features

- Auto-detects HikMicro camera via USB VID:PID (`2bdf:0102`)
- USB reset recovery if device fails to enumerate
- 256x192 native thermal resolution, upscaled to 768x576
- 6 colormaps: Inferno, Jet, Hot, Magma, Turbo, Plasma
- CLAHE contrast enhancement + temporal smoothing
- Hot/cold spot tracking with crosshair overlay
- Relative intensity scale bar
- Snapshot capture to PNG

## Install

### From .deb package

```bash
sudo dpkg -i hikmicro-thermal-viewer_1.0.0-1_all.deb
sudo apt-get install -f  # install dependencies if needed
```

### Manual

```bash
sudo apt install python3-opencv python3-numpy
python3 thermal_camera.py
```

## Usage

```bash
hikmicro-thermal-viewer
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save snapshot |
| `C` | Cycle colormap |
| `+` | Increase contrast |
| `-` | Decrease contrast |

## USB Permissions

The .deb package installs a udev rule so the camera works without root. For manual installs:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2bdf", ATTR{idProduct}=="0102", MODE="0666"' | sudo tee /etc/udev/rules.d/99-hikmicro-thermal.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Requirements

- Linux with V4L2 support
- Python 3
- python3-opencv
- python3-numpy
- HikMicro USB thermal camera (VID:PID `2bdf:0102`)

## License

MIT
