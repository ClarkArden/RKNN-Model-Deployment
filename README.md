# RKNN Model Deployment - YOLO11

A C++ implementation for deploying YOLO11 object detection model on Rockchip RK3588 platform using RKNN API.

## Features

- YOLO11 object detection on RK3588 NPU
- Support for int8, uint8, and fp32 quantization
- Hardware-accelerated inference using RKNN Runtime
- OpenCV-based image preprocessing and visualization
- Configurable detection parameters (confidence, NMS threshold)
- Built-in logger for debugging

## Prerequisites

### Hardware
- Rockchip RK3588 development board

### Software Dependencies
- RK3588 Linux SDK
- RKNPU2 SDK
- OpenCV (aarch64)
- RGA Library (Rockchip Graphics Acceleration)
- CMake >= 3.4.1
- C++17 compatible compiler

## Project Structure

```
rknn_model/
├── include/
│   ├── rknn_model.hpp      # Base model class
│   ├── yolo11.hpp          # YOLO11 detector implementation
│   ├── type.hpp            # Type definitions
│   ├── utils.hpp           # Utility functions
│   └── logger.hpp          # Logging utilities
├── src/
│   ├── main.cc             # Entry point
│   ├── rknn_model.cc       # Base model implementation
│   ├── yolo11.cc           # YOLO11 implementation
│   ├── utils.cc            # Utility functions
│   └── logger.cc           # Logger implementation
├── model/
│   ├── yolo11.rknn         # RKNN model file
│   ├── car.jpg             # Test image
│   └── coco_80_labels_list.txt  # Class labels
├── build.sh                # Cross-compilation script
└── CMakeLists.txt          # Build configuration
```

## Setup

### 1. Environment Configuration

Update the SDK path in `CMakeLists.txt` and `build.sh`:

```cmake
# CMakeLists.txt (line 4)
set(SDK_ROOT_PATH "/path/to/your/rk3588_linux_release")
```

```bash
# build.sh (line 17)
ENV_SETUP="/path/to/your/buildroot/output/rockchip_rk3588/host/environment-setup"
```

### 2. Prepare Model Files

Place your RKNN model file in the `model/` directory:
- `yolo11.rknn` - YOLO11 model converted to RKNN format
- `coco_80_labels_list.txt` - Class labels file
- Test images (e.g., `car.jpg`)

## Build

### Cross Compilation

```bash
# Clean build
./build.sh clean

# Build
./build.sh
```

The compiled executable will be located at `build/rknn_model`.

### Build Parameters

The build script will:
1. Load RK3588 cross-compilation environment
2. Configure CMake for aarch64 target
3. Build with multiple cores (`-j$(nproc)`)
4. Verify the output binary

## Usage

### Basic Example

```cpp
#include "yolo11.hpp"

int main() {
    // Configure detection parameters
    detector::DetectParam detect_param = {
        0.25,   // confidence threshold
        0.45,   // NMS threshold
        114,    // background fill color
        80      // number of classes
    };

    // Load image
    cv::Mat img = cv::imread("./model/car.jpg");

    // Create YOLO11 detector
    std::unique_ptr<rknn::Model> yolo11 =
        std::make_unique<detector::YOLO11>(
            "./model/yolo11.rknn",
            logger::Level::DEBUG,
            detect_param
        );

    // Run inference
    yolo11->inference(img);

    // Draw results
    yolo11->draw(img);

    return 0;
}
```

### Detection Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `confidence` | float | Confidence threshold for detection | 0.25 |
| `nms_threshold` | float | Non-Maximum Suppression threshold | 0.45 |
| `bf_color` | int | Background fill color | 114 |
| `class_num` | int | Number of object classes | 80 |

### Logger Levels

- `logger::Level::DEBUG` - Detailed debug information
- `logger::Level::INFO` - General information
- `logger::Level::WARN` - Warning messages
- `logger::Level::ERROR` - Error messages

## Deployment

Transfer the following files to your RK3588 board:

```bash
# Required files
build/rknn_model              # Executable
model/yolo11.rknn             # Model file
model/coco_80_labels_list.txt # Labels
model/car.jpg                 # Test image (optional)
```

Run on device:

```bash
cd /path/to/deployment
./rknn_model
```

## Architecture

### Class Hierarchy

```
rknn::Model (Base class)
    └── detector::YOLO11 (YOLO11 implementation)
```

### Key Components

1. **rknn::Model** - Abstract base class providing:
   - Model initialization
   - Inference pipeline
   - Tensor management
   - Logging utilities

2. **detector::YOLO11** - YOLO11 specific implementation:
   - Image preprocessing (resize, letterbox)
   - Multi-scale feature processing
   - Post-processing (NMS, coordinate transformation)
   - Result visualization

3. **Preprocessing Pipeline**:
   - Letterbox resizing to maintain aspect ratio
   - Color space conversion
   - Normalization

4. **Postprocessing Pipeline**:
   - Distribution Focal Loss (DFL) for bounding boxes
   - Confidence filtering
   - Non-Maximum Suppression (NMS)
   - Coordinate transformation to original image space

## Performance Considerations

- The RK3588 NPU provides hardware acceleration for model inference
- int8 quantization offers the best performance
- Image preprocessing is done on CPU using OpenCV
- RGA library can be used for hardware-accelerated image operations

## Troubleshooting

### Build Issues

1. **Environment setup file not found**
   - Verify the `ENV_SETUP` path in `build.sh`
   - Ensure RK3588 SDK is properly installed

2. **Missing dependencies**
   - Check RKNN API path in `CMakeLists.txt`
   - Verify OpenCV and RGA library paths

### Runtime Issues

1. **Model loading failed**
   - Ensure `.rknn` model file is accessible
   - Check model compatibility with RKNN runtime version

2. **Inference errors**
   - Enable DEBUG logging to diagnose issues
   - Verify input image format and dimensions

## License

This project is provided as-is for educational and development purposes.

## Acknowledgments

- Rockchip for RKNN SDK and RK3588 platform
- Ultralytics for YOLO11 architecture
- OpenCV community

## References

- [RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
- [YOLO11](https://github.com/ultralytics/ultralytics)
- [RK3588 Documentation](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html)
