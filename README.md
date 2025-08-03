# T4 to YOLO Format Converter

A unified conversion tool from T4 dataset format to YOLO format.  
Automatically detects and converts single or multiple T4 datasets.

## Features

- **Automatic Dataset Detection**: Automatically detects single T4 dataset or multiple datasets
- **Unified Class Mapping**: Consistent class ID assignment across multiple datasets
- **Multi-Camera Support**: Camera names included in filenames (avoiding duplicates)
- **Specific Camera Filtering**: Convert only specific cameras
- **Automatic Bounding Box Conversion**: Normalized conversion from T4 format to YOLO format
- **Proper Line Breaks**: Outputs correct YOLO annotation format

## Usage

### Basic Commands

```bash
# Recommended (after pip install -e . or python -m):
python -m t4_to_yolo_converter.main input_path output_path [--camera CAMERA_NAME] [--list]
# or if installed as a package:
t4-to-yolo input_path output_path [--camera CAMERA_NAME] [--list]
```

## Examples

### Single Dataset
```bash
# All cameras
python -m t4_to_yolo_converter.main "dataset_047" "yolo_output"
# or
t4-to-yolo "dataset_047" "yolo_output"

# Specific camera
python -m t4_to_yolo_converter.main "dataset_047" "yolo_output" --camera CAM_FRONT_NARROW
# or
t4-to-yolo "dataset_047" "yolo_output" --camera CAM_FRONT_NARROW
```

### Multiple Datasets
```bash
# List datasets
python -m t4_to_yolo_converter.main "datasets/" "yolo_batch" --list
# or
t4-to-yolo "datasets/" "yolo_batch" --list

# Convert all
python -m t4_to_yolo_converter.main "datasets/" "yolo_batch"
# or
t4-to-yolo "datasets/" "yolo_batch"
```

## Auto-Detection

The script automatically detects:
- **Single Dataset**: Input directory is a T4 dataset
- **Multiple Datasets**: Input directory contains multiple T4 datasets

## T4 Dataset Requirements

T4 datasets must contain:
```
dataset_dir/
└── annotation/
    ├── category.json
    ├── sample_data.json
    └── object_ann.json
```

## Output Format

### Filenames
- Single dataset: `CAM_FRONT_NARROW_00001.jpg`
- Multiple datasets: `dataset_name_CAM_FRONT_NARROW_00001.jpg`

### YOLO Annotations
Format: `class_id center_x center_y width height` (normalized)

Example:
```
12 0.806424 0.816398 0.364931 0.367204
9 0.479514 0.648118 0.044444 0.159677
```

## Example Output

### Single Dataset
```bash
$ python -m t4_to_yolo_converter.main "dataset_047" "yolo_output" --camera CAM_FRONT_NARROW

Found 1 T4 dataset(s) to convert
Creating unified class mapping with 25 classes
Processing dataset: dataset_047
  Processed 15 images with 497 annotations
Conversion completed successfully!
```

### Multiple Datasets
```bash
$ python -m t4_to_yolo_converter.main "datasets/" "yolo_batch" --list
Found 59 T4 dataset(s)
```

## YOLO Output

Creates `dataset.yaml` and organized directories:
```yaml
path: /path/to/yolo/dataset
train: images
val: images  
names:
  0: cone
  1: wall/fence
  # ...
```



## Important Notes

- **Bounding Box Conversion**: T4 `[x1, y1, x2, y2]` → YOLO `[center_x, center_y, width, height]` (normalized)
- **Camera Support**: Different cameras distinguished by filename prefixes
- **Required Files**: T4 datasets need `category.json`, `sample_data.json`, `object_ann.json`

## Installation & Usage

```bash
git clone <repository-url>
cd t4-to-yolo-converter
pip install -e .
# or pip install .
python -m t4_to_yolo_converter.main --help
# or t4-to-yolo --help
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
