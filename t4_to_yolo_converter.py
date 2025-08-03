#!/usr/bin/env python3
"""
Unified T4 to YOLO format converter
Automatically detects single or multiple T4 datasets and converts accordingly.

T4 format structure:
- category.json: class definitions with tokens and names
- sample_data.json: image files with dimensions and tokens
- object_ann.json: bounding box annotations with bbox [x1, y1, x2, y2] format

YOLO format structure:
- dataset.yaml: class names mapping
- images/: image files
- labels/: annotation files (.txt) with format: class_id center_x center_y width height (normalized)
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


def load_json(file_path: str) -> List[Dict]:
    """Load JSON file and return data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_t4_dataset(directory: Path) -> bool:
    """Check if directory is a T4 dataset (contains annotation folder with required files)"""
    annotation_dir = directory / "annotation"
    if not annotation_dir.exists():
        return False
    
    required_files = [
        annotation_dir / "category.json",
        annotation_dir / "sample_data.json", 
        annotation_dir / "object_ann.json"
    ]
    return all(f.exists() for f in required_files)


def find_t4_datasets(base_dir: Path) -> List[Path]:
    """Find all T4 dataset directories"""
    t4_datasets = []
    
    # Check if base_dir itself is a T4 dataset
    if is_t4_dataset(base_dir):
        t4_datasets.append(base_dir)
    else:
        # Search for T4 datasets in subdirectories
        for item in base_dir.iterdir():
            if item.is_dir() and is_t4_dataset(item):
                t4_datasets.append(item)
    
    return sorted(t4_datasets)


def create_class_mapping(category_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create class mapping from category.json
    Returns:
        token_to_id: mapping from category token to class ID
        id_to_name: mapping from class ID to class name
    """
    categories = load_json(category_file)
    
    token_to_id = {}
    id_to_name = {}
    
    for idx, category in enumerate(categories):
        if 'token' in category and 'name' in category:
            token_to_id[category['token']] = idx
            id_to_name[idx] = category['name']
    
    return token_to_id, id_to_name


def merge_class_mappings(category_files: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Merge class mappings from multiple T4 datasets"""
    all_categories = set()
    
    # Collect all unique categories
    for category_file in category_files:
        categories = load_json(category_file)
        for category in categories:
            if 'name' in category and category['name']:
                all_categories.add(category['name'])
    
    # Create consistent mapping
    sorted_categories = sorted(all_categories)
    id_to_name = {i: name for i, name in enumerate(sorted_categories)}
    name_to_id = {name: i for i, name in enumerate(sorted_categories)}
    
    return name_to_id, id_to_name


def create_sample_data_mapping(sample_data_file: str) -> Dict[str, Dict]:
    """
    Create mapping from sample_data_token to image info
    Returns dict with sample_data_token as key and image info as value
    """
    sample_data = load_json(sample_data_file)
    
    token_to_info = {}
    for data in sample_data:
        if 'token' in data and 'filename' in data:
            # Only process image files (jpg, png)
            if data.get('fileformat', '').lower() in ['jpg', 'jpeg', 'png']:
                token_to_info[data['token']] = {
                    'filename': data['filename'],
                    'width': data.get('width', 0),
                    'height': data.get('height', 0),
                    'sample_token': data.get('sample_token', ''),
                    'calibrated_sensor_token': data.get('calibrated_sensor_token', '')
                }
    
    return token_to_info


def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert T4 bbox format [x1, y1, x2, y2] to YOLO format [center_x, center_y, width, height]
    All values normalized by image dimensions
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center coordinates and dimensions
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize by image dimensions
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return [center_x_norm, center_y_norm, width_norm, height_norm]


def get_unique_filename(filename: str, dataset_prefix: str = "") -> str:
    """
    Generate unique filename by including camera name and optional dataset prefix
    Example: data/CAM_FRONT_NARROW/00001.jpg -> [dataset_]CAM_FRONT_NARROW_00001.jpg
    """
    path_parts = Path(filename).parts
    if len(path_parts) >= 2:
        camera_name = path_parts[-2]  # Get camera folder name
        file_name = path_parts[-1]    # Get actual filename
        name_stem = Path(file_name).stem
        extension = Path(file_name).suffix
        
        if dataset_prefix:
            return f"{dataset_prefix}_{camera_name}_{name_stem}{extension}"
        else:
            return f"{camera_name}_{name_stem}{extension}"
    else:
        if dataset_prefix:
            return f"{dataset_prefix}_{Path(filename).name}"
        else:
            return Path(filename).name


def process_annotations(object_ann_file: str, token_to_id: Dict[str, int], 
                       sample_data_mapping: Dict[str, Dict], dataset_prefix: str = "") -> Dict[str, List[str]]:
    """
    Process object annotations and convert to YOLO format
    Returns dict with unique image filename as key and list of YOLO annotation lines as value
    """
    annotations = load_json(object_ann_file)
    
    # Group annotations by sample_data_token (image)
    image_annotations = {}
    
    for ann in annotations:
        if not all(key in ann for key in ['sample_data_token', 'category_token', 'bbox']):
            continue
            
        sample_data_token = ann['sample_data_token']
        category_token = ann['category_token']
        bbox = ann['bbox']
        
        # Skip if sample_data_token not found in mapping (not an image)
        if sample_data_token not in sample_data_mapping:
            continue
            
        # Skip if category_token not found in class mapping
        if category_token not in token_to_id:
            continue
            
        image_info = sample_data_mapping[sample_data_token]
        img_width = image_info['width']
        img_height = image_info['height']
        
        if img_width <= 0 or img_height <= 0:
            continue
            
        # Convert bbox to YOLO format
        yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
        class_id = token_to_id[category_token]
        
        # Create YOLO annotation line
        yolo_line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
        
        # Get unique image filename
        original_filename = image_info['filename']
        unique_filename = get_unique_filename(original_filename, dataset_prefix)
        
        if unique_filename not in image_annotations:
            image_annotations[unique_filename] = []
        image_annotations[unique_filename].append(yolo_line)
    
    return image_annotations


def convert_single_dataset(t4_dataset_path: Path, output_dir: Path, 
                          unified_token_to_id: Dict[str, int], camera_filter: str = None) -> Tuple[int, int]:
    """
    Convert a single T4 dataset
    Returns: (images_copied, total_annotations)
    """
    print(f"Processing dataset: {t4_dataset_path.name}")
    
    # Paths to annotation files
    annotation_dir = t4_dataset_path / "annotation"
    category_file = annotation_dir / "category.json"
    sample_data_file = annotation_dir / "sample_data.json"
    object_ann_file = annotation_dir / "object_ann.json"
    
    # Create local category token to unified class ID mapping
    categories = load_json(str(category_file))
    local_token_to_id = {}
    for category in categories:
        if 'token' in category and 'name' in category and category['name']:
            if category['name'] in unified_token_to_id:
                local_token_to_id[category['token']] = unified_token_to_id[category['name']]
    
    # Create sample data mapping
    sample_data_mapping = create_sample_data_mapping(str(sample_data_file))
    
    # Apply camera filter if specified
    if camera_filter:
        filtered_mapping = {}
        for token, info in sample_data_mapping.items():
            if camera_filter in info['filename']:
                filtered_mapping[token] = info
        sample_data_mapping = filtered_mapping
        print(f"  Filtered to {len(sample_data_mapping)} images from camera: {camera_filter}")
    else:
        print(f"  Found {len(sample_data_mapping)} image files from all cameras")
    
    # For multiple datasets, use dataset name as prefix
    dataset_prefix = t4_dataset_path.name if len(list(t4_dataset_path.parent.iterdir())) > 1 else ""
    
    # Process annotations
    image_annotations = process_annotations(str(object_ann_file), local_token_to_id, 
                                          sample_data_mapping, dataset_prefix)
    print(f"  Processed annotations for {len(image_annotations)} images")
    
    if not image_annotations:
        print(f"  No valid annotations found for dataset: {t4_dataset_path.name}")
        return 0, 0
    
    # Setup output directories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create reverse mapping from unique filename to original filename
    unique_to_original = {}
    for token, info in sample_data_mapping.items():
        original_filename = info['filename']
        unique_filename = get_unique_filename(original_filename, dataset_prefix)
        unique_to_original[unique_filename] = original_filename
    
    # Copy images and create labels
    images_copied = 0
    total_annotations = 0
    
    for unique_filename, yolo_lines in image_annotations.items():
        if unique_filename not in unique_to_original:
            continue
            
        original_filename = unique_to_original[unique_filename]
        
        # Source and destination paths
        src_image_path = t4_dataset_path / original_filename
        dst_image_path = images_dir / unique_filename
        
        label_name = Path(unique_filename).stem + ".txt"
        label_path = labels_dir / label_name
        
        # Copy image if it exists and not already copied
        if src_image_path.exists():
            if not dst_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)
                images_copied += 1
                print(f"    Copied: {original_filename} -> {unique_filename}")
            
            # Create label file
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines) + '\n')
            total_annotations += len(yolo_lines)
            print(f"    Created label: {label_name} with {len(yolo_lines)} annotations")
        else:
            print(f"    Warning: Image not found: {src_image_path}")
    
    return images_copied, total_annotations


def create_dataset_yaml(yolo_output_path: str, id_to_name: Dict[int, str]):
    """Create dataset.yaml file for YOLO"""
    
    dataset_config = {
        'path': str(Path(yolo_output_path).absolute()),
        'train': 'images',
        'val': 'images',  # Same as train for now, you can split later
        'test': 'images',
        'names': id_to_name
    }
    
    yaml_path = Path(yolo_output_path) / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Created dataset config: {yaml_path}")


def convert_t4_to_yolo(input_path: str, output_path: str, camera_filter: str = None):
    """
    Main function to convert T4 format to YOLO format
    Automatically detects single or multiple datasets
    
    Args:
        input_path: Path to T4 dataset directory or directory containing multiple T4 datasets
        output_path: Path to output YOLO dataset directory
        camera_filter: Optional camera name filter (e.g., 'CAM_FRONT_NARROW')
    """
    
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    # Find T4 datasets
    t4_datasets = find_t4_datasets(input_dir)
    
    if not t4_datasets:
        raise FileNotFoundError(f"No T4 datasets found in: {input_path}")
    
    print(f"Found {len(t4_datasets)} T4 dataset(s) to convert:")
    for i, dataset in enumerate(t4_datasets, 1):
        print(f"  {i}. {dataset.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unified class mapping
    print("\nCreating unified class mapping...")
    if len(t4_datasets) == 1:
        # Single dataset - use its class mapping directly
        category_file = t4_datasets[0] / "annotation" / "category.json"
        _, id_to_name = create_class_mapping(str(category_file))
        name_to_id = {name: idx for idx, name in id_to_name.items()}
    else:
        # Multiple datasets - merge class mappings
        category_files = [str(dataset / "annotation" / "category.json") for dataset in t4_datasets]
        name_to_id, id_to_name = merge_class_mappings(category_files)
    
    print(f"Unified class mapping created with {len(id_to_name)} classes")
    
    # Convert all datasets
    total_images = 0
    total_annotations = 0
    
    print(f"\nStarting conversion with camera filter: {camera_filter or 'All cameras'}")
    
    for dataset_path in t4_datasets:
        images_copied, annotations_count = convert_single_dataset(
            dataset_path, output_dir, name_to_id, camera_filter
        )
        total_images += images_copied
        total_annotations += annotations_count
    
    # Create dataset.yaml
    print("\nCreating dataset configuration...")
    create_dataset_yaml(str(output_dir), id_to_name)
    
    # Summary
    print("\nConversion completed successfully!")
    print(f"Total datasets processed: {len(t4_datasets)}")
    print(f"Total images copied: {total_images}")
    print(f"Total annotations created: {total_annotations}")
    print(f"YOLO dataset saved to: {output_dir}")
    print("\nDataset structure:")
    print(f"├── {output_dir}/")
    print("│   ├── dataset.yaml")
    print("│   ├── images/")
    print("│   └── labels/")


def main():
    parser = argparse.ArgumentParser(description="Convert T4 format to YOLO format (Unified)")
    parser.add_argument("input_path", help="Path to T4 dataset directory or directory containing T4 datasets")
    parser.add_argument("output_path", help="Path to output YOLO dataset directory")
    parser.add_argument("--camera", help="Filter by camera name (e.g., CAM_FRONT_NARROW)", default=None)
    parser.add_argument("--list", action="store_true", help="List found T4 datasets and exit")
    
    args = parser.parse_args()
    
    try:
        input_dir = Path(args.input_path)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {args.input_path}")
            return 1
        
        # Find and list datasets
        t4_datasets = find_t4_datasets(input_dir)
        
        if not t4_datasets:
            print(f"No T4 datasets found in: {args.input_path}")
            return 1
        
        if args.list:
            print(f"Found {len(t4_datasets)} T4 dataset(s):")
            for i, dataset in enumerate(t4_datasets, 1):
                print(f"  {i}. {dataset}")
            return 0
        
        convert_t4_to_yolo(args.input_path, args.output_path, args.camera)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
