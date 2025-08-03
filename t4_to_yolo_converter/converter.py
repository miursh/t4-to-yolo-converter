
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_t4_dataset(directory: Path) -> bool:
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
    t4_datasets = []
    if is_t4_dataset(base_dir):
        t4_datasets.append(base_dir)
    else:
        for item in base_dir.iterdir():
            if item.is_dir() and is_t4_dataset(item):
                t4_datasets.append(item)
    return sorted(t4_datasets)

def create_class_mapping(category_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    categories = load_json(category_file)
    token_to_id = {}
    id_to_name = {}
    for idx, category in enumerate(categories):
        if 'token' in category and 'name' in category:
            token_to_id[category['token']] = idx
            id_to_name[idx] = category['name']
    return token_to_id, id_to_name

def merge_class_mappings(category_files: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    all_categories = set()
    for category_file in category_files:
        categories = load_json(category_file)
        for category in categories:
            if 'name' in category and category['name']:
                all_categories.add(category['name'])
    sorted_categories = sorted(all_categories)
    id_to_name = {i: name for i, name in enumerate(sorted_categories)}
    name_to_id = {name: i for i, name in enumerate(sorted_categories)}
    return name_to_id, id_to_name

def create_sample_data_mapping(sample_data_file: str) -> Dict[str, Dict]:
    sample_data = load_json(sample_data_file)
    token_to_info = {}
    for data in sample_data:
        if 'token' in data and 'filename' in data:
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
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return [center_x_norm, center_y_norm, width_norm, height_norm]

def get_unique_filename(filename: str, dataset_prefix: str = "") -> str:
    path_parts = Path(filename).parts
    if len(path_parts) >= 2:
        camera_name = path_parts[-2]
        file_name = path_parts[-1]
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
    annotations = load_json(object_ann_file)
    image_annotations = {}
    for ann in annotations:
        if not all(key in ann for key in ['sample_data_token', 'category_token', 'bbox']):
            continue
        sample_data_token = ann['sample_data_token']
        category_token = ann['category_token']
        bbox = ann['bbox']
        if sample_data_token not in sample_data_mapping:
            continue
        if category_token not in token_to_id:
            continue
        image_info = sample_data_mapping[sample_data_token]
        img_width = image_info['width']
        img_height = image_info['height']
        if img_width <= 0 or img_height <= 0:
            continue
        yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
        class_id = token_to_id[category_token]
        yolo_line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
        original_filename = image_info['filename']
        unique_filename = get_unique_filename(original_filename, dataset_prefix)
        if unique_filename not in image_annotations:
            image_annotations[unique_filename] = []
        image_annotations[unique_filename].append(yolo_line)
    return image_annotations

def convert_single_dataset(t4_dataset_path: Path, output_dir: Path, 
                          unified_token_to_id: Dict[str, int], camera_filter: str = None, allowed_class_names=None, id_to_name=None) -> Tuple[int, int]:
    print(f"Processing dataset: {t4_dataset_path.name}")
    annotation_dir = t4_dataset_path / "annotation"
    category_file = annotation_dir / "category.json"
    sample_data_file = annotation_dir / "sample_data.json"
    object_ann_file = annotation_dir / "object_ann.json"
    categories = load_json(str(category_file))
    local_token_to_id = {}
    for category in categories:
        if 'token' in category and 'name' in category and category['name']:
            if category['name'] in unified_token_to_id:
                local_token_to_id[category['token']] = unified_token_to_id[category['name']]
    sample_data_mapping = create_sample_data_mapping(str(sample_data_file))
    if camera_filter:
        filtered_mapping = {}
        for token, info in sample_data_mapping.items():
            if camera_filter in info['filename']:
                filtered_mapping[token] = info
        sample_data_mapping = filtered_mapping
        print(f"  Filtered to {len(sample_data_mapping)} images from camera: {camera_filter}")
    else:
        print(f"  Found {len(sample_data_mapping)} image files from all cameras")
    dataset_prefix = t4_dataset_path.name if len(list(t4_dataset_path.parent.iterdir())) > 1 else ""
    image_annotations = process_annotations(str(object_ann_file), local_token_to_id, 
                                          sample_data_mapping, dataset_prefix)
    print(f"  Processed annotations for {len(image_annotations)} images")
    if not image_annotations:
        print(f"  No valid annotations found for dataset: {t4_dataset_path.name}")
        return 0, 0
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    unique_to_original = {}
    for token, info in sample_data_mapping.items():
        original_filename = info['filename']
        unique_filename = get_unique_filename(original_filename, dataset_prefix)
        unique_to_original[unique_filename] = original_filename
    images_copied = 0
    total_annotations = 0
    for unique_filename, yolo_lines in image_annotations.items():
        if unique_filename not in unique_to_original:
            continue
        # Filter by allowed_class_names if specified
        filtered_lines = []
        for yolo_line in yolo_lines:
            class_id = int(yolo_line.split()[0])
            if allowed_class_names and id_to_name:
                if id_to_name.get(class_id) not in allowed_class_names:
                    continue
            filtered_lines.append(yolo_line)
        if not filtered_lines:
            continue
        original_filename = unique_to_original[unique_filename]
        src_image_path = t4_dataset_path / original_filename
        dst_image_path = images_dir / unique_filename
        label_name = Path(unique_filename).stem + ".txt"
        label_path = labels_dir / label_name
        if src_image_path.exists():
            if not dst_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)
                images_copied += 1
                print(f"    Copied: {original_filename} -> {unique_filename}")
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(filtered_lines) + '\n')
            total_annotations += len(filtered_lines)
            print(f"    Created label: {label_name} with {len(filtered_lines)} annotations")
        else:
            print(f"    Warning: Image not found: {src_image_path}")
    return images_copied, total_annotations

def create_dataset_yaml(yolo_output_path: str, id_to_name: Dict[int, str]):
    dataset_config = {
        'path': str(Path(yolo_output_path).absolute()),
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'names': id_to_name
    }
    yaml_path = Path(yolo_output_path) / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
    print(f"Created dataset config: {yaml_path}")

def convert_t4_to_yolo(input_path: str, output_path: str, camera_filter: str = None, allowed_class_names=None):
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    t4_datasets = find_t4_datasets(input_dir)
    if not t4_datasets:
        raise FileNotFoundError(f"No T4 datasets found in: {input_path}")
    print(f"Found {len(t4_datasets)} T4 dataset(s) to convert:")
    for i, dataset in enumerate(t4_datasets, 1):
        print(f"  {i}. {dataset.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nCreating unified class mapping...")
    if len(t4_datasets) == 1:
        category_file = t4_datasets[0] / "annotation" / "category.json"
        _, id_to_name = create_class_mapping(str(category_file))
        name_to_id = {name: idx for idx, name in id_to_name.items()}
    else:
        category_files = [str(dataset / "annotation" / "category.json") for dataset in t4_datasets]
        name_to_id, id_to_name = merge_class_mappings(category_files)
    print(f"Unified class mapping created with {len(id_to_name)} classes")
    total_images = 0
    total_annotations = 0
    print(f"\nStarting conversion with camera filter: {camera_filter or 'All cameras'}")
    allowed_class_names_set = set(allowed_class_names) if allowed_class_names else None
    for dataset_path in t4_datasets:
        images_copied, annotations_count = convert_single_dataset(
            dataset_path, output_dir, name_to_id, camera_filter, allowed_class_names_set, id_to_name
        )
        total_images += images_copied
        total_annotations += annotations_count
    print("\nCreating dataset configuration...")
    create_dataset_yaml(str(output_dir), id_to_name)
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
