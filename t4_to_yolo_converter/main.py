import sys
from .converter import convert_t4_to_yolo
import argparse
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser(description="Convert T4 format to YOLO format (Unified)")
    parser.add_argument("input_path", help="Path to T4 dataset directory or directory containing T4 datasets")
    parser.add_argument("output_path", help="Path to output YOLO dataset directory")
    parser.add_argument("--camera", help="Filter by camera name (e.g., CAM_FRONT_NARROW)", default=None)
    parser.add_argument("--classes", help="Comma-separated list of class names to convert (e.g. car,motorcycle,bicycle,pedestrian)", default=None)
    parser.add_argument("--classes-file", help="Path to a YAML file containing a class list (key: classes)", default=None)
    parser.add_argument("--list", action="store_true", help="List found T4 datasets and exit")
    args = parser.parse_args()

    try:
        input_dir = Path(args.input_path)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {args.input_path}")
            return 1
        # Find and list datasets
        from .converter import find_t4_datasets
        t4_datasets = find_t4_datasets(input_dir)
        if not t4_datasets:
            print(f"No T4 datasets found in: {args.input_path}")
            return 1
        if args.list:
            print(f"Found {len(t4_datasets)} T4 dataset(s):")
            for i, dataset in enumerate(t4_datasets, 1):
                print(f"  {i}. {dataset}")
            return 0
        class_list = None
        if args.classes and args.classes_file:
            print("Error: --classes and --classes-file cannot be used together.")
            return 1
        if args.classes:
            class_list = [c.strip() for c in args.classes.split(",") if c.strip()]
        elif args.classes_file:
            with open(args.classes_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
                    class_list = [str(c).strip() for c in data["classes"] if str(c).strip()]
                else:
                    print("Error: YAML file must contain a 'classes' key with a list of class names.")
                    return 1
        convert_t4_to_yolo(args.input_path, args.output_path, args.camera, class_list)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
