import argparse
import os
from roboflow import Roboflow
from ultralytics import YOLO

def download_dataset(api_key, workspace, project_name, version_number, format="yolov11"):
    """Downloads the dataset from Roboflow."""
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(format)
    return dataset.location

def train_model(data_yaml_path, model_name='yolo11s.pt', epochs=200, imgsz=640, batch=16, project_name='yolov11n_letilapia', patience=50):
    """Trains a YOLO model."""
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=project_name,
        patience=patience
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Download dataset and train YOLO model.")

    # Dataset arguments
    parser.add_argument("--api_key", type=str, required=True, help="Roboflow API Key")
    parser.add_argument("--workspace", type=str, default="transferlearning-z9nnr", help="Roboflow workspace name")
    parser.add_argument("--project_name", type=str, default="letilapia-igyjs", help="Roboflow project name")
    parser.add_argument("--version_number", type=int, default=3, help="Roboflow dataset version number")
    parser.add_argument("--dataset_format", type=str, default="yolov11", help="Dataset format (e.g., yolov11)")

    # Training arguments
    parser.add_argument("--model_base", type=str, default="yolo11s.pt", help="Base YOLO model to use for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--run_name", type=str, default="yolov11n_letilapia", help="Name for the training run")
    parser.add_argument("--patience", type=int, default=50, help="Patience for early stopping")

    args = parser.parse_args()

    print("--- Downloading Dataset ---")
    dataset_location = download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project_name=args.project_name,
        version_number=args.version_number,
        format=args.dataset_format
    )
    data_yaml_path = os.path.join(dataset_location, "data.yaml")
    print(f"Dataset downloaded to: {dataset_location}")
    print(f"Path to data.yaml: {data_yaml_path}")

    print("\n--- Training Model ---")
    train_model(
        data_yaml_path=data_yaml_path,
        model_name=args.model_base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project_name=args.run_name,
        patience=args.patience
    )
    print("Training complete.")

if __name__ == "__main__":
    main()
