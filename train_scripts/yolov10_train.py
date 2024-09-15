import time
import torch
from ultralytics import YOLO
import os
import yaml
import argparse

def update_yaml_paths(yaml_file):
    current_dir = os.getcwd()
    new_path = os.path.join(current_dir, "data/sample")
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)
    if 'path' in yaml_data:
        yaml_data['path'] = new_path
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(yaml_data, file)

def update_yaml_paths_dataset(yaml_file):
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, "..", "RDD2022"))
    data_root = parent_dir
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data['path'] = data_root
    yaml_data['train'] = 'glob_train.txt'
    yaml_data['val'] = 'glob_val.txt'
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(yaml_data, file)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="Update YAML paths and run the training.")
    parser.add_argument('--dataset', action='store_true', help="Update YAML paths and run the training.")
    return parser.parse_args()
def main():
    args = parse_args()

    yaml_files = [
        "global_train.yaml",
        # "train_China_Drone.yaml",
        # "train_China_MotorBike.yaml",
        # "train_Czech.yaml",
        # "train_India.yaml",
        # "train_Japan.yaml",
        # "train_Norway.yaml",
        # "train_United_States.yaml"
    ]

    datasets = [
        "global_WHOLE"
        # "China_Drone",
        # "China_MotorBike",
        # "Czech",
        # "India",
        # "Japan",
        # "Norway",
        # "United_States"
    ]
    
    if args.dataset:
        yaml_files = [os.path.abspath(os.path.join('..', 'RDD2022', y)) for y in yaml_files]
        update_yaml_paths_dataset(yaml_files[0])
    elif  args.test:
        update_yaml_paths('global_train.yaml')
        
    batch_sizes = [16,32]
    learning_rates = [0.01] #[0.001, 0.0005, 0.0001]
    lrf1 = 0.01

    optimizer = "SGD" #"auto"
    model_names = ["yolov10n"] #["yolov8s", "yolov8m", "yolov8l"]
    ###### Check if CUDA is available and set the device accordingly
    if torch.cuda.is_available():
        device_num = [0]  # Use GPU 0 if available
    else:
        device_num = ['cpu']  # Fallback to CPU if GPU is not available
    image_size = 640
    for yaml_file, dataset in zip(yaml_files, datasets):
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for model_name in model_names:
                    name = f"{dataset}_{optimizer}_{batch_size}_lr_{learning_rate}_{lrf1}_{model_name}_{image_size}"
                    start_time = time.time()
                    print("=" * 72)
                    print(f"Training with {yaml_file}, BatchSize={batch_size}, LearningRate={learning_rate}, Model={model_name}, Name={name}")
                    print("=" * 72)
                    model = YOLO(f'{model_name}.pt')
                    results = model.train(
                        data=f'{yaml_file}',
                        epochs=200,
                        imgsz=image_size,
                        device=device_num,
                        batch=batch_size,
                        name=name,
                        optimizer=optimizer,
                        save_period = 25,
                        cos_lr=True,
                        patience=0 
                    )

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    hours, remainder = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(remainder, 60)

                    print("#" * 69)
                    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
                    print("#" * 69)
                    
if __name__ == "__main__":
    main()
