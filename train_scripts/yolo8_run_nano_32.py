import sys
import time
from ultralytics import YOLO
yaml_files = [
    "global_train.yaml",
    # "global_train_whole_pathFix.yaml",
    # "global_train_whole.yaml",
    # "global_train811.yaml",
    # "train_China_Drone_811.yaml",
    # "train_China_MotorBike_811.yaml",
    # "train_Czech_811.yaml",
    # "train_India_811.yaml",
    # "train_Japan_811.yaml",
    # "train_Norway_811.yaml",
    # "train_United_States_811.yaml",
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

batch_sizes = [32]

learning_rates = [0.01] #[0.001, 0.0005, 0.0001] #, 0.001, 0.0001]
lrf1 = 0.01

optimizer = "SGD" #"auto"
model_names = ["yolov8n"] #["yolov8s", "yolov8m", "yolov8l"]
device_num = [1]
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