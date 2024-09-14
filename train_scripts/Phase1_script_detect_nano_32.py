from ultralytics import YOLO
import glob
import os
import numpy as np

test_dataset_path = "./orddc/team_usc/datasets/split_dataset/"
images_list = glob.glob(os.path.join(test_dataset_path, "test/images/*.jpg"))

############################################
ious = [0.999]
confs = [0.25, 0.3, 0.2, 0.15]
size = 960
############################################
#### v8s_16 // v8n_32 // v8n_16 // v8s_32
model_name = "global_WHOLE_SGD_32_lr_0.01_0.01_yolov8n_960"
csv_folder_name = "./CSV_results_nano"
csv_subfolder = f"{csv_folder_name}/{model_name}"

os.makedirs(csv_folder_name, exist_ok=True)
os.makedirs(csv_subfolder, exist_ok=True)
############################################

weights_output_pairs = [
    (f"/runs/detect/global_WHOLE_SGD_32_lr_0.01_0.01_yolov8n_960/weights/last.pt", f"./{csv_folder_name}/{model_name}/{model_name}_last.csv"),
]
weights_output_pairs += [
    (f"/runs/detect/global_WHOLE_SGD_32_lr_0.01_0.01_yolov8n_960/weights/epoch{epoch}.pt", f"./{csv_folder_name}/{model_name}/{model_name}_epoch{epoch}.csv")
    for epoch in [175,150,125,100,75,50,25] #[130, 150, 180, 100, 80, 60, 50, 30]
]

############################################
for ioui in ious:
    for confi in confs:
        for weights, output_file in weights_output_pairs:
            model = YOLO(weights)
            output_file_iou_conf = output_file.replace('.csv', f'_iou{ioui}_conf{confi}_size{size}.csv')
            with open(output_file_iou_conf, "w") as f:
                for img in images_list:
                    results = model.predict(source=img, conf=confi, iou=ioui, imgsz=size, augment=True, agnostic_nms=True, save_txt=False, save=False, device=0)
                    assert len(results) == 1
                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy()  
                        cls1 = result.boxes.cls.view(-1,1).cpu().numpy()
                        cls1 = cls1.astype(int) + 1
                        conf = result.boxes.conf.view(-1,1).cpu().numpy()
                        out_tensor = np.concatenate([cls1, boxes], axis=-1).astype(int)
                        filename = os.path.basename(result.path)
                        f.write(filename + ',')
                        for box in out_tensor:
                            f.write(' '.join(map(str, box)) + ' ')
                        f.write('\n')
                        print(f"Using IoU: {ioui}, Confidence: {confi}, {output_file_iou_conf}")
                        
######################################################
