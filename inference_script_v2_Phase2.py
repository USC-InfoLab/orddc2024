from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import yaml
import sys
import subprocess
import shutil
from datetime import datetime

from ensemble_boxes import weighted_boxes_fusion, nms, non_maximum_weighted, soft_nms
from orddc_2024.predictors.ultralytics_predictor import UltralyticsPredictor
from orddc_2024.predictors.yolov5_predictor import Yolov5Predictor
# from orddc_2024.predictors.megvii_predictor import MegviiPredictor

def download_models():
    print("Downloading models using gdown...")
    # url = 'https://drive.google.com/uc?id=1-1i6SWWMxsPGURhCxVJ8Gq2-V5tYXYBX' # 60_bak
    url = 'https://drive.google.com/uc?id=1-I-9AnU9PkRroDja7gmA_5rrguSzC7us' # 60
    output_zip = './models_ph2.zip'
    subprocess.run(['gdown', url, '--output', output_zip], check=True)
    shutil.unpack_archive(output_zip, './')
    os.remove(output_zip)

    print("Model download and extraction complete.")
    
def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_prediction(framework, predictor_class, model_params, images_path):
    print(f"Running predictions for framework: {framework}")
    def predict_model(model_param):
        predictor = predictor_class(framework, [model_param])
        predictor.load([model_param], images_path)
        return predictor.predict(), predictor.images

    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(predict_model, model_param) for model_param in model_params]
        for future in as_completed(futures):
            # try:
            results.append(future.result())
            # except Exception as e:
                # print(f"{framework} model failed with error: {e}")

    combined_predictions = [[], [], []]
    images = None
    for (predictions, img_list) in results:
        if images is None:
            images = img_list
        combined_predictions[0].extend(predictions[0])
        combined_predictions[1].extend(predictions[1])
        combined_predictions[2].extend(predictions[2])

    return combined_predictions, images

def main(yaml_file, images_path, output_csv):
    start_time = datetime.now()
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = load_yaml_config(yaml_file)
    ultra_models_params = config['models'].get('yolov8', [])
    yolov5_weights_params = config['models'].get('yolov5', [])
    # yolox_weights_params = config['models'].get('yolox', [])
    print("YOLOv8 Model Parameters:", ultra_models_params)
    print("YOLOv5 Model Parameters:", yolov5_weights_params)
    # print("YOLOX Model Parameters:", yolox_weights_params)
    predictions = {
        'yolov8': (UltralyticsPredictor, ultra_models_params),
        'yolov5': (Yolov5Predictor, yolov5_weights_params),
        # 'yolox': (MegviiPredictor, yolox_weights_params)
    }
    for framework, (predictor_class, model_params) in predictions.items():
        if model_params:
            print(f"{framework} loaded with {len(model_params)} model(s).")
            
            # Check if all weight paths exist
            all_weights_exist = True
            for param in model_params:
                weight_path = param.get('weight')
                if weight_path and os.path.exists(weight_path):
                    print(f"  - {framework} weight loaded from {weight_path}")
                else:
                    print(f"  - {framework} weight missing or invalid path: {weight_path}")
                    all_weights_exist = False
            
            if not all_weights_exist:
                print(f"Warning: Some weights for {framework} are missing or have invalid paths.")
        else:
            print(f"{framework} has no models loaded.")
    results = {}
    predictor_instance = None 

    with ProcessPoolExecutor() as executor:
        futures = {}
        for framework, (predictor_class, model_params) in predictions.items():
            if model_params and all(os.path.exists(param['weight']) for param in model_params):
                print(f"Submitting prediction task for framework: {framework}")
                future = executor.submit(run_prediction, framework, predictor_class, model_params, images_path)
                futures[future] = framework

        for future in as_completed(futures):
            framework = futures[future]
            # try:
            results[framework] = future.result()
            predictor_instance = predictions[framework][0]("repo_name", framework) 
            # except Exception as e:
            #     print(f"{framework} prediction failed with error: {e}")
    yolov8_predictions, yolov8_images = results.get('yolov8', (([], [], []), []))
    yolov5_predictions, yolov5_images = results.get('yolov5', (([], [], []), []))
    # yolox_predictions, yolox_images = results.get('yolox', (([], [], []), []))
    
    end_time = datetime.now()  
    elapsed_time = end_time - start_time  
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}") 
    print(f"Elapsed Time: {elapsed_time}") 
    
    ## Combine predictions
    print("Ensemble Predictions")
    boxes_list = yolov8_predictions[0] + yolov5_predictions[0] # + yolox_predictions[0]
    scores_list = yolov8_predictions[1] + yolov5_predictions[1] # + yolox_predictions[1]
    labels_list = yolov8_predictions[2] + yolov5_predictions[2] # + yolox_predictions[2]

    if yolov8_images:
        images = yolov8_images
    elif yolov5_images:
        images = yolov5_images
    # elif yolox_images:
    #     images = yolox_images
    else:
        print("No images available.")
        return

    ## Weighted Boxes Fusion (WBF)
    ensembled_boxes, ensembled_scores, ensembled_labels = [], [], []
    weights = [1] * len(boxes_list)
    iou_thr = 0.999
    skip_box_thr = 0.0001
    sigma=0.1
    
    for i in range(len(images)):
        image_boxes_list = [model_boxes[i] for model_boxes in boxes_list]
        image_scores_list = [model_scores[i] for model_scores in scores_list]
        image_labels_list = [model_labels[i] for model_labels in labels_list]

        boxes, scores, labels = weighted_boxes_fusion(
            image_boxes_list, image_scores_list, image_labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        ensembled_boxes.append(boxes)
        ensembled_scores.append(scores)
        ensembled_labels.append(labels)

    ## Save ensembled results to a CSV file
    with open(output_csv, 'w') as f:
        for img_idx, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            img_width, img_height = predictor_instance.get_image_size(img_path)

            result_list = [
                f"{int(label)} {int(box[0] * img_width)} {int(box[1] * img_height)} {int(box[2] * img_width)} {int(box[3] * img_height)}"
                for box, score, label in zip(ensembled_boxes[img_idx], ensembled_scores[img_idx], ensembled_labels[img_idx])
            ]

            result_str = ' '.join(result_list)
            trimmed_result_str = " ".join(result_str.split()[:2500])

            f.write(f"{img_name},{trimmed_result_str}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference_script_v2.py <images_path> <output_csv>")
        sys.exit(1)

    yaml_file = "./model_ph2.yaml"
    images_path = sys.argv[1]
    output_csv = sys.argv[2]
    
    ## Check if the models directory exists, if not download the models
    if not os.path.exists('./models_ph2'):
        print("Models directory './models_ph2' does not exist.")
        download_models()
        
    main(yaml_file, images_path, output_csv)
    print("models_ph2_Prediction done.")
