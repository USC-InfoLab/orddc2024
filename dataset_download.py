import requests
import os
import shutil
import xml.etree.ElementTree as ET
import yaml
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd

def download_file(url, dest_folder, dest_filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # Get total file size
    block_size = 1024
    progress = 0
    
    if response.status_code == 200:
        os.makedirs(dest_folder, exist_ok=True)
        file_path = os.path.join(dest_folder, dest_filename)
        
        with open(file_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                progress += len(data)
                percent_complete = (progress / total_size) * 100
                print(f"\rDownloading: {percent_complete:.2f}%", end='') 
        print(f"\nDownload completed: {file_path}")
        return file_path
    else:
        print(f"Failed to download file: {response.status_code}")
        return None

def unzip_file(zip_path):
    shutil.unpack_archive(zip_path, './')
    print(f"Unzip completed")
    os.remove(zip_path)  # Delete the ZIP file after unzipping
    print(f"ZIP file deleted: {zip_path}")

def unzip_datasets(datasets, base_path):
    for dataset in datasets:
        zip_filename = f'{dataset}.zip'
        zip_path = os.path.join(base_path, zip_filename)
        dataset_folder = os.path.join(base_path, dataset)
        
        if not os.path.exists(dataset_folder):
            print(f"Extracting {zip_path}...")
            shutil.unpack_archive(zip_path, base_path)  # Unzip the archive
            os.remove(zip_path)  # Remove the zip file after extraction
            print(f"Extracted and removed {zip_filename}")
        else:
            print(f"{dataset} is already extracted.")

def convert_to_yolo_format(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def process_dataset(dataset, base_path, class_mapping):
    xml_folder_path = os.path.join(base_path, dataset, 'train', 'annotations', 'xmls')
    labels_folder_path = os.path.join(base_path, dataset, 'train', 'labels')

    os.makedirs(labels_folder_path, exist_ok=True)

    for root, dirs, files in os.walk(xml_folder_path):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                tree = ET.parse(xml_path)
                xml_root = tree.getroot()
                size = xml_root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)

                yolo_labels = []

                for obj in xml_root.findall('object'):
                    label = obj.find('name').text
                    if label in class_mapping:
                        class_id = class_mapping[label]
                        bndbox = obj.find('bndbox')
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        bbox = (xmin, ymin, xmax, ymax)
                        yolo_bbox = convert_to_yolo_format(bbox, img_width, img_height)

                        yolo_labels.append(f"{class_id} " + " ".join(map(str, yolo_bbox)))

                if yolo_labels:
                    label_filename = file.replace('.xml', '.txt')
                    label_path = os.path.join(labels_folder_path, label_filename)

                    with open(label_path, 'w') as f:
                        f.write("\n".join(yolo_labels) + "\n")

def parse_labels(base_path, class_mapping, image_list, dataset):
    labels_folder_path = os.path.join(base_path, dataset, 'train', 'labels')
    
    class_counter = Counter()

    for image in image_list:
        image_name = os.path.basename(image).rsplit('.', 1)[0]
        label_path = os.path.join(labels_folder_path, image_name + '.txt')

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_id)]
                    class_counter[class_name] += 1
    return class_counter

def create_location_txt(datasets,base_path,class_mapping):
    stats = []
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        images_folder_path = os.path.join(base_path, dataset, 'train', 'images')
        all_images = [os.path.join(images_folder_path, file) for file in os.listdir(images_folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
        # Split the images into train+val and test (90:10 ratio)
        train_images, val_images = train_test_split(all_images, test_size=0.1, random_state=42)
        # Parse labels and calculate class distribution
        train_class_counter = parse_labels(base_path, class_mapping,train_images, dataset)
        val_class_counter = parse_labels(base_path, class_mapping,val_images, dataset)
        train_images_relative = [os.path.relpath(image, os.path.join(base_path, dataset)).replace("\\", "/").replace(f"{dataset}/", "") for image in train_images]
        val_images_relative = [os.path.relpath(image, os.path.join(base_path, dataset)).replace("\\", "/").replace(f"{dataset}/", "") for image in val_images]

        stats.append({
            'Dataset': dataset,
            'Total Images': len(all_images),
            'Train Images': len(train_images),
            'Val Images': len(val_images),
            'Train D00': train_class_counter['D00'],
            'Train D10': train_class_counter['D10'],
            'Train D20': train_class_counter['D20'],
            'Train D40': train_class_counter['D40'],
            'Val D00': val_class_counter['D00'],
            'Val D10': val_class_counter['D10'],
            'Val D20': val_class_counter['D20'],
            'Val D40': val_class_counter['D40'],
        })

        train_txt_path = os.path.join(base_path, dataset, 'train.txt')
        val_txt_path = os.path.join(base_path, dataset, 'val.txt')

        with open(train_txt_path, 'w') as f:
            for image in train_images_relative:
                f.write(f"{image}\n")

        with open(val_txt_path, 'w') as f:
            for image in val_images_relative:
                f.write(f"{image}\n")

        print(f"Finished processing dataset: {dataset}")

    df_stats = pd.DataFrame(stats)
    print(df_stats)
    print("All datasets processed successfully.")

def concatenate_txt_files(datasets,base_path,file_type):
    concatenated_lines = []
    total_lines = 0
    for dataset in datasets:
        file_path = os.path.join(base_path, dataset, file_type)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                line_count = len(lines)
                total_lines += line_count
                for line in lines:
                    updated_line = os.path.join(dataset, line.lstrip("./")).replace("\\", "/")
                    concatenated_lines.append(updated_line)
                print(f"{file_path}: {line_count} lines")

    output_file_path = os.path.join(base_path, f'glob_{file_type}')
    with open(output_file_path, 'w') as f:
        f.writelines(concatenated_lines)
    
    print(f"Total lines in glob_{file_type}: {total_lines}")

def create_dataset_yaml(base_path, dataset):
    data = {
        'train': f'./{dataset}/train.txt',
        'val': f'./{dataset}/val.txt',
        'nc': 4,
        'names': {
            0: 'D00',
            1: 'D10',
            2: 'D20',
            3: 'D40'
        }
    }
    yaml_file_path = os.path.join(base_path, f'train_{dataset}.yaml')
    with open(yaml_file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created {yaml_file_path}")

def create_global_yaml(base_path):
    data = {
        'train': './glob_train.txt',
        'val': './glob_val.txt',
        'nc': 4,
        'names': {
            0: 'D00',
            1: 'D10',
            2: 'D20',
            3: 'D40'
        }
    }
    yaml_file_path = os.path.join(base_path, 'global_train.yaml')
    with open(yaml_file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created {yaml_file_path}")

def main():
    ###Step 0: Download Dataset
    url = 'https://figshare.com/ndownloader/files/38030910'
    dest_folder = './'
    zip_filename = 'RDD2022_released_through_CRDDC2022.zip'
    zip_file_path = download_file(url, dest_folder, zip_filename)
    if zip_file_path:
        unzip_file(zip_file_path)
    ###Step 1: Unzip and Preprocessing Dataset
    base_path = './RDD2022'
    datasets = ['China_Drone', 'China_MotorBike', 'Czech', 'India', 'Japan', 'Norway', 'United_States']
    class_mapping = {'D00': 0, 'D10': 1, 'D20': 2, 'D40': 3}
    file_types = ['train.txt', 'val.txt']
    unzip_datasets(datasets, base_path)
    ###Step 2: Convert XML annotations to YOLO format
    # Process each dataset for labels [XML to YOLO txt]
    for dataset in datasets:
        process_dataset(dataset, base_path, class_mapping)
    create_location_txt(datasets,base_path,class_mapping)
    ### Step 3: Create txt files for trining
    ## Create txt Location files
    for file_type in file_types:
        concatenate_txt_files(datasets,base_path,file_type)
    ### Step 4: Create YAML file
    ## Create global YAML file
    create_global_yaml(base_path)
    for dataset in datasets:
        create_dataset_yaml(base_path, dataset)
if __name__ == "__main__":
    main()