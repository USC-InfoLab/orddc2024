# orddc_2024


# Optimized Road Damage Detection Challenge ([ORDDC'2024](https://orddc2024.sekilab.global/)) Submission

*The models will be automatically downloaded when you run the script.*

## Setup

### 1. Download the Project

You can either unzip the provided ZIP file or clone the repository from GitHub.

**Option 1: Unzip the ZIP File**

```bash
unzip orddc2024.zip -d ./
cd orddc_2024
```

**Option 2: Clone from GitHub**

```bash
git clone https://github.com/USC-InfoLab/orddc2024.git
cd orddc2024
```

### 2. Create and Activate the Conda Environment

Create a new Conda environment named orddc_2024 with **Python 3.10** and activate it:
```bash
conda create -n orddc2024 python=3.10 -y
conda activate orddc2024
```

### 3. Install the required packages

Install all the required packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Detection / Submission

1. Run the inference script using one of the following commands, depending on the phase you are working on.

### For Phase 1 Best Result
> Execute the Phase 1 inference script with:
```bash
python inference_script_v2_Phase1.py ./images ./output_csv_file_Phase1.csv

# Example:
python inference_script_v2_Phase2.py ./train_scripts/data/sample/ ./output_csv_file_Phase1.csv
```
Output: ./Phase1_output_Approach23_Conf_0.3.csv
* Note: When running **inference_script_v2_Phase1.py**, the CSV file **[Phase1_output_Approach23_Conf_0.3.csv](https://drive.google.com/file/d/1a-SuWHjl0WF_upPrHoaP_7UGwq0Qwwe3/view?usp=sharing)** will be downloaded along with the models.

#### Phase 1. submission results (F1-Score)
| Model(img_size)      | Nano | Small | Medium | Large | XLarge |
|:------------:|:------:|:-------:|:--------:|:-------:|:--------:|
| YOLOv5(640)     |  ✓(tiny)   |   ✓   |   ✓    |   ✓   |   ✓    |
| YOLOv8(640)     |  ✓   |   ✓   |   ✓    |   ✓   |   ✓    |
| YOLOv10(640)    |  ✓   |   ✓   |   ✓    |   ✓   |   ✓    |
> Phase 1 Best Score:

| F1-Score (6 countries)  |
|:-----------------------:|
|0.7664                   |

### For Phase 2 Best result
> Execute the Phase 2 inference script with:
```bash
python inference_script_v2_Phase2.py ./images ./output_csv_file_Phase2.csv

# Example:
python inference_script_v2_Phase2.py ./train_scripts/data/sample/ ./output_csv_file_Phase2.csv
```
#### Phase 2. Submission Results (F1-Score and Inference Speed)

| Model(img_size)      | Nano | Small |
|:----------:|:----:|:-----:|
| YOLOv5(640)     |      |   ✓   |
| YOLOv8(960)     |   ✓  |       |
> Phase 1 Best Score:

| F1-Score (6 countries)  | Inference Speed (6 countries) (sec/image) |
|:-----------------------:|:----------------------------------------:|
| 0.7017                  | 0.0432                                   |


### Positional Arguments:
- **./images**: Path to the directory containing images for inference
- **./output_csv_file_Phase1.csv**: output CSV file name including directory name

## Train

> This script is designed to train YOLOv8 models using various configurations specified in the script. It automates the process of running multiple experiments by looping through different datasets, batch sizes, learning rates, and model types. The script outputs the time taken for each training session and saves the trained model at specified intervals.

```bash
cd train_scripts
python yolov8_run_nano_32.py
```

> This script is designed to train YOLOv10 models.
```bash
cd train_scripts
python yolov10_run_nano_32.py
```

### Training Configuration: The script uses predefined configurations for training, which include:

`yaml_files`: The list of configuration files that define the datasets and their specific settings (e.g., global_train.yaml).

`model_names`: YOLOv8 model names to be used for training (e.g., yolov8n).

`datasets`: Corresponding names for the datasets being used.

`batch_sizes`: The batch size for training (e.g., 32 or 16).

`learning_rates`: List of learning rates to experiment with (e.g., [0.01]).

`optimizer`: Optimizer used for training (e.g., SGD).

`device_num`: GPU device number for training (e.g., [0]).

`image_size`: The size of the input image (e.g., 640 or 960).

### Batch Phase 1 Submission Script
> This script is designed to generate CSV files for the **Phase 1 submission** using YOLOv8 and YOLOv10 models.
(You will need to update `test_dataset_path` and `weights_output_pairs` with the paths to the trained models. However, sample files are already located in the train_scripts directory.)
You can run the following command:
```bash
cd train_scripts
python Phase1_script_detect_nano_32.py
```
Submission csv files will be saved under `train_scripts\CSV_results_nano` folder