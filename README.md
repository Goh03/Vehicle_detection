# Vehicle detection

## Dataset
This research utilizes the vehicle detection dataset from the SOICT Hackathon 2024 Competition hosted by AI-Hub. The dataset is designed to evaluate object detection models under real-world conditions, focusing on the challenges posed by lighting variations between daytime and nighttime. You can access the dataset here:
https://aihub.ml/competitions/755.

Additionally, the dataset, along with the split data, enhanced images, weights, and yaml files, is available on Kaggle:
https://www.kaggle.com/datasets/bchutrn/dataset.

## Yolo 

```python
# Clone the YOLOv5 repository
git clone https://github.com/ultralytics/yolov5

# Navigate to the cloned directory
cd yolov5

# Install required packages
pip install -r requirements.txt

# Train
#Replace train.py, vehicle.yaml and best.pt with your file paths
python train.py --img 640 --batch 16 --epochs 20 --data vehicle.yaml --weights best.pt

# Evaluation
#Replace val.py, best.pt and vehicle.yaml with your file paths
python val.py --weights best.pt --data vehicle.yaml --img 640

# Inference
#Replace detect.py, best.pt and "val" with your file paths
python detect.py --weights best.pt --source "val"
```
## RT-DETR 

```python
pip install ultralytics
from ultralytics import RTDETR

# Load pretrained model
# Replace "best.pt" with your file path
model = RTDETR("best.pt")

# Train 
# Replace "vehicle.yaml" with your file path
results = model.train(data="vehicle.yaml", epochs=20, imgsz=640)

# Evaluation
# Replace "vehicle.yaml" with your file path
val_results = model.val(data="vehicle.yaml", imgsz=640, save=True)

# Inference 
# Replace "test data" with your test set file path
results = model("test dataset")
```

## Retinexformer
### Installation

Follow these steps to set up the environment and install dependencies. We recommend using Conda for environment management.

**Prerequisites:**
*   Conda installed
*   NVIDIA GPU with CUDA 11.8 compatible drivers (if using GPU acceleration)

**Steps:**

1.  **Create Conda Environment:**
    Open your terminal and run the following command to create a new Conda environment named `torch2` with Python 3.9:
    ```bash
    conda create -n torch2 python=3.9 -y
    ```

2.  **Activate Environment:**
    Activate the newly created environment:
    ```bash
    conda activate torch2
    ```
    *(You'll need to run this command every time you want to work on this project in a new terminal session.)*

3.  **Install Core Dependencies (PyTorch with CUDA):**
    Install PyTorch, TorchVision, TorchAudio, and the appropriate CUDA toolkit version using Conda:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    *Note: Ensure `pytorch-cuda=11.8` matches your system's CUDA capabilities. Adjust if necessary.*

4.  **Install Other Pip Dependencies:**
    Install the remaining Python packages using pip:
    ```bash
    pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
    pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
    ```

5.  **Install BasicSR:**
    Navigate to the root directory of this project (where `setup.py` is located) and install the BasicSR library in development mode. The `--no_cuda_ext` flag skips compiling custom CUDA extensions, which might simplify installation if you don't need them or encounter build issues.
    ```bash
    python setup.py develop --no_cuda_ext
    ```

### Testing / Inference

To run inference using the pre-trained models:

1.  **Download Pre-trained Models:**
    *   Download the necessary model weights from:
        *   **Baidu Disk:** [[Link to Baidu Disk](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2)] (Extraction code: `cyh2`)
        *   **Google Drive:** [[Link to Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link)]
    *   Create a folder named `pretrained_weights` in the root directory of the Retinexformer repository.
    *   Place the downloaded model file(s) (e.g., `LOL_v2_synthetic.pth`) inside the `pretrained_weights` folder.

2.  **Run Inference Script:**
    Make sure your `torch2` conda environment is activated.

    *   **Example for LOL-v2-synthetic dataset:**
        Run the following command from the project's root directory:
        ```bash
        python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic
        ```