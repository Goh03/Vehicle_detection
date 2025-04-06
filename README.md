# Installation

Hmmmmmm

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
