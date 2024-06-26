This is a forked repository of [Ultralytics](https://ultralytics.com) that utilizes [ceva's](https://www.ceva-ip.com) LiteML package to quantize YOLOv8 model. For any other example, follow the official repository ([YOLOv8](https://github.com/ultralytics/ultralytics)).

## <div align="center">Documentation</div>

See below for a quickstart installation and usage example.

<details open>
<summary>Install</summary>
Create an environment with Python>=3.8.

```bash
conda create yolov8-liteml python==3.8
conda activate yolov8-liteml
```

Install LiteML package.
  
```bash
unzip LiteML.zip
cd LiteML/
bash ./install.sh
pip install schema
```

Change directory to the desried location for installation. Clone ultralytics_LiteML repository, navigate to the cloned directory and pip install.

```bash
git clone https://github.com/CEVA-AI-Labs/ultralytics_LiteML.git
cd ultralytics_LiteML
pip install -e .
pip install dill
pip install pycocotools==2.0.7
```

</details>

<details open>
<summary>Usage</summary>
  
Run [yolov8_liteml_qat](https://github.com/CEVA-AI-Labs/ultralytics_LiteML/blob/main/examples/YOLOv8-LiteML-QAT/yolov8_liteml_qat.py) example for performing QAT on a pretrained model, or run [yolov8_validate_pretrained_qat](https://github.com/CEVA-AI-Labs/ultralytics_LiteML/blob/main/examples/YOLOv8-LiteML-QAT/yolov8_validate_pretrained_qat.py) to validate a pretrained QAT model.

### Python
Create a config.yaml file. Select number of bits for the data and weights quantization. Set per_channel to True for per channel quantization or False for per tensor quantization.
```yaml
QAT:
    fully_quantized: False
    device: 'cuda'
    data_quantization:
      status: On
      bits: 8
      custom_bits: { }
      symmetric: False
      static: { "status": False,
                "observer": ''
                }
      per_channel: True

    weights_quantization:
      status: On
      bits: 4
      symmetric: False
      custom_bits:  {}
      per_channel: True
```

Perform QAT on a pretrained model.

```python
from ultralytics import YOLO
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Wrap DetectionModel with LiteML
retrainer_cfg = RetrainerConfig("liteml_config.yaml")
model.model = RetrainerModel(model.model, config=retrainer_cfg).to(device)

# Train the model
results = model.train(data='coco_ailabs.yaml', epochs=10, imgsz=640, save_period=1, fraction=0.01, device=device)

# Validate the model
metrics = model.val(data='coco_ailabs.yaml')
```

</details>
