This is a forked repository of [Ultralytics](https://ultralytics.com) that utilizes [ceva's](https://www.ceva-ip.com) LiteML package to quantize YOLOv8 model. For any other example, follow the official repository ([YOLOv8](https://github.com/ultralytics/ultralytics)).

## <div align="center">Documentation</div>

See below for a quickstart installation and usage example.

<details open>
<summary>Install</summary>
Create an environment with Python>=3.8.
  
Install LiteML package.
  
```bash
unzip LiteML.zip -d LiteML
cd LiteML/
bash ./install.sh
```

Clone ultralytics_LiteML repository, navigate to the cloned directory and pip install.

```bash
git clone https://github.com/CEVA-AI-Labs/ultralytics_LiteML.git
cd ultralytics
pip install -e .
pip install dill
```

</details>

<details open>
<summary>Usage</summary>
  
Run [YOLOv8-LiteML](https://github.com/CEVA-AI-Labs/ultralytics_LiteML/tree/main/examples/YOLOv8-LiteML) example for performing QAT on a pretrained model.

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
                "observer":
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

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Wrap DetectionModel with LiteML
retrainer_cfg = RetrainerConfig("liteml_config.yaml")
model.model = RetrainerModel(model.model, config=retrainer_cfg).cuda()

# Train the model
results = model.train(data='coco_ailabs.yaml', epochs=10, imgsz=640, device=0)

# Validate the model
metrics = model.val(data='coco_ailabs.yaml')
```

</details>
