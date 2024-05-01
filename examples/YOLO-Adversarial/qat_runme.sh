#conda activate /projects/vbu_projects/users/royj/conda_envs/liteml_trustai/
# yolov3 - 1h per epoch on gpu 1
#python qat.py --model yolov3.pt --cfg configs/w4a4_per_channel_per_channel.yaml --epochs 10 --fraction 1
#python qat.py --model yolov3.pt --cfg configs/w4a8_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qat.py --model yolov3.pt --cfg configs/w5a5_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qat.py --model yolov3.pt --cfg configs/w6a6_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qat.py --model yolov3.pt --cfg configs/w8a8_per_channel_per_channel.yaml --epochs 3 --fraction 1

# yolov5n - 23m per epoch on gpu 1
#python qat.py --model yolov5n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qat.py --model yolov5n.pt --cfg configs/w4a8_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qat.py --model yolov5n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qat.py --model yolov5n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qat.py --model yolov5n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --epochs 3 --fraction 1

# yolov8n - 23m per epoch on gpu 1
python qat.py --model yolov8n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --epochs 10 --fraction 1
python qat.py --model yolov8n.pt --cfg configs/w4a8_per_channel_per_channel.yaml --epochs 10 --fraction 1
python qat.py --model yolov8n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --epochs 10 --fraction 1
python qat.py --model yolov8n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --epochs 10 --fraction 1
#python qat.py --model yolov8n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --epochs 3 --fraction 1

# yolov3 - 1h per epoch on gpu 1
#python qat.py --model yolov3.pt --cfg configs/w4a4_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov3.pt --cfg configs/w4a8_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov3.pt --cfg configs/w5a5_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov3.pt --cfg configs/w6a6_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov3.pt --cfg configs/w8a8_per_channel_per_tensor.yaml --epochs 3 --fraction 1

# yolov5n - 23m per epoch on gpu 1
#python qat.py --model yolov5n.pt --cfg configs/w4a4_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov5n.pt --cfg configs/w4a8_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov5n.pt --cfg configs/w5a5_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov5n.pt --cfg configs/w6a6_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov5n.pt --cfg configs/w8a8_per_channel_per_tensor.yaml --epochs 3 --fraction 1

# yolov8n - 23m per epoch on gpu 1
#python qat.py --model yolov8n.pt --cfg configs/w4a4_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov8n.pt --cfg configs/w4a8_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov8n.pt --cfg configs/w5a5_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov8n.pt --cfg configs/w6a6_per_channel_per_tensor.yaml --epochs 3 --fraction 1
#python qat.py --model yolov8n.pt --cfg configs/w8a8_per_channel_per_tensor.yaml --epochs 3 --fraction 1