#conda activate /projects/vbu_projects/users/royj/conda_envs/liteml_trustai/
# yolov3 - ?? per epoch on gpu 1
#python qaat.py --model yolov3.pt --cfg configs/w4a4_per_channel_per_channel.yaml --epochs 10 --fraction 1
#python qaat.py --model yolov3.pt --cfg configs/w4a8_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qaat.py --model yolov3.pt --cfg configs/w5a5_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qaat.py --model yolov3.pt --cfg configs/w6a6_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qaat.py --model yolov3.pt --cfg configs/w8a8_per_channel_per_channel.yaml --epochs 3 --fraction 1

# QAT only
# yolov5n - ?? per epoch on gpu 1
#python qaat.py --model yolov5n.pt --cfg float --at True --epochs 3 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --epochs 3 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --epochs 3 --fraction 1

# QAT + Adversarial training = QAAT
#python qaat.py --model yolov5n.pt --cfg float --at True --epochs 20 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --at True --epochs 20 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --at True --epochs 20 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --at True --epochs 20 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --at True --epochs 20 --fraction 1
#python qaat.py --model yolov5n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --at True --epochs 20 --fraction 1

# QAT + Adversarial training = QAAT
# yolov8n - 4h?? per epoch on gpu 2
#python qaat.py --model yolov8n.pt --cfg float --at True --epochs 3 --fraction 1
#python qaat.py --model yolov8n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --at True --epochs 3 --fraction 1
#python qaat.py --model yolov8n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --at True --epochs 3 --fraction 1
#python qaat.py --model yolov8n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --at True --epochs 3 --fraction 1
#python qaat.py --model yolov8n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --at True --epochs 3 --fraction 1
python qaat.py --model yolov8n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --at True --epochs 3 --fraction 1

# QAT only
# yolov8n - 30m per epoch on gpu 2
python qaat.py --model yolov8n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --epochs 3 --fraction 1
python qaat.py --model yolov8n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --epochs 3 --fraction 1
python qaat.py --model yolov8n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --epochs 3 --fraction 1
python qaat.py --model yolov8n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --epochs 3 --fraction 1
python qaat.py --model yolov8n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --epochs 3 --fraction 1