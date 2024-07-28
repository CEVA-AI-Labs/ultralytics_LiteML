# ************* yolov5n *************
# 1. PTQ attack
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/float.yaml --quantization_type float --attack True --results_path yolov5n/robustness/float_attack
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov5n/robustness/w4a4_ptq_attack
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov5n/robustness/w5a5_ptq_attack
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov5n/robustness/w6a6_ptq_attack
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov5n/robustness/w7a7_ptq_attack
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov5n/robustness/w8a8_ptq_attack

# 1. PTQ attack - adv examples generated on float model
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov5n/robustness/w4a4_ptq_attack_adv_float
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov5n/robustness/w5a5_ptq_attack_adv_float
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov5n/robustness/w6a6_ptq_attack_adv_float
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov5n/robustness/w7a7_ptq_attack_adv_float
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov5n/robustness/w8a8_ptq_attack_adv_float

# 2. PTQ No attack
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/float.yaml --quantization_type float --results_path yolov5n/robustness/float
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov5n/robustness/w4a4_ptq
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov5n/robustness/w5a5_ptq
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov5n/robustness/w6a6_ptq
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov5n/robustness/w7a7_ptq
#python robustness_evaluation.py --model yolov5n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov5n/robustness/w8a8_ptq

# 3. QAAT attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/float_e-3_f-1.0_AT/weights/best.pt --cfg configs/float.yaml --quantization_type float --attack True --results_path yolov5n/robustness/float_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w4a4_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --attack True --results_path yolov5n/robustness/w4a4_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w5a5_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --attack True --results_path yolov5n/robustness/w5a5_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w6a6_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --attack True --results_path yolov5n/robustness/w6a6_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w7a7_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --attack True --results_path yolov5n/robustness/w7a7_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w8a8_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt  --quantization_type QAT --attack True --results_path yolov5n/robustness/w8a8_qaat_attack

# 4. QAAT no attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/float_e-3_f-1.0_AT/weights/best.pt --cfg configs/float.yaml --quantization_type float --results_path yolov5n/robustness/float_qaat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w4a4_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --results_path yolov5n/robustness/w4a4_qaat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w5a5_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --results_path yolov5n/robustness/w5a5_qaat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w6a6_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --results_path yolov5n/robustness/w6a6_qaat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w7a7_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --results_path yolov5n/robustness/w7a7_qaat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAAT/w8a8_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt  --quantization_type QAT --results_path yolov5n/robustness/w8a8_qaat

# 5. QAT attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w4a4_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --attack True --results_path yolov5n/robustness/w4a4_qat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w5a5_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --attack True --results_path yolov5n/robustness/w5a5_qat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w6a6_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --attack True --results_path yolov5n/robustness/w6a6_qat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w7a7_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --attack True --results_path yolov5n/robustness/w7a7_qat_attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w8a8_per_channel_per_channel_e-3_f-1.0/weights/best.pt  --quantization_type QAT --attack True --results_path yolov5n/robustness/w8a8_qat_attack

# 6. QAT no attack
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w4a4_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --results_path yolov5n/robustness/w4a4_qat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w5a5_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --results_path yolov5n/robustness/w5a5_qat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w6a6_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --results_path yolov5n/robustness/w6a6_qat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w7a7_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --results_path yolov5n/robustness/w7a7_qat
#python robustness_evaluation.py --model runs/detect/yolov5n/QAT/w8a8_per_channel_per_channel_e-3_f-1.0/weights/best.pt  --quantization_type QAT --results_path yolov5n/robustness/w8a8_qat


# *************************************

# ************* YOLOv8 *************
# 1. attack
python robustness_evaluation.py --model yolov8n.pt --cfg configs/float.yaml --quantization_type PTQ --attack True --results_path yolov8n/robustness/float_attack_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov8n/robustness/w4a4_ptq_attack_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov8n/robustness/w5a5_ptq_attack_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov8n/robustness/w6a6_ptq_attack_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov8n/robustness/w7a7_ptq_attack_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --quantization_type PTQ --attack True --results_path yolov8n/robustness/w8a8_ptq_attack_v2

# 1. PTQ attack - adv examples generated on float model
#python robustness_evaluation.py --model yolov8n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov8n/robustness/w4a4_ptq_attack_adv_float
#python robustness_evaluation.py --model yolov8n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov8n/robustness/w5a5_ptq_attack_adv_float
#python robustness_evaluation.py --model yolov8n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov8n/robustness/w6a6_ptq_attack_adv_float
#python robustness_evaluation.py --model yolov8n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov8n/robustness/w7a7_ptq_attack_adv_float
#python robustness_evaluation.py --model yolov8n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --quantization_type PTQ --attack True --attacked_model float --results_path yolov8n/robustness/w8a8_ptq_attack_adv_float

# 2. No attack
python robustness_evaluation.py --model yolov8n.pt --cfg configs/float.yaml --quantization_type PTQ --results_path yolov8n/robustness/float_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w4a4_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov8n/robustness/w4a4_ptq_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w5a5_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov8n/robustness/w5a5_ptq_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w6a6_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov8n/robustness/w6a6_ptq_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w7a7_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov8n/robustness/w7a7_ptq_v2
python robustness_evaluation.py --model yolov8n.pt --cfg configs/w8a8_per_channel_per_channel.yaml --quantization_type PTQ --results_path yolov8n/robustness/w8a8_ptq_v2

# 3. QAAT attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/float_e-3_f-1.0_AT/weights/best.pt --cfg configs/float.yaml --quantization_type float --attack True --results_path yolov8n/robustness/float_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w4a4_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --attack True --results_path yolov8n/robustness/w4a4_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w5a5_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --attack True --results_path yolov8n/robustness/w5a5_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w6a6_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --attack True --results_path yolov8n/robustness/w6a6_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w7a7_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --attack True --results_path yolov8n/robustness/w7a7_qaat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w8a8_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt  --quantization_type QAT --attack True --results_path yolov8n/robustness/w8a8_qaat_attack

# 4. QAAT no attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/float_e-3_f-1.0_AT/weights/best.pt --cfg configs/float.yaml --quantization_type float --results_path yolov8n/robustness/float_qaat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w4a4_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --results_path yolov8n/robustness/w4a4_qaat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w5a5_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --results_path yolov8n/robustness/w5a5_qaat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w6a6_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --results_path yolov8n/robustness/w6a6_qaat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w7a7_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt --quantization_type QAT --results_path yolov8n/robustness/w7a7_qaat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAAT/w8a8_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt  --quantization_type QAT --results_path yolov8n/robustness/w8a8_qaat

# 5. QAT attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w4a4_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --attack True --results_path yolov8n/robustness/w4a4_qat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w5a5_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --attack True --results_path yolov8n/robustness/w5a5_qat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w6a6_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --attack True --results_path yolov8n/robustness/w6a6_qat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w7a7_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --attack True --results_path yolov8n/robustness/w7a7_qat_attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w8a8_per_channel_per_channel_e-3_f-1.0/weights/best.pt  --quantization_type QAT --attack True --results_path yolov8n/robustness/w8a8_qat_attack

# 6. QAT no attack
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w4a4_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --results_path yolov8n/robustness/w4a4_qat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w5a5_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --results_path yolov8n/robustness/w5a5_qat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w6a6_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --results_path yolov8n/robustness/w6a6_qat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w7a7_per_channel_per_channel_e-3_f-1.0/weights/best.pt --quantization_type QAT --results_path yolov8n/robustness/w7a7_qat
#python robustness_evaluation.py --model runs/detect/yolov8n/QAT/w8a8_per_channel_per_channel_e-3_f-1.0/weights/best.pt  --quantization_type QAT --results_path yolov8n/robustness/w8a8_qat

# *************************************