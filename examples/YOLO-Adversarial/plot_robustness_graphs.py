
from matplotlib import pyplot as plt
import pandas as pd

def plot_robustness_curves_2_inputs(clean_df, attack_df, figure_name):
    map_float_clean = clean_df.loc['float', 'mAP50_95']
    map_clean = clean_df['mAP50_95'].drop('float').to_list()
    map_float_attack = attack_df.loc['float', 'mAP50_95']
    map_attack = attack_df['mAP50_95'].drop('float').to_list()
    x = clean_df.index.drop('float').to_list()

    plt.figure(1)
    plt.plot(x, map_clean, color='b', marker='o', label='clean')
    plt.axhline(y=map_float_clean, color='b', linestyle='--', label='float clean')
    plt.plot(x, map_attack, color='r', marker='o', label='attacked')
    plt.axhline(y=map_float_attack, color='r', linestyle='--', label='float attacked')

    plt.legend()
    plt.xlabel('bits')
    plt.ylabel('mAP50-95')
    plt.xticks(x)
    plt.savefig(f"runs/robustness_test/{figure_name}.png")

def plot_robustness_curves_4_inputs(clean_df, attack_df, qaat_clean_df, qaat_attack_df, figure_name):
    map_float_clean = clean_df.loc['float', 'mAP50_95']
    map_clean = clean_df['mAP50_95'].drop('float').to_list()
    map_float_attack = attack_df.loc['float', 'mAP50_95']
    map_attack = attack_df['mAP50_95'].drop('float').to_list()
    map_float_clean_qaat = qaat_clean_df.loc['float', 'mAP50_95']
    map_clean_qaat = qaat_clean_df['mAP50_95'].drop('float').to_list()
    map_float_attack_qaat = qaat_attack_df.loc['float', 'mAP50_95']
    map_attack_qaat = qaat_attack_df['mAP50_95'].drop('float').to_list()
    x = clean_df.index.drop('float').to_list()

    plt.figure(1)
    plt.plot(x, map_clean, color='b', marker='o', label='clean')
    plt.axhline(y=map_float_clean, color='b', linestyle='--', label='float clean')
    plt.plot(x, map_attack, color='r', marker='o', label='attacked')
    plt.axhline(y=map_float_attack, color='r', linestyle='--', label='float attacked')

    plt.plot(x, map_clean_qaat, color='g', marker='o', label='clean QAAT')
    plt.axhline(y=map_float_clean_qaat, color='g', linestyle='--', label='float clean QAAT')
    plt.plot(x, map_attack_qaat, color='y', marker='o', label='attacked QAAT')
    plt.axhline(y=map_float_attack_qaat, color='y', linestyle='--', label='float attacked QAAT')
    plt.legend()
    plt.xlabel('bits')
    plt.ylabel('mAP50-95')
    plt.xticks(x)
    plt.savefig(f"runs/robustness_test/{figure_name}.png")


def plot_robustness_curves_2_figures(clean_df, attack_df, qaat_clean_df, qaat_attack_df, figure_name):
    map_float_clean = clean_df.loc['float', 'mAP50_95']
    map_clean = clean_df['mAP50_95'].drop('float').to_list()
    map_float_attack = attack_df.loc['float', 'mAP50_95']
    map_attack = attack_df['mAP50_95'].drop('float').to_list()
    map_float_clean_qaat = qaat_clean_df.loc['float', 'mAP50_95']
    map_clean_qaat = qaat_clean_df['mAP50_95'].drop('float').to_list()
    map_float_attack_qaat = qaat_attack_df.loc['float', 'mAP50_95']
    map_attack_qaat = qaat_attack_df['mAP50_95'].drop('float').to_list()
    x = clean_df.index.drop('float').to_list()
    map_max = max(map_float_clean, map_float_clean_qaat) + 0.1

    plt.figure(1)
    plt.plot(x, map_clean, color='b', marker='o', label='clean')
    plt.axhline(y=map_float_clean, color='b', linestyle='--', label='float clean')
    plt.plot(x, map_attack, color='r', marker='o', label='attacked')
    plt.axhline(y=map_float_attack, color='r', linestyle='--', label='float attacked')
    plt.legend()
    plt.xlabel('bits')
    plt.ylabel('mAP50-95')
    plt.ylim((0, map_max))
    plt.xticks(x)
    plt.savefig(f"runs/robustness_test/{figure_name}_PTQ.png")

    plt.figure(2)
    plt.plot(x, map_clean_qaat, color='g', marker='o', label='clean QAAT')
    plt.axhline(y=map_float_clean_qaat, color='g', linestyle='--', label='float clean QAAT')
    plt.plot(x, map_attack_qaat, color='y', marker='o', label='attacked QAAT')
    plt.axhline(y=map_float_attack_qaat, color='y', linestyle='--', label='float attacked QAAT')
    plt.legend()
    plt.xlabel('bits')
    plt.ylabel('mAP50-95')
    plt.ylim((0, map_max))
    plt.xticks(x)
    plt.savefig(f"runs/robustness_test/{figure_name}_QAAT.png")

def create_figure_name(yolo_version, attacked_model, quantization_granularity, qat):
    """ create figure name according to the test """
    if 'yolov3' in yolo_version:
        yolo_name = 'yolov3'
    elif 'yolov5n' in yolo_version:
        yolo_name = 'yolov5n'
    elif 'yolov8n' in yolo_version:
        yolo_name = 'yolov8n'
    fig_name = f'{yolo_name}_attacked-model-{attacked_model}_quantization-{quantization_granularity}'
    return fig_name


def create_table(dict):
    results = []
    for key, path in dict.items():
        df = pd.read_csv(path)
        row = {'bits': key, 'mAP50_95': df['mAP50_95'][0], 'mAP50': df['mAP50'][0]}
        results.append(row)
    df_results = pd.DataFrame(results).set_index('bits')
    return df_results


def main():
    figure_name = 'robustness_curve_ptq'
    yolo_model = 'yolov8n' #  ['yolov3', 'yolov5n', 'yolov8n']
    figure_name = yolo_model + '_' + figure_name
    ptq_clean_dict = {'float': f'runs/detect/{yolo_model}/robustness/float/map.csv',
                   4: f'runs/detect/{yolo_model}/robustness/w4a4_ptq/map.csv',
                   5: f'runs/detect/{yolo_model}/robustness/w5a5_ptq/map.csv',
                   6: f'runs/detect/{yolo_model}/robustness/w6a6_ptq/map.csv',
                   7: f'runs/detect/{yolo_model}/robustness/w7a7_ptq/map.csv',
                   8: f'runs/detect/{yolo_model}/robustness/w8a8_ptq/map.csv',
                   }
    qaat_clean_dict = {'float': f'runs/detect/{yolo_model}/robustness/float_qaat/map.csv',
                      4: f'runs/detect/{yolo_model}/robustness/w4a4_qaat/map.csv',
                      5: f'runs/detect/{yolo_model}/robustness/w5a5_qaat/map.csv',
                      6: f'runs/detect/{yolo_model}/robustness/w6a6_qaat/map.csv',
                      7: f'runs/detect/{yolo_model}/robustness/w7a7_qaat/map.csv',
                      8: f'runs/detect/{yolo_model}/robustness/w8a8_qaat/map.csv',
                      }
    ptq_attack_dict = {'float': f'runs/detect/{yolo_model}/robustness/float_attack/map.csv',
                   4: f'runs/detect/{yolo_model}/robustness/w4a4_ptq_attack/map.csv',
                   5: f'runs/detect/{yolo_model}/robustness/w5a5_ptq_attack/map.csv',
                   6: f'runs/detect/{yolo_model}/robustness/w6a6_ptq_attack/map.csv',
                   7: f'runs/detect/{yolo_model}/robustness/w7a7_ptq_attack/map.csv',
                   8: f'runs/detect/{yolo_model}/robustness/w8a8_ptq_attack/map.csv',
                   }
    qaat_attack_dict = {'float': f'runs/detect/{yolo_model}/robustness/float_qaat_attack/map.csv',
                      4: f'runs/detect/{yolo_model}/robustness/w4a4_qaat_attack/map.csv',
                      5: f'runs/detect/{yolo_model}/robustness/w5a5_qaat_attack/map.csv',
                      6: f'runs/detect/{yolo_model}/robustness/w6a6_qaat_attack/map.csv',
                      7: f'runs/detect/{yolo_model}/robustness/w7a7_qaat_attack/map.csv',
                      8: f'runs/detect/{yolo_model}/robustness/w8a8_qaat_attack/map.csv',
                      }
    clean_df = create_table(ptq_clean_dict)
    attack_df = create_table(ptq_attack_dict)
    # clean_qaat_df = create_table(qaat_clean_dict)
    # attack_qaat_df = create_table(qaat_attack_dict)

    plot_robustness_curves_2_inputs(clean_df, attack_df, figure_name)
    # plot_robustness_curves_4_inputs(clean_df, attack_df, clean_qaat_df, attack_qaat_df, figure_name)
    # plot_robustness_curves_2_figures(clean_df, attack_df, clean_qaat_df, attack_qaat_df, figure_name)


if __name__ == '__main__':
    main()
