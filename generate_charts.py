import os
import json
import matplotlib.pyplot as plt
import matplotlib
import torch
matplotlib.use('Agg')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

def get_model_config_from_ckpt(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        config = {'c1': 32, 'c2': 64, 'c3': 64, 'c4': 64, 'c5': 1024}
        for k in state_dict.keys():
            if 'v1.conv.weight' in k: config['c1'] = state_dict[k].shape[0]
            elif 'v2.conv.weight' in k: config['c2'] = state_dict[k].shape[0]
            elif 'v3.conv.weight' in k: config['c3'] = state_dict[k].shape[0]
            elif 'v5.conv.weight' in k: config['c4'] = state_dict[k].shape[0]
            elif 'v9.conv.weight' in k: config['c5'] = state_dict[k].shape[0]
        return config
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def calc_macs(config):

    c1, c2, c3, c4, c5 = config['c1'], config['c2'], config['c3'], config['c4'], config['c5']
    h, w = 640, 232
    macs = 0
    macs += c1 * 3 * 5 * 5 * (h//2) * (w//2)       # v1: s=2 -> h/2
    macs += c2 * c1 * 5 * 5 * (h//8) * (w//8)      # v2: pool1(s=2)+v2(s=2) -> h/8
    macs += c3 * c2 * 5 * 5 * (h//16) * (w//16)    # v3: s=2 -> h/16
    macs += c4 * c3 * 5 * 5 * (h//32) * (w//32)    # v4: s=2 -> h/32
    
    macs += 4 * c4 * c4 * 5 * 5 * (h//64) * (w//64) 
    
    macs += c5 * c4 * 15 * 15 * 1 * 1 
    
    return macs / 1e9  

def load_real_results():
    if not os.path.exists('real_evaluation_results.json'):
        print("Error: real_evaluation_results.json not found!")
        return []
        
    with open('real_evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    baseline_result = next((r for r in results if r.get('model') == 'baseline_best'), None)
    baseline_params = baseline_result.get('params', 15638813) if baseline_result else 15638813
    baseline_config = {'c1': 32, 'c2': 64, 'c3': 64, 'c4': 64, 'c5': 1024}
    baseline_macs = calc_macs(baseline_config)
    
    processed = []
    for r in results:
        model_name = r.get('model', 'unknown')
        config = get_model_config_from_ckpt(f"{model_name}.pth") or baseline_config
        macs = calc_macs(config)
        
        params = r.get('params', baseline_params)
        sparsity = (1 - params / baseline_params) * 100 if baseline_params else 0
        
        processed.append({
            'model': model_name,
            'params_m': params / 1e6,
            'macs_g': macs,
            'sparsity_pct': sparsity,
            'recall': r.get('recall', 0.0),
            'f1': r.get('f1', 0.0),
            'fnr': r.get('fnr', 0.0),
            'latency': r.get('latency', 0.0),
            'fps': r.get('throughput_bs1', 0.0),
        })
    return processed

def generate_master_benchmark_table():
    print("Generating Master Benchmark Table with REAL data...")
    data = load_real_results()
    
    if not data:
        print("No valid data found. Chart generation aborted.")
        return
        
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('off')
    
    headers = ['Method', 'Sparsity(%)', 'Params(M)', 'MACs(G)', 'Recall(%)', 'F1(%)', 'FNR(%)', 'Latency(ms)', 'FPS']
    
    model_order = [
        ('Baseline', ['baseline_best']),
        ('Naive L1', ['naive_l1_r20', 'naive_l1_r40', 'naive_l1_r60']),
        ('Naive FN', ['naive_fn_r20', 'naive_fn_r40', 'naive_fn_r60']),
        ('Ours C1', ['ours_c1_r20', 'ours_c1_r40', 'ours_c1_r60']),
        ('Ours C2', ['ours_c2_r20', 'ours_c2_r40', 'ours_c2_r60']),
        ('Ours C3', ['ours_c3_r20', 'ours_c3_r40', 'ours_c3_r60']),
        ('Ours C4', ['ours_c4_r20', 'ours_c4_r40', 'ours_c4_r60']),
    ]
    
    rows = []
    row_group_map = [] 
    
    group_idx = 0
    for method, models in model_order:
        for model in models:
            d = next((x for x in data if x['model'] == model), None)
            if d:
                rows.append([
                    method, 
                    f"{d['sparsity_pct']:.1f}", 
                    f"{d['params_m']:.2f}", 
                    f"{d['macs_g']:.2f}",
                    f"{d['recall']:.1f}", 
                    f"{d['f1']:.1f}", 
                    f"{d['fnr']:.2f}",
                    f"{d['latency']:.1f}", 
                    f"{d['fps']:.0f}"
                ])
                row_group_map.append(group_idx)
        group_idx += 1
                
    if not rows:
        print("Error: Target models not found in the JSON data.")
        return
        
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2F5597') 
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    bg_colors = ['#FFFFFF', '#E9EBF5']
    for row_idx in range(len(rows)):
        current_group = row_group_map[row_idx]
        bg_color = bg_colors[current_group % 2]
        
        for col_idx in range(len(headers)):
            cell = table[(row_idx + 1, col_idx)]
            cell.set_facecolor(bg_color)
            if col_idx == 0:
                cell.set_text_props(fontweight='bold')
                
    plt.title('Master Benchmark Table (REAL Evaluation Results)', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('chart1_master_benchmark.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ chart1_master_benchmark.png generated successfully.")

if __name__ == '__main__':
    generate_master_benchmark_table()