#!/usr/bin/env python
"""生成新旧模型测试结果对比柱状图。"""
import csv
import numpy as np
import matplotlib.pyplot as plt

OLD_CSV = 'Swin-Unet/model_out/Synapse_旧版_验证集bug/result_figures/results.csv'
NEW_CSV = 'Swin-Unet/model_out/Synapse/result_figures/results.csv'
OUTPUT = 'Swin-Unet/model_out/Synapse/result_figures/test_comparison.png'

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

old = load(OLD_CSV)
new = load(NEW_CSV)
organs = ['Aorta','Gallbladder','Kidney(L)','Kidney(R)','Liver','Pancreas','Spleen','Stomach']

old_dsc = [np.mean([float(r[o + '_Dice']) for r in old]) for o in organs]
new_dsc = [np.mean([float(r[o + '_Dice']) for r in new]) for o in organs]
old_hd = [np.mean([float(r[o + '_HD95']) for r in old]) for o in organs]
new_hd = [np.mean([float(r[o + '_HD95']) for r in new]) for o in organs]

om = np.mean([float(r['Mean_Dice']) for r in old])
nm = np.mean([float(r['Mean_Dice']) for r in new])
ohm = np.mean([float(r['Mean_HD95']) for r in old])
nhm = np.mean([float(r['Mean_HD95']) for r in new])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

x = np.arange(len(organs))
w = 0.35

# DSC 柱状图
b1 = ax1.bar(x - w/2, old_dsc, w, label=f'Old: avg {om:.4f}', color='#d62728', alpha=0.85, edgecolor='black')
b2 = ax1.bar(x + w/2, new_dsc, w, label=f'New: avg {nm:.4f}', color='#1f77b4', alpha=0.85, edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(organs, rotation=25, ha='right', fontsize=10)
ax1.set_ylabel('Dice Score', fontsize=12)
ax1.set_title('Per-Class DSC Comparison', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(b1, old_dsc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='#d62728')
for bar, val in zip(b2, new_dsc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='#1f77b4')

# HD95 柱状图
b3 = ax2.bar(x - w/2, old_hd, w, label=f'Old: avg {ohm:.2f}', color='#d62728', alpha=0.85, edgecolor='black')
b4 = ax2.bar(x + w/2, new_hd, w, label=f'New: avg {nhm:.2f}', color='#1f77b4', alpha=0.85, edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(organs, rotation=25, ha='right', fontsize=10)
ax2.set_ylabel('HD95 (mm)', fontsize=12)
ax2.set_title('Per-Class HD95 Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle(
    f'Test Results: Old (150ep, 18cases) vs New (300ep, 15+3 split)  |  '
    f'DSC: {om:.4f} → {nm:.4f} ({"+" if nm > om else ""}{nm-om:+.4f})  |  '
    f'HD95: {ohm:.2f} → {nhm:.2f} ({"+" if nhm > ohm else ""}{nhm-ohm:+.2f})',
    fontsize=12, fontweight='bold', y=1.03)
fig.tight_layout()
fig.savefig(OUTPUT, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUTPUT}')
