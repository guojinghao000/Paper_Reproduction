#!/usr/bin/env python
"""对比新旧两次训练的测试结果 (DSC + HD95)。"""
import csv
import numpy as np

OLD_CSV = 'Swin-Unet/model_out/Synapse_旧版_验证集bug/result_figures/results.csv'
NEW_CSV = 'Swin-Unet/model_out/Synapse/result_figures/results.csv'

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

old = load(OLD_CSV)
new = load(NEW_CSV)
organs = ['Aorta','Gallbladder','Kidney(L)','Kidney(R)','Liver','Pancreas','Spleen','Stomach']

print(f"{'Organ':<14} {'Old DSC':>9} {'New DSC':>9} {'Diff':>9} | {'Old HD95':>9} {'New HD95':>9} {'Diff':>9}")
print('-' * 100)

for o in organs:
    od = np.mean([float(r[o + '_Dice']) for r in old])
    nd = np.mean([float(r[o + '_Dice']) for r in new])
    oh = np.mean([float(r[o + '_HD95']) for r in old])
    nh = np.mean([float(r[o + '_HD95']) for r in new])
    ds = '↑' if nd > od else '↓'
    hs = '↑' if nh > oh else '↓'
    print(f"{o:<14} {od:9.4f} {nd:9.4f} {ds}{abs(nd-od):8.4f} | {oh:9.2f} {nh:9.2f} {hs}{abs(nh-oh):8.2f}")

om = np.mean([float(r['Mean_Dice']) for r in old])
nm = np.mean([float(r['Mean_Dice']) for r in new])
oh = np.mean([float(r['Mean_HD95']) for r in old])
nh = np.mean([float(r['Mean_HD95']) for r in new])

print('-' * 100)
ds = '↑' if nm > om else '↓'
hs = '↑' if nh > oh else '↓'
print(f"{'Mean':<14} {om:9.4f} {nm:9.4f} {ds}{abs(nm-om):8.4f} | {oh:9.2f} {nh:9.2f} {hs}{abs(nh-oh):8.2f}")
