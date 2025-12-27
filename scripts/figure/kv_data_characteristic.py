#! /usr/bin/env python
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compute import QuantMethod, RepackMethod
from utils.config import PackKVCacheConfig
from evaluation.evaluation import cr_evaluation
from utils.util import get_logger, block_other_logger
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64
ctx_len = 1024

font_path = 'Founders_Grotesk/FoundersGrotesk-Regular.otf'
founders_reg_prop = FontProperties(fname=font_path, size=24)

config = PackKVCacheConfig(
                        enable_quant=True,
                        model_name="meta-llama/Llama-2-13b-hf",
                        quant_method=QuantMethod.PackKV,
                        repack_method=RepackMethod.GREEDY,
                        high_precision_zero_point=False,
                        block_size=BLOCK_SIZE,
                        buffer_size=BUFFER_SIZE,
                        pack_size=16,
                        k_quant_scale_rel=0.1,
                        v_quant_scale_rel=0.2
            )

logger = get_logger(__file__)
block_other_logger(logger)

before_and_after = []
cr_result = cr_evaluation(
        config=config,
        ctx_len=ctx_len,
        enable_save=False,
        logger=logger,
        before_and_after_repacking=before_and_after
    )

sample = before_and_after[19]
pack_idx = 10
sample_width = 100
hidden_dim_ = sample[0].shape[2]
k_sample_ = sample[0][pack_idx, : ,:sample_width]

k_quant_tensor = sample[0][:,:, :hidden_dim_//2]
v_quant_tensor = sample[0][:,:, hidden_dim_//2:]

# Create output directory if it doesn't exist
os.makedirs('kv_data_characteristic', exist_ok=True)

# 1. Draw heatmap for k_sample_ (following repacking visualization method)
# Convert tensor to numpy if it's not already
if hasattr(k_sample_, 'cpu'):
    k_sample_data = k_sample_.cpu().numpy()
else:
    k_sample_data = np.array(k_sample_)

# Create heatmap for k_sample_
fig, ax = plt.subplots(1, 1, figsize=(4, 4))

# Create heatmap without axes, labels, or any other elements
ax.imshow(k_sample_data, cmap='coolwarm', aspect='equal', interpolation='nearest')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')

# Remove all spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Save as separate PDF and SVG
plt.savefig('kv_data_characteristic/k_sample_heatmap.svg', format='svg', bbox_inches='tight', pad_inches=0)
plt.close()

print("K sample heatmap saved as kv_data_characteristic/k_sample_heatmap.pdf and k_sample_heatmap.svg")

# 2. Draw histograms for k and v quant tensors
# Convert tensors to numpy
if hasattr(k_quant_tensor, 'cpu'):
    k_data = k_quant_tensor.cpu().numpy()
else:
    k_data = np.array(k_quant_tensor)

if hasattr(v_quant_tensor, 'cpu'):
    v_data = v_quant_tensor.cpu().numpy()
else:
    v_data = np.array(v_quant_tensor)

# Create combined histogram for both k and v tensors
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
combined_min = min(int(k_data.min()), int(v_data.min()))
combined_max = max(int(k_data.max()), int(v_data.max()))
bins = np.arange(combined_min, combined_max + 2) - 0.5

ax.hist(k_data.flatten(), bins=bins, alpha=0.7, color='#356ba0', edgecolor='none', rwidth=1.0, label='K Quant Tensor')
ax.hist(v_data.flatten(), bins=bins, alpha=0.7, color='#ff6b6b', edgecolor='none', rwidth=1.0, label='V Quant Tensor')

ax.legend(prop=founders_reg_prop, fontsize=96)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
ax.tick_params(axis='both', which='major', labelsize=24)

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((6, 6))
ax.yaxis.set_major_formatter(formatter)

ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=False))

plt.tight_layout()

ax.yaxis.get_offset_text().set_fontsize(24)

plt.savefig('kv_data_characteristic/kv_histogram.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.close()

print("K and V histogram saved as kv_data_characteristic/kv_histogram.pdf")

print("\nAll visualizations completed and saved to kv_data_characteristic/ directory")
