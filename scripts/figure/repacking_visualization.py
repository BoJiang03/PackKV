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

BLOCK_SIZE = 64
BUFFER_SIZE = 128 + 64
ctx_len = 1024

font_path = 'Founders_Grotesk/FoundersGrotesk-Regular.otf'
founders_reg_prop = FontProperties(fname=font_path)

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
sample_width = 64
hidden_dim_ = sample[0].shape[2]
v_start_ = hidden_dim_ // 2
k_before_ = sample[0][pack_idx, : ,:sample_width]
k_after_ = sample[1][pack_idx, : ,:sample_width]
v_before_ = sample[0][pack_idx, :, v_start_:v_start_ + sample_width]
v_after_ = sample[1][pack_idx, :, v_start_:v_start_ + sample_width]

# Visualization
tensors = [k_before_, k_after_, v_before_, v_after_]
names = ['k_before', 'k_after', 'v_before', 'v_after']

# Create output directory if it doesn't exist
os.makedirs('before_after_repacking', exist_ok=True)

for tensor, name in zip(tensors, names):
    # Convert tensor to numpy if it's not already
    if hasattr(tensor, 'cpu'):
        data = tensor.cpu().numpy()
    else:
        data = np.array(tensor)
    
    # Create a single figure for each tensor
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # Create heatmap without axes, labels, or any other elements
    ax.imshow(data, cmap='coolwarm', aspect='equal', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('')
    
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save as separate PDF and SVG
    plt.savefig(f'before_after_repacking/{name}.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'before_after_repacking/{name}.svg', format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Visualization saved as before_after_reapacking/{name}.pdf and {name}.svg")

# Create combined visualization with better spacing
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Convert tensors to numpy
k_before_data = k_before_.cpu().numpy() if hasattr(k_before_, 'cpu') else np.array(k_before_)
k_after_data = k_after_.cpu().numpy() if hasattr(k_after_, 'cpu') else np.array(k_after_)
v_before_data = v_before_.cpu().numpy() if hasattr(v_before_, 'cpu') else np.array(v_before_)
v_after_data = v_after_.cpu().numpy() if hasattr(v_after_, 'cpu') else np.array(v_after_)

# Plot k_before (top-left)
axes[0, 0].imshow(k_before_data, cmap='coolwarm', aspect='equal', interpolation='nearest')
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])
axes[0, 0].set_title('K Before', fontsize=14, pad=15, fontproperties=founders_reg_prop)
for spine in axes[0, 0].spines.values():
    spine.set_visible(False)

# Plot k_after (bottom-left)
axes[1, 0].imshow(k_after_data, cmap='coolwarm', aspect='equal', interpolation='nearest')
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])
axes[1, 0].set_title('K After', fontsize=14, pad=15, fontproperties=founders_reg_prop)
for spine in axes[1, 0].spines.values():
    spine.set_visible(False)

# Plot v_before (top-right)
axes[0, 1].imshow(v_before_data, cmap='coolwarm', aspect='equal', interpolation='nearest')
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
axes[0, 1].set_title('V Before', fontsize=14, pad=15, fontproperties=founders_reg_prop)
for spine in axes[0, 1].spines.values():
    spine.set_visible(False)

# Plot v_after (bottom-right)
axes[1, 1].imshow(v_after_data, cmap='coolwarm', aspect='equal', interpolation='nearest')
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
axes[1, 1].set_title('V After', fontsize=14, pad=15, fontproperties=founders_reg_prop)
for spine in axes[1, 1].spines.values():
    spine.set_visible(False)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Add arrows and text for "greedy repacking" with better positioning
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import ConnectionPatch

# Arrow and text between k_before and k_after
arrow_k = ConnectionPatch((0.5, 0), (0.5, 1), "axes fraction", "axes fraction",
                         axesA=axes[1, 0], axesB=axes[0, 0], 
                         arrowstyle="->", shrinkA=5, shrinkB=5, 
                         mutation_scale=20, fc="black", ec="black", lw=2)
fig.add_artist(arrow_k)
fig.text(0.25, 0.5, 'Greedy\nRepacking', ha='center', va='center', 
         fontsize=12, fontweight='bold', fontproperties=founders_reg_prop,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8))

# Arrow and text between v_before and v_after
arrow_v = ConnectionPatch((0.5, 0), (0.5, 1), "axes fraction", "axes fraction",
                         axesA=axes[1, 1], axesB=axes[0, 1], 
                         arrowstyle="->", shrinkA=5, shrinkB=5, 
                         mutation_scale=20, fc="black", ec="black", lw=2)
fig.add_artist(arrow_v)
fig.text(0.75, 0.5, 'Greedy\nRepacking', ha='center', va='center', 
         fontsize=12, fontweight='bold', fontproperties=founders_reg_prop,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8))

plt.savefig('before_after_repacking/combined_visualization.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.savefig('before_after_repacking/combined_visualization.svg', format='svg', bbox_inches='tight', pad_inches=0.2)
plt.close()

print("Combined visualization saved as before_after_repacking/combined_visualization.pdf and combined_visualization.svg")