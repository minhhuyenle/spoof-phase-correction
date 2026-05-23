#!/usr/bin/env python3
"""
Script to apply intensity mask to phase data and save visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This script assumes the notebook kernel is running and variables are available
# It will be imported by the notebook cell

def apply_and_plot_intensity_mask(phase1_corrected, intensity1, phase2_corrected, intensity2):
    """Apply intensity mask to phase for both polarization states"""
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Intensity Mask Applied to Phase", fontsize=16)
    
    results = {}
    
    for idx, (phase_data, intensity_data, label) in enumerate([
        (phase1_corrected, intensity1, "Pol1"),
        (phase2_corrected, intensity2, "Pol2")
    ]):
        # Normalize intensity to 0-1 range
        intensity_norm = (intensity_data - intensity_data.min()) / (intensity_data.max() - intensity_data.min())
        
        # Create mask: keep only regions where intensity is above 40% of max
        threshold = 0.4
        intensity_mask = intensity_norm > threshold
        
        # Apply mask to phase - set masked-out regions to NaN
        phase_masked = np.copy(phase_data)
        phase_masked[~intensity_mask] = np.nan
        
        # Store results
        results[label] = {
            'phase_masked': phase_masked,
            'intensity_mask': intensity_mask,
            'intensity_norm': intensity_norm
        }
        
        # Plot original intensity
        axs[idx, 0].imshow(intensity_norm, cmap='gray')
        axs[idx, 0].set_title(f"{label} - Normalized Intensity", fontsize=12)
        axs[idx, 0].axis('off')
        
        # Plot intensity mask
        axs[idx, 1].imshow(intensity_mask, cmap='gray')
        axs[idx, 1].set_title(f"{label} - Intensity Mask (threshold={threshold})", fontsize=12)
        axs[idx, 1].axis('off')
        
        # Plot masked phase
        cmap_phase = sns.color_palette("icefire_r", as_cmap=True)
        valid_phase = phase_masked[intensity_mask]
        vmin_phase = np.nanpercentile(valid_phase, 2)
        vmax_phase = np.nanpercentile(valid_phase, 98)
        
        im_masked = axs[idx, 2].imshow(phase_masked, cmap=cmap_phase, vmin=vmin_phase, vmax=vmax_phase)
        axs[idx, 2].set_title(f"{label} - Phase (Intensity Masked)", fontsize=12)
        axs[idx, 2].axis('off')
        plt.colorbar(im_masked, ax=axs[idx, 2], label="Phase (rad)", fraction=0.046, pad=0.04)
        
        print(f"Mask statistics for {label}:")
        print(f"  {intensity_mask.sum()} pixels above threshold ({100*intensity_mask.sum()/intensity_mask.size:.1f}%)")
        print(f"  Phase range (masked): {np.nanmin(valid_phase):.4f} to {np.nanmax(valid_phase):.4f} rad")
    
    plt.tight_layout()
    plt.savefig('/Users/huyenle/Desktop/20260412_Beads/intensity_mask_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == '__main__':
    print("This script is meant to be imported, not run directly.")
    print("Use: from apply_intensity_mask import apply_and_plot_intensity_mask")
