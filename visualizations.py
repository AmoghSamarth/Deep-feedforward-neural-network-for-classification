import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ACCENT_COLOR = '#2563EB'
ACCENT_SECONDARY = '#64748B'
ACCENT_LIGHT = '#DBEAFE'
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#0F172A'
MUTED_TEXT = '#64748B'
GRID_COLOR = '#F1F5F9'

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#E2E8F0'
plt.rcParams['axes.linewidth'] = 0.8

def plot_training_curves(history):
    epochs_range = range(1, len(history.history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)

    ax1.set_facecolor(BG_COLOR)
    ax1.plot(epochs_range, [a * 100 for a in history.history['accuracy']], label='Train', color=ACCENT_COLOR, linewidth=2)
    ax1.plot(epochs_range, [a * 100 for a in history.history['val_accuracy']], label='Validation', color=ACCENT_SECONDARY, linewidth=2, linestyle='--')
    ax1.set_title('Accuracy (%)', fontsize=12, fontweight='600', color=TEXT_COLOR, pad=10)
    ax1.set_xlabel('Epoch', fontsize=10, color=MUTED_TEXT)
    ax1.set_ylabel('Accuracy (%)', fontsize=10, color=MUTED_TEXT)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle='-', color=GRID_COLOR, linewidth=1)
    ax1.legend(frameon=False, fontsize=9)
    for spine in ax1.spines.values():
        spine.set_color('#E2E8F0')

    ax2.set_facecolor(BG_COLOR)
    ax2.plot(epochs_range, history.history['loss'], label='Train', color=ACCENT_COLOR, linewidth=2)
    ax2.plot(epochs_range, history.history['val_loss'], label='Validation', color=ACCENT_SECONDARY, linewidth=2, linestyle='--')
    ax2.set_title('Loss (Cross-Entropy)', fontsize=12, fontweight='600', color=TEXT_COLOR, pad=10)
    ax2.set_xlabel('Epoch', fontsize=10, color=MUTED_TEXT)
    ax2.set_ylabel('Loss', fontsize=10, color=MUTED_TEXT)
    ax2.grid(True, linestyle='-', color=GRID_COLOR, linewidth=1)
    ax2.legend(frameon=False, fontsize=9)
    for spine in ax2.spines.values():
        spine.set_color('#E2E8F0')

    plt.tight_layout()
    return fig
