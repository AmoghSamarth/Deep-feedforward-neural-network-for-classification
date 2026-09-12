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

def plot_confusion_matrix_figure(cm, class_names):
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.replace('Iris ', '') for c in class_names],
        yticklabels=[c.replace('Iris ', '') for c in class_names],
        cbar=False,
        ax=ax,
        linewidths=1,
        linecolor='#FFFFFF',
        square=True,
        annot_kws={"size": 12, "fontweight": "600", "color": TEXT_COLOR}
    )

    ax.set_title('Confusion Matrix', fontsize=12, fontweight='600', pad=12, color=TEXT_COLOR)
    ax.set_xlabel('Predicted Class', fontsize=10, fontweight='500', labelpad=8, color=MUTED_TEXT)
    ax.set_ylabel('Actual Class', fontsize=10, fontweight='500', labelpad=8, color=MUTED_TEXT)
    ax.tick_params(axis='both', labelsize=10, colors=MUTED_TEXT)
    
    plt.tight_layout()
    return fig

def plot_feature_distributions(df):
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)

    features = [
        ('sepal_length', 'Sepal Length'),
        ('sepal_width', 'Sepal Width'),
        ('petal_length', 'Petal Length'),
        ('petal_width', 'Petal Width')
    ]
    palette = [ACCENT_COLOR, '#38BDF8', '#94A3B8']

    for idx, (col, title) in enumerate(features):
        ax = axes[idx]
        ax.set_facecolor(BG_COLOR)
        sns.boxplot(
            x='species',
            y=col,
            data=df,
            palette=palette,
            ax=ax,
            width=0.45,
            boxprops=dict(alpha=0.9, edgecolor=TEXT_COLOR, linewidth=0.8),
            medianprops=dict(color=TEXT_COLOR, linewidth=1.2),
            whiskerprops=dict(color=MUTED_TEXT, linewidth=0.8),
            capprops=dict(color=MUTED_TEXT, linewidth=0.8)
        )
        ax.set_title(title, fontsize=10, fontweight='600', color=TEXT_COLOR, pad=8)
        ax.set_xlabel('')
        ax.set_ylabel('cm' if idx == 0 else '', fontsize=9, color=MUTED_TEXT)
        ax.tick_params(axis='x', labelsize=8.5, colors=MUTED_TEXT)
        ax.tick_params(axis='y', labelsize=8.5, colors=MUTED_TEXT)
        ax.grid(True, linestyle='-', color=GRID_COLOR, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color('#E2E8F0')

    plt.tight_layout()
    return fig
