"""
Visualization Utilities for Deep Feedforward Neural Network (FNN)
Course: Pattern Recognition (TAE 1)
Student: Amogh Samarth | USN: CM23034

Minimalist, publication-grade figures using a consistent modern accent palette.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent Modern Minimalist Styling
ACCENT_COLOR = '#2563EB'       # Primary Accent Blue
ACCENT_SECONDARY = '#64748B'   # Neutral Slate
ACCENT_LIGHT = '#DBEAFE'       # Light Tint
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#0F172A'
MUTED_TEXT = '#64748B'
GRID_COLOR = '#F1F5F9'

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#E2E8F0'
plt.rcParams['axes.linewidth'] = 0.8


def plot_training_curves(history):
    """
    Plots clean, minimalist training vs validation accuracy and loss curves.
    """
    epochs_range = range(1, len(history.history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)

    # 1. Accuracy Curve
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

    # 2. Loss Curve
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
    """
    Renders a minimalist Confusion Matrix heatmap with subtle colors.
    """
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
    """
    Creates a clean, minimal 2x2 grid showing feature distributions across species.
    """
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


def draw_fnn_architecture():
    """
    Generates a clean, modern, spacious neural network topology diagram:
    4 Inputs -> 16 Dense (ReLU) -> 16 Dense (ReLU) -> 8 Dense (ReLU) -> 3 Output (Softmax).
    """
    layer_sizes = [4, 16, 16, 8, 3]
    layer_titles = ['INPUT', 'DENSE', 'DENSE', 'DENSE', 'OUTPUT']
    layer_subtitles = ['4 Features', '16 Neurons\nReLU', '16 Neurons\nReLU', '8 Neurons\nReLU', '3 Classes\nSoftmax']

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    x_coords = [0.10, 0.30, 0.50, 0.70, 0.90]
    
    # Calculate vertical spacing
    node_positions = []
    for l_idx, count in enumerate(layer_sizes):
        # We draw clean evenly spaced nodes
        display_nodes = min(count, 12)
        spacing = 0.74 / (display_nodes + 1)
        y_positions = [(i + 1) * spacing + 0.08 for i in range(display_nodes)]
        node_positions.append(y_positions)

    # Draw connection lines (subtle, clean)
    for l in range(len(layer_sizes) - 1):
        x1, x2 = x_coords[l], x_coords[l + 1]
        for y1 in node_positions[l]:
            for y2 in node_positions[l + 1]:
                ax.plot([x1, x2], [y1, y2], color='#CBD5E1', alpha=0.18, linewidth=0.6, zorder=1)

    # Draw nodes
    for l, (x, y_list) in enumerate(zip(x_coords, node_positions)):
        is_output = (l == len(layer_sizes) - 1)
        is_input = (l == 0)
        node_color = ACCENT_COLOR if is_output else ('#3B82F6' if is_input else '#60A5FA')

        for y in y_list:
            circle = plt.Circle((x, y), 0.016, color=node_color, ec='#FFFFFF', lw=1.2, zorder=3)
            ax.add_patch(circle)

        # Header Box
        ax.text(x, 0.93, layer_titles[l], ha='center', va='bottom',
                fontsize=9.5, fontweight='700', color=TEXT_COLOR)
        ax.text(x, 0.85, layer_subtitles[l], ha='center', va='top',
                fontsize=8, color=MUTED_TEXT, linespacing=1.2)

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.04, 1.02)
    plt.tight_layout()
    return fig


def plot_probability_chart(probabilities, class_names):
    """
    Clean horizontal bar chart showing Softmax class probabilities.
    """
    fig, ax = plt.subplots(figsize=(6, 2.0), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    classes = [c.replace('Iris ', '') for c in class_names]
    probs = [probabilities[c] * 100 for c in class_names]
    
    max_idx = int(np.argmax(probs))
    colors = [ACCENT_COLOR if i == max_idx else '#94A3B8' for i in range(len(classes))]

    bars = ax.barh(classes, probs, color=colors, height=0.45, edgecolor='none')
    
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        text_x = width + 2 if width < 85 else width - 10
        text_color = TEXT_COLOR if width < 85 else '#FFFFFF'
        ax.text(text_x, bar.get_y() + bar.get_height() / 2, f"{prob:.1f}%",
                ha='left' if width < 85 else 'right', va='center',
                fontsize=9.5, fontweight='600', color=text_color)

    ax.set_xlim(0, 105)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=10, colors=TEXT_COLOR)
    ax.tick_params(axis='x', labelsize=8, colors=MUTED_TEXT)
    ax.grid(True, axis='x', linestyle='-', color=GRID_COLOR, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')
    
    plt.tight_layout()
    return fig
