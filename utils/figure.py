import os
from typing import List
import matplotlib

# 非交互式后端，便于服务器/脚本环境保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def draw_loss_curve(steps: List[int], train_losses: List[float], out_path: str = 'figures/loss.png'):
    _ensure_dir(out_path)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, train_losses, label='Train Loss')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_accuracy_curve(train_steps: List[int], train_accs: List[float], val_steps: List[int], val_accs: List[float], out_path: str = 'figures/accuracy.png'):
    _ensure_dir(out_path)
    plt.figure(figsize=(7, 4))
    if train_steps and train_accs:
        plt.plot(train_steps, train_accs, label='Train Acc')
    if val_steps and val_accs:
        plt.plot(val_steps, val_accs, label='Val Acc')
    plt.xlabel('Global Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_f1_curve(train_steps: List[int], train_f1s: List[float], val_steps: List[int], val_f1s: List[float], out_path: str = 'figures/f1.png'):
    _ensure_dir(out_path)
    plt.figure(figsize=(7, 4))
    if train_steps and train_f1s:
        plt.plot(train_steps, train_f1s, label='Train Macro-F1')
    if val_steps and val_f1s:
        plt.plot(val_steps, val_f1s, label='Val Macro-F1')
    plt.xlabel('Global Step')
    plt.ylabel('Macro-F1')
    plt.title('Macro-F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


