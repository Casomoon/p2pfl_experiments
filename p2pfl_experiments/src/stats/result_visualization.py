from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, save_dir: Path, cid: int, current_round: int): 
    cm = confusion_matrix(y_true=labels, y_pred=preds)
    fig = plt.figure(figsize=(6,6))
    colormap = sns.color_palette("flare", as_cmap=True)
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, ax=ax, annot=True, cmap=colormap)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix Node {cid} Round {current_round}")
    save_cm_path = save_dir/f"CM_{cid}_{current_round}.png"
    plt.savefig(fname=save_cm_path)
    plt.close(fig=fig)
#class ResultVisualizer:
    

if __name__ == "__main__": pass