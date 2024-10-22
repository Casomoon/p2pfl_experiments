import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
from p2pfl.management.logger import logger
from threading import Lock

plot_lock = Lock()

#use the "Agg" matplotlib backend to avoid files only being half rendered in the png
def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, save_dir: Path, cid: int, current_round: int): 
    # base confusion_matrix
    cm = confusion_matrix(y_true=labels, y_pred=preds)
    cm_percentage = cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
    logger.info("plotter", str(cm_percentage))
    # annotation np array to combine full numbers together with percentages
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): 
            annotations[i,j] = f"{cm[i,j]}\n({cm_percentage[i,j]:.2%})"
    colormap = sns.color_palette("RdYlGn", as_cmap=True)
    # Lock plotting and file saving to avoid thread conflicts resulting in scuffed plots
    with plot_lock: 
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1)
        sns.heatmap(cm, ax=ax, annot=annotations, cmap=colormap, fmt='', annot_kws={"fontsize":10})
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion matrix Node {cid} Round {current_round}", fontsize=14)
        save_cm_path = save_dir/f"CM_{cid}_{current_round}.png"
        plt.savefig(fname=save_cm_path)
        plt.close(fig=fig)
#class ResultVisualizer:
    

if __name__ == "__main__": pass