from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, current_round: int): 
    cm = confusion_matrix(y_true=labels, y_pred=preds)
    fig = plt.figure(figsize=(6,6))
    colormap = sns.color_palette("flare", as_cmap=True)
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, ax=ax, annot=True, cmap=colormap)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
#class ResultVisualizer:
    

if __name__ == "__main__": pass