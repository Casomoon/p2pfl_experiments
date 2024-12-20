import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
from p2pfl.management.logger import logger
from threading import Lock
import torch
import gc
plot_lock = Lock()

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
    colormap = sns.color_palette("Blues", as_cmap=True)
    # Lock plotting and file saving to avoid thread conflicts resulting in scuffed plots
    with plot_lock: 
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1)
        sns.heatmap(cm_percentage, ax=ax, annot=annotations, cmap=colormap, fmt='', annot_kws={"fontsize":10}, vmin=0.0,vmax=1.0)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion matrix Node {cid} Round {current_round}", fontsize=14)
        save_cm_path = save_dir/f"CM_{cid}_{current_round}.png"
        plt.savefig(fname=save_cm_path)
        plt.close(fig=fig)


class DFLAnalyzer(): 
    def __init__(self, run_dir: Path): 
        self.run_dir = run_dir
        self.client_dirs = [dir for dir in self.run_dir.iterdir() if dir.is_dir() and dir.name!="plots"]
        self.client_dfs = self.load_client_dfs(self.client_dirs)
    def load_client_dfs(self, client_dirs: list[Path]): 
        client_dfs = {}
        for cl_dir in client_dirs: 
            print(f"CL_DIR: {cl_dir}")
            client_idx = int(cl_dir.stem.split(sep="_")[1])
            client_csv_path  = cl_dir/f"BERT_{client_idx}"/"version_0"/"metrics.csv"
            print(client_csv_path)
            assert client_csv_path.exists()
            client_df = pd.read_csv(client_csv_path)
            client_dfs.update({client_idx: {"df_base" : client_df}})
        return client_dfs
    def sep_train_val_test(self, ):
        sets = ["train", "val", "test"]
        for cl_idx, data_dict in self.client_dfs.items():
            df = data_dict["df_base"]
            # seperate each client df into train, val, test sets 
            for set_name in sets:
                set_df: pd.DataFrame = df[[col for col in df.columns if set_name in col]]
                set_df.dropna(inplace=True)
                set_df.reset_index(inplace=True)
                set_df.drop("index",axis=1, inplace=True)
                set_df.columns = [col.replace(f"{set_name}_", "") for col in set_df.columns]
            data_dict[set_name] = set_df
    
    def plot_metrics(self):
        """Plots test_f1, test_acc, and test_loss for all clients."""
        metrics_to_plot = ["f1", "acc", "loss"]
        metric_names = {"f1": "F1 Score", "acc": "Accuracy", "loss": "Loss"}
        plot_analyzation = self.run_dir/"plots"
        if not plot_analyzation.exists():
            plot_analyzation.mkdir()
        for metric in metrics_to_plot:
            fig,ax = plt.subplots(figsize=(10, 6))
            max_epochs = 0
            sorted_clients = sorted(self.client_dfs.items(), key=lambda x: x[0])
            for client_idx, data_dict in sorted_clients:
                if 'test' in data_dict:
                    test_df = data_dict['test']
                    if metric in test_df.columns:
                        exclude_zeroth_epoch = test_df.iloc[1:]
                        max_epochs=max(max_epochs, len(exclude_zeroth_epoch))
                        ax.plot(exclude_zeroth_epoch[metric], label=f'Client {client_idx}')
            
            ax.set_title(f"Test {metric_names[metric]} for All Clients")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_names[metric])
            ax.set_xticks(range(1, max_epochs + 1))
            legend = ax.legend(
            loc="upper center", 
            bbox_to_anchor=(0.5, -0.1), 
            ncol=5, 
            frameon=True
            )
            plt.grid(True)
            plt.savefig(plot_analyzation/f"{metric_names[metric]}.png", bbox_inches='tight')
            plt.close(fig=fig)

    def plot_model_divergence(self, start_round:int=1, end_round:int=3): 
        plot_analyzation = self.run_dir/"plots"
        model_paths = [cl_dir/f"BERT_Lightning_{int(cl_dir.stem.split(sep='_')[1])}" for cl_dir in self.client_dirs]
        avg_divs = []
        for round in range(start_round, end_round+1):
            paths_this_round = [path.with_name(f"{path.name}_{round}.pth") for path in model_paths]
            for path in paths_this_round: 
                print(f"Assert Path {path} exists.")
                assert path.exists()
            ru_su = self.ru_su_average_model(paths_this_round)
            avg_div_this_round = self.get_avg_div(ru_su, paths_this_round).to("cpu")
            avg_divs.append(avg_div_this_round)
        # actually plot it now 
        print(f"avg_divs : {avg_divs}")
        plt.figure(figsize=(8, 5))
        plt.plot([1,2,3], avg_divs, marker='o', linestyle='-', linewidth=2)
        plt.title('Durchschnittliche Modeldivergenz zwischen Runden', fontsize=14)
        plt.xlabel('Runde', fontsize=12)
        plt.ylabel('Divergenz (L2 Norm)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks([1,2,3], fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(fname=plot_analyzation/"model_div.png")
    def ru_su_average_model(self, paths:list[Path]):
        ru_su = None
        num_models = len(paths)
        for model_path in paths:
            print(f"loading model {model_path}")
            cur_model = torch.load(model_path)
            if ru_su is None: 
                ru_su = {key: torch.zeros_like(value) for key, value in cur_model.items()}
            for key in ru_su.keys():
                ru_su[key]+=cur_model[key]
            print(f"removing from RAM")
            del cur_model
            gc.collect()
        averaged_model = {key: value/num_models for key, value in ru_su.items()}
        return averaged_model
    
    def get_avg_div(self, avg_model: dict, paths: list[Path]):
        divergences = []
        #calculate l2 norm divergences
        for path in paths:
            model = torch.load(path)
            divergence = torch.sqrt(sum(torch.sum((model[key] - avg_model[key]) ** 2) for key in avg_model.keys()))
            divergences.append(divergence)
            del model
            gc.collect()
        average_divergence = sum(divergences)/len(divergences)
        return average_divergence