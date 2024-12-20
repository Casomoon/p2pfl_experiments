import pandas as pd 
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
def main(): 
    root = Path(__file__).resolve().parents[2]
    run_res_path = root/"run_results/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()
    dir = args.dir 
    eval_dir = run_res_path/f"{dir}"
    csv_loc = eval_dir/"SingleBertTraining"/"version_0"/"metrics.csv"
    plot_single_bert(csv_loc, eval_dir)


def plot_single_bert(csv_loc: Path, eval_dir: Path):
    assert csv_loc.exists()
    metrics = pd.read_csv(csv_loc)
    # filter test 
    test_metrics = metrics[[col for col in metrics.columns if col.startswith("test_") or col == "epoch"]]
    test_metrics.dropna(inplace=True)
    
    test_metrics['epoch'] = test_metrics['epoch'] + 1
    # Plot the test metrics
    plt.figure(figsize=(10, 6))
    for metric in test_metrics.columns:
        if metric != 'epoch':
            plt.plot(test_metrics['epoch'], test_metrics[metric], label=metric)

    plt.title('Test Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.xticks(ticks=test_metrics['epoch'].unique(), labels=test_metrics['epoch'].unique().astype(int))
    plt.legend()
    plt.grid()
    plt.savefig(eval_dir/"test_metrics.png")
    plt.close()




    