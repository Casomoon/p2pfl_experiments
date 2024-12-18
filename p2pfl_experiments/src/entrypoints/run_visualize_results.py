from ..stats.result_visualization import DFLAnalyzer
from pathlib import Path
import argparse
def main(): 
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()
    print(root)
    results_dir: Path = root/"run_results"/args.dir 
    assert results_dir.exists()
    analyzer = DFLAnalyzer(results_dir)
    analyzer.sep_train_val_test()
    analyzer.plot_metrics()
    analyzer.plot_model_divergence()