import argparse
from optimal_config_stats_analysis import run_optimal_model_analysis
from config import (
    seed_list,
    disney_key_param_list,
    books_key_param_list,
    reddit_key_param_list,
    enron_key_param_list,
    weibo_key_param_list,
)


param_grid_dict = {
    "disney": disney_key_param_list,
    "books": books_key_param_list,
    "reddit": reddit_key_param_list,
    "enron": enron_key_param_list,
    "weibo": weibo_key_param_list,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run optimal model analysis for specified dataset"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="The name of the dataset to analyze"
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name not in param_grid_dict:
        print(
            f"Error: Dataset '{dataset_name}' not found in param_grid_dict. Skipping this dataset"
        )
        return

    key_param_list = param_grid_dict[dataset_name]
    print(f"Running analysis for dataset: {dataset_name}")
    try:
        run_optimal_model_analysis(
            {dataset_name: key_param_list}, seed_list, report_multi=True
        )
        print(f"Analysis for dataset '{dataset_name}' completed successfully.")
    except Exception as e:
        print(f"Error occurred while analyzing dataset '{dataset_name}': {e}")


if __name__ == "__main__":
    main()
