import os
import csv
from datetime import datetime
import torch
import random
import numpy as np
import gc

from config import parse_arguments
from train import train_real_datasets


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_new_seed(seed_list=None):
    """Generate a new random seed for retry attempts."""
    while True:
        seed = random.randint(0, 10000)
        # Ensure the new seed is not in the original seed list to avoid repetition
        if seed not in seed_list:
            return seed


def run_optimal_model_analysis(param_grid_dict, seed_list, report_multi=False):
    now = datetime.now()
    task_start_timestamp = now.strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    max_retries = 5

    for dataset, hyperparam_config_list in param_grid_dict.items():
        for hyperparam_config_index, key_hyperparams in enumerate(
            hyperparam_config_list
        ):
            args = parse_arguments()
            args.dataset = dataset

            for param_name, param_value in key_hyperparams.items():
                setattr(args, param_name, param_value)

            if not args.single:
                args.gamma = args.gamma_list
            else:
                args.gamma = [args.gamma]

            all_results = []
            error_logs = []
            repeat_roc_auc = []
            repeat_pr_auc = []

            for curr_seed in seed_list:
                success = False
                retries = 0
                use_main_seed = True
                while not success and retries < max_retries:
                    try:
                        if use_main_seed:
                            set_seed(curr_seed)
                            print(
                                f"Running on dataset {dataset}, hyperparam set {hyperparam_config_index}, using main seed {curr_seed} from seed list provided"
                            )
                            use_main_seed = False
                        else:
                            # Generate a new seed for retry attempts
                            curr_seed = generate_new_seed(seed_list)
                            set_seed(curr_seed)
                            print(
                                f"Retrying on for dataset {dataset}, hyperparam set {hyperparam_config_index}, attempt {retries + 1}, retry seed {curr_seed}"
                            )

                        (
                            loss,
                            loss_per_node,
                            curr_best_auc,
                            curr_best_pr_auc,
                            timestamp,
                        ) = train_real_datasets(
                            args,
                            device=device,
                            epoch_num=args.epoch_num,
                            lr=args.lr,
                            lambda_loss1=args.lambda_loss1,
                            lambda_loss2=args.lambda_loss2,
                            lambda_loss3=args.lambda_loss3,
                            sample_size=args.sample_size,
                            loss_step=args.loss_step,
                            hidden_dim=args.hidden_dimension,
                            real_loss=args.real_loss,
                            calculate_contextual=args.calculate_contextual,
                            calculate_structural=args.calculate_structural,
                            calculate_joint=args.calculate_joint,
                            case="metrics_mean_std_stats",
                            task_start_timestamp=f"{task_start_timestamp}/hyperparam_{hyperparam_config_index}",
                        )

                        all_results.append(
                            (curr_seed, curr_best_auc, curr_best_pr_auc, timestamp)
                        )
                        repeat_roc_auc.append(curr_best_auc)
                        repeat_pr_auc.append(curr_best_pr_auc)

                        # Run sucessfully, exit and move to the next seed
                        success = True
                        del loss, loss_per_node
                        gc.collect()
                        torch.cuda.empty_cache()

                    except Exception as e:
                        error_message = f"Error  dataset={dataset}, hyperparam set {hyperparam_config_index}, attempt {retries + 1}, seed {curr_seed}. Error: {str(e)}"
                        print(error_message)
                        error_logs.append(error_message)
                        retries += 1
                        if retries == max_retries:
                            error_message = f"Max retries reached for dataset={dataset}, hyperparam set {hyperparam_config_index}, currently seed is {curr_seed}. Repeat trying unitl model trained successfully."
                            print(error_message)
                            error_logs.append(error_message)
                            retries = 0

            # Create directory to save results and error logs
            end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            encoder = args.encoder
            suffix = args.suffix
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../results"
            )
            dataset_part = f"{dataset}_{suffix}" if suffix is not None else dataset
            result_dir = os.path.join(
                base_dir,
                dataset_part,
                encoder,
                "analysis",
                "metrics_mean_std_stats",
                f"Running_{task_start_timestamp}",
                f"hyperparam_{hyperparam_config_index}",
                "summary",
            )
            os.makedirs(result_dir, exist_ok=True)

            # Create error log file and save error logs
            error_log_file = os.path.join(
                result_dir,
                f"error_logs_{dataset}_{encoder}_hyperparam{hyperparam_config_index}_{end_timestamp}.txt",
            )
            with open(error_log_file, "w") as f:
                f.write("Error Log:\n")
                for log in error_logs:
                    f.write(f"{log}\n")
            print(f"Error logs saved to: {error_log_file}")

            # Create file to save all seeds and their corresponding ROC-AUC and PR-AUC values
            all_result_file_path = os.path.join(
                result_dir,
                f"compare_results_{dataset}_{encoder}_hyperparam{hyperparam_config_index}_{end_timestamp}.csv",
            )
            with open(all_result_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(key_hyperparams)
                writer.writerow(["Seed", "ROC-AUC", "PR-AUC", "Timestamp"])
                for seed_val, roc_auc, pr_auc, timestamp in all_results:
                    writer.writerow([seed_val, roc_auc, pr_auc, timestamp])
            print(
                f"All seeds and their ROC-AUC and PR-AUC values have been saved to: {all_result_file_path}"
            )

            # Save top 10 ROC-AUC and PR-AUC results with their seeds.
            top_10_file_path = os.path.join(
                result_dir,
                f"top_10_params_{dataset}_{encoder}_hyperparam{hyperparam_config_index}_{end_timestamp}.txt",
            )

            if len(seed_list) > 10:
                # We remove top 1 and bottom results to avoid outliers.
                sorted_by_roc_auc = sorted(
                    all_results, key=lambda x: x[1], reverse=True
                )[1:11]
                sorted_by_pr_auc = sorted(
                    all_results, key=lambda x: x[2], reverse=True
                )[1:11]
            else:
                print(
                    f"Seed list contains {len(seed_list)} seeds, which is not more than 10. Saving all results all removing top and bottom seeds."
                )
                sorted_by_roc_auc = sorted(
                    all_results, key=lambda x: x[1], reverse=True
                )
                sorted_by_pr_auc = sorted(all_results, key=lambda x: x[2], reverse=True)

            # Calculate mean and std of ROC-AUC and PR-AUC for the selected results
            roc_auc_values = np.array(
                [row[1] for row in sorted_by_roc_auc], dtype=float
            )
            pr_auc_values = np.array([row[2] for row in sorted_by_pr_auc], dtype=float)
            mean_best_roc_auc = sum(roc_auc_values) / len(roc_auc_values)
            mean_best_pr_auc = sum(pr_auc_values) / len(pr_auc_values)
            std_best_roc_auc = np.std(roc_auc_values, ddof=1)
            std_best_pr_auc = np.std(pr_auc_values, ddof=1)

            with open(top_10_file_path, "w") as f:
                f.write(f"Key Hyperparams: {key_hyperparams}\n\n")
                f.write("Top 10 Parameter Statistics:\n")
                f.write(
                    f"Mean ROC-AUC (Std): {mean_best_roc_auc:.4f} ({std_best_roc_auc:.4f})\n"
                )
                f.write(
                    f"Mean PR-AUC (Std): {mean_best_pr_auc:.4f} ({std_best_pr_auc:.4f})\n"
                )
                f.write("Top 10 Parameter Configurations:\n")
                f.write("------Based on ROC-AUC------\n")
                for i, (seed_val, roc_auc, pr_auc, timestamp) in enumerate(
                    sorted_by_roc_auc
                ):
                    f.write(f"Rank {i+1}:\n")
                    f.write(f"Seed: {seed_val}\n")
                    f.write(f"ROC-AUC: {roc_auc}\n")
                    f.write(f"Correspond PR-AUC: {pr_auc}\n")
                    f.write(f"Timestamp: {timestamp}\n\n")
                f.write("------Based on PR-AUC------\n")
                for i, (seed_val, roc_auc, pr_auc, timestamp) in enumerate(
                    sorted_by_pr_auc
                ):
                    f.write(f"Rank {i+1}:\n")
                    f.write(f"Seed: {seed_val}\n")
                    f.write(f"PR-AUC: {pr_auc}\n")
                    f.write(f"Correspond ROC-AUC: {roc_auc}\n")
                    f.write(f"Timestamp: {timestamp}\n\n")

            print(f"Top 10 configurations saved to: {top_10_file_path}")
