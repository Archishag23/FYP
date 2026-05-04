import os
import csv
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from pygod.metric import eval_roc_auc
from pygod.generator import gen_contextual_outlier, gen_structural_outlier

from model import GNNStructEncoder
from utils import (
    get_normalization,
    eval_pr_auc,
    count_ones_and_zeros,
    gen_joint_structural_outlier,
)
from preprocess.preprocess_data import PreDataLoader


def train(
    args,
    dataset,
    full_data_name,
    y,
    yc,
    ys,
    yj,
    ysj,
    lr,
    epoch,
    device,
    lambda_loss1,
    lambda_loss2,
    lambda_loss3,
    hidden_dim,
    sample_size=10,
    loss_step=20,
    real_loss=False,
    calculate_contextual=True,
    calculate_structural=True,
    calculate_joint=True,
    case=None,
    task_start_timestamp=None,
):

    encoder = args.encoder

    try:
        data = dataset.data
    except:
        data = dataset

    in_nodes = data.edge_index[0, :]
    out_nodes = data.edge_index[1, :]

    edge_index, edge_weight = get_normalization(
        data.edge_index, data.x.size(0), False, data.x.dtype, "sys"
    )
    avg_edge_index, avg_edge_weight = get_normalization(
        data.edge_index, data.x.size(0), False, data.x.dtype, "rw", laplacian=False
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    avg_edge_index = avg_edge_index.to(device)
    avg_edge_weight = avg_edge_weight.to(device)

    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))

    neighbor_num_list = torch.tensor(neighbor_num_list).to(device)

    in_dim = data.x.shape[1]
    GAEModel = GNNStructEncoder(
        args,
        dataset,
        in_dim,
        hidden_dim,
        2,
        sample_size,
        device=device,
        neighbor_num_list=neighbor_num_list,
        lambda_loss1=lambda_loss1,
        lambda_loss2=lambda_loss2,
        lambda_loss3=lambda_loss3,
    )
    GAEModel.to(device)
    degree_params = list(map(id, GAEModel.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params, GAEModel.parameters())

    opt = torch.optim.Adam(
        [
            {"params": base_params},
            {"params": GAEModel.degree_decoder.parameters(), "lr": 5e-3},
        ],
        lr=lr,
        weight_decay=0.0003,
    )
    min_loss = float("inf")
    arg_min_loss_per_node = None

    best_auc = 0
    best_pr_auc = 0
    best_auc_contextual = 0
    best_auc_dense_structural = 0
    best_auc_joint_type = 0
    best_auc_structure_type = 0

    # Create results folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "..", "results", full_data_name, encoder)
    path_parts = ["analysis", case]
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if task_start_timestamp is not None:
        path_parts.extend([f"Running_{task_start_timestamp}", f"result_{timestamp}"])
    else:
        path_parts.append(f"result_{timestamp}")

    result_dir = os.path.join(base_dir, *path_parts)
    os.makedirs(result_dir, exist_ok=True)

    loss_values = []
    h_loss_values = []
    degree_loss_values = []
    feature_loss_values = []

    contextual_score_values = []
    benchmark_score_values = []  # for real anomaly or combined anomaly
    benchmark_pr_score_values = []  # for real anomaly or combined anomaly
    dense_structural_score_values = []
    joint_type_score_values = []
    structure_type_score_values = []
    for i in tqdm(range(1, epoch + 1, 1)):

        (
            loss,
            h_loss,
            feature_loss,
            degree_loss,
            loss_per_node,
            h_loss_per_node,
            degree_loss_per_node,
            feature_loss_per_node,
        ) = GAEModel(
            edge_index,
            edge_weight,
            avg_edge_index,
            avg_edge_weight,
            data.x,
            neighbor_num_list,
            neighbor_dict,
        )

        loss_per_node = loss_per_node.cpu().detach()
        h_loss_per_node = h_loss_per_node.cpu().detach()
        degree_loss_per_node = degree_loss_per_node.cpu().detach()
        feature_loss_per_node = feature_loss_per_node.cpu().detach()

        h_loss_per_node_norm = h_loss_per_node / (
            torch.max(h_loss_per_node) - torch.min(h_loss_per_node)
        )
        degree_loss_per_node_norm = degree_loss_per_node / (
            torch.max(degree_loss_per_node) - torch.min(degree_loss_per_node)
        )
        feature_loss_per_node_norm = feature_loss_per_node / (
            torch.max(feature_loss_per_node) - torch.min(feature_loss_per_node)
        )

        comb_loss = (
            args.h_loss_weight * h_loss_per_node_norm
            + args.degree_loss_weight * degree_loss_per_node_norm
            + args.feature_loss_weight * feature_loss_per_node_norm
        )

        if real_loss:
            comp_loss = loss_per_node
        else:
            comp_loss = comb_loss

        auc_score = eval_roc_auc(y.numpy(), comp_loss.numpy()) * 100
        pr_auc_score = eval_pr_auc(y.numpy(), comp_loss.numpy()) * 100
        print()
        print(
            "Dataset Name: ",
            full_data_name,
            ", ROC-AUC Score(benchmark/combined): ",
            auc_score,
        )
        print(
            "Dataset Name: ",
            full_data_name,
            ", Pr-AUC Score(benchmark/combined): ",
            pr_auc_score,
        )

        if auc_score > best_auc:
            best_epoch = i

            # Store scores and labels
            best_scores = comp_loss.numpy()
            labels = y.numpy()
            score_label_df = pd.DataFrame(
                {"score": best_scores.ravel(), "label": labels.ravel()}
            )
            score_label_df.to_csv(
                os.path.join(
                    result_dir, f"best_score_label_{full_data_name}_{encoder}.csv"
                ),
                index=False,
            )

            # Store AUC-ROC curve
            fpr, tpr, thresholds_roc = roc_curve(
                y.numpy(), comp_loss.numpy()
            )  # label, score
            precision, recall, thresholds_pr = precision_recall_curve(
                y.numpy(), comp_loss.numpy()
            )  # label, score

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (area = {auc_score:.4f})")
            plt.plot([0, 1], [0, 1], "k--", label="Random guess")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve @ Epoch = {best_epoch}")
            plt.legend()
            plt.savefig(
                os.path.join(
                    result_dir, f"best_roc_auc_curve_{full_data_name}_{encoder}.png"
                )
            )
            plt.close()

            ## Store PR curve
            plt.figure()
            plt.plot(recall, precision, label=f"PR curve (area = {pr_auc_score:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve @ Epoch = {best_epoch}")
            plt.legend()
            plt.savefig(
                os.path.join(
                    result_dir, f"best_pr_auc_curve_{full_data_name}_{encoder}.png"
                )
            )
            plt.close()

            # ROC-AUC
            roc_conf_matrix_data = []
            for i, threshold in enumerate(thresholds_roc, 1):
                preds = (best_scores >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
                roc_conf_matrix_data.append(
                    {
                        "Index": i,
                        "threshold": threshold,
                        "true_negative": tn,
                        "false_positive": fp,
                        "false_negative": fn,
                        "true_positive": tp,
                        "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0,
                        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
                        "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                        "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                    }
                )

            # Save results to CSV
            roc_conf_matrix_df = pd.DataFrame(roc_conf_matrix_data)
            roc_csv_path = os.path.join(
                result_dir, f"roc_conf_matrix_thresholds_{full_data_name}_{encoder}.csv"
            )
            roc_conf_matrix_df.to_csv(
                roc_csv_path, index=False, sep=",", encoding="utf-8-sig"
            )

            # PR-AUC
            pr_conf_matrix_data = []
            for i, threshold in enumerate(thresholds_pr, 1):
                preds = (best_scores >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
                pr_conf_matrix_data.append(
                    {
                        "Index": i,
                        "threshold": threshold,
                        "true_negative": tn,
                        "false_positive": fp,
                        "false_negative": fn,
                        "true_positive": tp,
                        "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0,
                        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
                        "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                        "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                    }
                )

            # Save results to CSV
            pr_conf_matrix_df = pd.DataFrame(pr_conf_matrix_data)
            pr_csv_path = os.path.join(
                result_dir, f"pr_conf_matrix_thresholds_{full_data_name}_{encoder}.csv"
            )
            pr_conf_matrix_df.to_csv(
                pr_csv_path, index=False, sep=",", encoding="utf-8-sig"
            )

        best_auc = max(best_auc, auc_score)
        best_pr_auc = max(best_pr_auc, pr_auc_score)

        if calculate_contextual:
            contextual_auc_score = eval_roc_auc(yc.numpy(), comp_loss.numpy()) * 100
            print(
                "Dataset Name: ",
                full_data_name,
                ", AUC Score (contextual): ",
                contextual_auc_score,
            )
            best_auc_contextual = max(best_auc_contextual, contextual_auc_score)

        if calculate_structural:
            dense_structural_auc_score = (
                eval_roc_auc(ys.numpy(), comp_loss.numpy()) * 100
            )
            print(
                "Dataset Name: ",
                full_data_name,
                ", AUC Score (structural): ",
                dense_structural_auc_score,
            )
            best_auc_dense_structural = max(
                best_auc_dense_structural, dense_structural_auc_score
            )

        if calculate_joint:
            joint_type_auc_score = eval_roc_auc(yj.numpy(), comp_loss.numpy()) * 100
            print(
                "Dataset Name: ",
                full_data_name,
                ", AUC Score (joint-type): ",
                joint_type_auc_score,
            )
            best_auc_joint_type = max(best_auc_joint_type, joint_type_auc_score)

        if calculate_structural and calculate_joint:
            structure_type_auc_score = (
                eval_roc_auc(ysj.numpy(), comp_loss.numpy()) * 100
            )
            print(
                "Dataset Name: ",
                full_data_name,
                ", AUC Score (structure type): ",
                structure_type_auc_score,
            )
            best_auc_structure_type = max(
                best_auc_structure_type, structure_type_auc_score
            )

        print(
            "==========================================================================================="
        )

        print(
            "Dataset Name: ",
            full_data_name,
            " Best ROC-AUC Score(benchmark/combined): ",
            best_auc,
        )
        print(
            "Dataset Name: ",
            full_data_name,
            " Best Pr-AUC Score(benchmark/combined): ",
            best_pr_auc,
        )

        if calculate_contextual:
            print(
                "Dataset Name: ",
                full_data_name,
                " Best AUC Score (contextual): ",
                best_auc_contextual,
            )

        if calculate_structural:
            print(
                "Dataset Name: ",
                full_data_name,
                " Best AUC Score (structural): ",
                best_auc_dense_structural,
            )

        if calculate_joint:
            print(
                "Dataset Name: ",
                full_data_name,
                " Best AUC Score (joint-type): ",
                best_auc_joint_type,
            )

        if calculate_structural and calculate_joint:
            print(
                "Dataset Name: ",
                full_data_name,
                " Best AUC Score (structure type): ",
                best_auc_structure_type,
            )

        print(
            "==========================================================================================="
        )

        if loss < min_loss:
            min_loss = loss
            arg_min_loss_per_node = loss_per_node
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss = loss.cpu().detach()
        h_loss = h_loss.cpu().detach()
        degree_loss = degree_loss.cpu().detach()
        feature_loss = feature_loss.cpu().detach()

        loss_values.append(loss)
        h_loss_values.append(h_loss)
        degree_loss_values.append(degree_loss)
        feature_loss_values.append(feature_loss)

        benchmark_score_values.append(auc_score)
        benchmark_pr_score_values.append(pr_auc_score)
        if calculate_contextual:
            contextual_score_values.append(contextual_auc_score)
        if calculate_structural:
            dense_structural_score_values.append(dense_structural_auc_score)
        if calculate_joint:
            joint_type_score_values.append(joint_type_auc_score)
        if calculate_structural and calculate_joint:
            structure_type_score_values.append(structure_type_auc_score)

        if args.plot_loss and i % loss_step == 0:
            # Plot Loss
            plt.figure(1)
            plt.figure(figsize=(15, 9))
            plt.plot(np.array(loss_values), "r", label="Loss")
            plt.plot(np.array(h_loss_values), "g", label="Neighbor_Loss")
            plt.plot(np.array(feature_loss_values), "y", label="Feature_Loss")
            plt.title("Loss over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.pause(0.1)  # Pause to allow the image to update
            plt.clf()  # Clear the current figure to prepare for the next iteration

            # Plot AUC Scores
            plt.figure(2)
            plt.figure(figsize=(15, 9))
            plt.plot(
                np.array(benchmark_score_values),
                "g",
                label="Benchmark/Combined ROC-AUC",
            )
            plt.plot(
                np.array(benchmark_pr_score_values),
                "r",
                label="Benchmark/Combined Pr-AUC",
            )

            if calculate_contextual:
                plt.plot(np.array(contextual_score_values), "b", label="Contextual AUC")

            if calculate_structural:
                plt.plot(
                    np.array(dense_structural_score_values),
                    "c",
                    label="Dense Structural AUC",
                )

            if calculate_joint:
                plt.plot(np.array(joint_type_score_values), "m", label="Joint Type AUC")

            if calculate_structural and calculate_joint:
                plt.plot(
                    np.array(structure_type_score_values),
                    "y",
                    label="Structure Type AUC",
                )

            plt.title("AUC Scores over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("AUC Score")
            plt.legend()
            plt.pause(0.1)  # Pause to allow the image to update
            plt.clf()  # Clear the current figure to prepare for the next iteration

    # Save Final Figures
    # Save Loss Figure
    plt.figure(1)
    plt.figure(figsize=(15, 9))
    plt.plot(np.array(loss_values), "r", label="Loss")
    plt.plot(np.array(h_loss_values), "g", label="Neighbor_Loss")
    plt.plot(np.array(feature_loss_values), "y", label="Feature_Loss")
    plt.title("Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(result_dir, f"loss_{full_data_name}_{encoder}_{timestamp}.png")
    )
    plt.close()

    # Save AUC Scores Figure
    plt.figure(2)
    plt.figure(figsize=(15, 9))
    plt.plot(np.array(benchmark_score_values), "g", label="Benchmark ROC-AUC")
    plt.plot(np.array(benchmark_pr_score_values), "r", label="Benchmark Pr-AUC")
    if calculate_contextual:
        plt.plot(np.array(contextual_score_values), "b", label="Contextual AUC")
    if calculate_structural:
        plt.plot(
            np.array(dense_structural_score_values), "c", label="Dense Structural AUC"
        )
    if calculate_joint:
        plt.plot(np.array(joint_type_score_values), "m", label="Joint Type AUC")
    if calculate_structural and calculate_joint:
        plt.plot(np.array(structure_type_score_values), "y", label="Structure Type AUC")
    plt.title("AUC Scores over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("AUC Score")
    plt.legend()
    plt.savefig(
        os.path.join(
            result_dir, f"auc_scores_{full_data_name}_{encoder}_{timestamp}.png"
        )
    )
    plt.close()

    results_file = os.path.join(
        result_dir, f"results_{full_data_name}_{encoder}_{timestamp}.csv"
    )

    with open(results_file, "a", newline="") as file:
        writer = csv.writer(file)
        if os.stat(results_file).st_size == 0:
            writer.writerow(
                [
                    "Dataset Name",
                    "Encoder",
                    "Best ROC-AUC (benchmark/combined)",
                    "Best Pr-AUC (benchmark/combined)",
                    "Best AUC (contextual)",
                    "Best AUC (structural)",
                    "Best AUC (joint-type)",
                    "Best AUC (structure type)",
                ]
            )
        writer.writerow(
            [
                full_data_name,
                encoder,
                best_auc,
                best_pr_auc,
                best_auc_contextual,
                best_auc_dense_structural,
                best_auc_joint_type,
                best_auc_structure_type,
            ]
        )

    args_file = os.path.join(
        result_dir, f"args_{full_data_name}_{encoder}_{timestamp}.csv"
    )

    # Convert args to dictionary and save as JSON file
    args_dict = vars(args)
    with open(args_file, "w") as file:
        json.dump(args_dict, file, indent=4)

    return (
        min_loss.item(),
        arg_min_loss_per_node.cpu().detach(),
        best_auc,
        best_pr_auc,
        timestamp,
    )


def train_real_datasets(
    args,
    device,
    epoch_num=200,
    lr=5e-6,
    lambda_loss1=1e-2,
    lambda_loss2=1e-3,
    lambda_loss3=1e-3,
    sample_size=8,
    loss_step=20,
    hidden_dim=None,
    real_loss=False,
    calculate_contextual=False,
    calculate_structural=False,
    calculate_joint=False,
    case=None,
    task_start_timestamp=None,
):

    dataset_str = args.dataset

    synthetic_para = {
        "cora": {
            1: {
                "structural_n": 6,
                "structural_m": 5,
                "contextual_n": 30,
                "contextual_k": 10,
                "joint_structural_n": 30,
                "joint_structural_m": 10,
            },
            5: {
                "structural_n": 28,
                "structural_m": 5,
                "contextual_n": 140,
                "contextual_k": 10,
                "joint_structural_n": 140,
                "joint_structural_m": 10,
            },
            10: {
                "structural_n": 18,
                "structural_m": 15,
                "contextual_n": 270,
                "contextual_k": 10,
                "joint_structural_n": 270,
                "joint_structural_m": 10,
            },
            20: {
                "structural_n": 36,
                "structural_m": 15,
                "contextual_n": 540,
                "contextual_k": 10,
                "joint_structural_n": 540,
                "joint_structural_m": 10,
            },
        },
        "amazon": {
            1: {
                "structural_n": 6,
                "structural_m": 5,
                "contextual_n": 30,
                "contextual_k": 10,
                "joint_structural_n": 30,
                "joint_structural_m": 10,
            },
            5: {
                "structural_n": 28,
                "structural_m": 5,
                "contextual_n": 140,
                "contextual_k": 10,
                "joint_structural_n": 140,
                "joint_structural_m": 10,
            },
            10: {
                "structural_n": 18,
                "structural_m": 15,
                "contextual_n": 270,
                "contextual_k": 10,
                "joint_structural_n": 270,
                "joint_structural_m": 10,
            },
            20: {
                "structural_n": 36,
                "structural_m": 15,
                "contextual_n": 540,
                "contextual_k": 10,
                "joint_structural_n": 540,
                "joint_structural_m": 10,
            },
        },
    }
    if dataset_str in ["cora", "amazon"]:
        if args.inject_typ == "str":
            args.structural_n = synthetic_para[dataset_str][args.inject_percentage][
                "structural_n"
            ]
            args.structural_m = synthetic_para[dataset_str][args.inject_percentage][
                "structural_m"
            ]
            synthetic_para = [
                args.inject_typ,
                args.inject_percentage,
                args.structural_n,
                args.structural_m,
            ]

        elif args.inject_typ == "ctx":
            args.contextual_n = synthetic_para[dataset_str][args.inject_percentage][
                "contextual_n"
            ]
            args.contextual_k = synthetic_para[dataset_str][args.inject_percentage][
                "contextual_k"
            ]
            synthetic_para = [
                args.inject_typ,
                args.inject_percentage,
                args.contextual_n,
                args.contextual_k,
            ]

        elif args.inject_typ == "jnt":
            args.joint_structural_n = synthetic_para[dataset_str][
                args.inject_percentage
            ]["joint_structural_n"]
            args.joint_structural_m = synthetic_para[dataset_str][
                args.inject_percentage
            ]["joint_structural_m"]
            synthetic_para = [
                args.inject_typ,
                args.inject_percentage,
                args.joint_structural_n,
                args.joint_structural_m,
            ]

        else:
            error_message = (
                f"Dataset {dataset_str} is supposed to use synthetic anomaly since there is no organic anomaly inside. "
                f'And type of injected anomaly data is supposed to be "str", "ctx", or "jnt", instead of the unknown type {args.inject_typ}.'
            )
            raise ValueError(error_message)

    else:
        synthetic_para = None

    try:
        dataset, full_data_name = PreDataLoader(
            args.dataset,
            args.encoder,
            suffix=args.suffix,
            synthetic_para=synthetic_para,
        )
    except TypeError as e:
        raise TypeError(f"Type error occurred: {e}")
    except ValueError as e:
        raise ValueError(f"Value error occurred: {e}")

    print(f"{args.dataset} loaded!")

    try:
        data = dataset.data
    except:
        data = dataset

    # Initialize as zero-dimensional tensors; adjust dtype based on your function's output
    yc = torch.tensor([], dtype=torch.int32)
    ys = torch.tensor([], dtype=torch.int32)
    yj = torch.tensor([], dtype=torch.int32)
    ysj = torch.tensor([], dtype=torch.int32)

    if calculate_contextual:
        if dataset_str == "inj_cora":
            yc = (data.y >> 0 & 1).to(torch.int32)  # Ensure this is a tensor
        else:
            data, yc = gen_contextual_outlier(
                data=data, n=args.contextual_n, k=args.contextual_k
            )

        yc = yc.cpu().detach()

    if calculate_structural:
        if dataset_str == "inj_cora":
            ys = (data.y >> 1 & 1).to(
                torch.int32
            )  # Ensure this is a tensor # structural outliers
        else:
            data, ys = gen_structural_outlier(
                data=data, n=args.structural_n, m=args.structural_m, p=0.2
            )

        ys = ys.cpu().detach()

    if calculate_joint:
        data, yj = gen_joint_structural_outlier(
            data=data, n=args.joint_structural_n, m=args.joint_structural_m
        )
        yj = yj.cpu().detach()
        if calculate_structural:
            ysj = torch.logical_or(ys, yj).int()

    if args.use_combine_outlier:
        if calculate_contextual and calculate_structural:
            data.y = torch.logical_or(ys, yc).int()
        else:
            print("=" * 25)
            print("Logical Error!")
            print("Combine Outlier is required to derive")
            output = "However, "
            if calculate_contextual:
                output += "Contextual Anomalies are derived"
            else:
                output += "Contextual Anomalies are NOT derived"

            output += " and "

            if calculate_structural:
                output += "Structural Anomalies are derived"
            else:
                output += "Structural Anomalies are NOT derived"

            print(output)
            print("=" * 25)
            print()

    y = data.y.bool()  # binary labels (inlier/outlier)
    y = y.cpu().detach()

    edge_index = data.edge_index.cpu()

    num_nodes = data.x.shape[0]
    self_edges = torch.tensor(
        [[i for i in range(num_nodes)], [i for i in range(num_nodes)]]
    )
    edge_index = torch.cat([edge_index, self_edges], dim=1)
    data.edge_index = edge_index
    data = data.to(device)

    for name, tensor in [
        ("Organic", y),
        ("Contextual", yc),
        ("Structrual", ys),
        ("Joint", yj),
        ("Structure", ysj),
    ]:
        if tensor.numel() > 0:
            ones, _, ones_percentage, _ = count_ones_and_zeros(tensor)
            print(
                f"{name}: Number of such type of anomalies = {ones}, Percentage = {ones_percentage * 100}%"
            )

    loss, loss_per_node, best_auc, best_pr_auc, timestamp = train(
        args,
        dataset,
        full_data_name,
        y,
        yc,
        ys,
        yj,
        ysj,
        lr=lr,
        epoch=epoch_num,
        device=device,
        lambda_loss1=lambda_loss1,
        lambda_loss2=lambda_loss2,
        lambda_loss3=lambda_loss3,
        hidden_dim=hidden_dim,
        sample_size=sample_size,
        loss_step=loss_step,
        real_loss=real_loss,
        calculate_contextual=calculate_contextual,
        calculate_structural=calculate_structural,
        calculate_joint=calculate_joint,
        case=case,
        task_start_timestamp=task_start_timestamp,
    )

    return loss, loss_per_node, best_auc, best_pr_auc, timestamp
