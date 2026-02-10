import argparse
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import polars as pl
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--labelled_data_path', default='', help='Path to file with cells and their cell cycle labels.')
parser.add_argument('-f', '--cell_cycle_features_path', default='', help='Path to file with all cells and their cell cycle features.')
parser.add_argument('-x', '--cell_coordinates', default='', help='Path to file with all cell overlay coordinates.')
parser.add_argument('-c', '--cell_cycle_directory', default='', help='Directory for writing cell cycle output files.')

args = parser.parse_args()


### Function for merging labelled cells with respective classification features
def make_train_test_dataset(classification_features, labelled_data_path, cell_cycle_features_path, output_dir):
    """
    Combine labelled cells and Cell_IDs with their respective features to be used in cell cycle classification.

    Args:
        classification_features (list(str)): features to select for cell cycle classification
        labelled_data_path (str): path to labelled dataset
        cell_cycle_features_path (str): path to file with all cells and their cell cycle features
        output_dir (str): location to save combined dataset to
    """

    # Load labelled data and attach classification features
    labelled_data = (
        pl
        .read_csv(labelled_data_path)
        .select(["Cell_ID", "Saved_Label"])
        .with_columns(
            pl.col("Saved_Label").replace({"G1": 0, "S/G2": 1, "MAT": 2}).cast(pl.Int64).alias("Saved_Label")
        )
    )

    labelled_cells_and_features = (
        pl
        .read_csv(cell_cycle_features_path)
        .select(classification_features)
        .unique()
        .join(labelled_data, on="Cell_ID", how="right")
    )

    labelled_cells_and_features = (
        labelled_cells_and_features
        .with_columns(
            [
                pl.col(col).cast(pl.Float64)
                for col in labelled_cells_and_features.columns
                if col not in ["Replicate", "Cell_ID", "ORF", "Name", "Strain_ID", "Saved_Label"]
            ]
        )
    )

    labelled_cells_and_features.write_csv(f"{output_dir}/labelled_cells_and_features.csv")
    return labelled_cells_and_features

### Function for checking model performance across different training/testing set sizes
def compare_train_test_sizes_and_model_performance(labelled_cells_and_features, output_dir, training_testing_sizes=None):
    """
    Test how model performs on different training/testing set sizes. Metrics used will be per-set size
    log loss and set size X class F1 scores. Outputs will be printed to a log file for viewing.

    Args:
        labelled_cells_and_features (pl.DataFrame): dataframe with labelled cells and cell cycle classification features
        output_dir (str): location to save outputs to
        training_testing_sizes (Optional(list[int])): optional parameter; can provide specific set sizes for training/testing
    """
    # Shuffle labelled cell dataframe rows
    labelled_cells_and_features = labelled_cells_and_features.to_pandas()
    labelled_cells_and_features = labelled_cells_and_features.sample(frac=1, random_state=1705).reset_index(drop=True)

    # Create logfile
    log_file = open(f"{output_dir}/model_performance_log.txt", "a")

    # Declare feature vector and target variable
    X = labelled_cells_and_features.drop(["Replicate", "Cell_ID", "ORF", "Name", "Strain_ID", "Saved_Label"], axis=1)  # only keep features
    y = labelled_cells_and_features["Saved_Label"]
    classes = list(set(y))
    classes.sort()

    # Define train sizes as part of checking whether training set is large enough
    if training_testing_sizes:
        train_sizes = training_testing_sizes
    else:
        train_sizes = [1000, 2000, 4000, 6000, 8000, 10000, labelled_cells_and_features.shape[0]]
    n_splits = 5  # 4 parts train, 1 part test

    per_class_f1_scores = defaultdict(list)  # new key added to dict if it doesn't exist (instead of throwing error)

    for size in train_sizes:
        log_file.write(f"Training with {size} samples...\n")

        # Subset data
        X_sub = X.iloc[:size]
        y_sub = y[:size]

        # StratifiedKFold preserves percentage of samples for each class
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1705)
        fold_losses = []
        fold_test_accuracies = []
        fold_train_accuracies = []

        # train_idx and test_idx are indices of train and test samples for each fold
        # If n_splits is 5, there will be 5 train_idx lists and 5 test_idx lists of indices
        for train_idx, test_idx in skf.split(X_sub, y_sub):
            X_train, X_test = X_sub.iloc[train_idx], X_sub.iloc[test_idx]  # train/test features
            y_train, y_test = y_sub[train_idx], y_sub[test_idx]  # train/test labels

            # Fit model on train data
            lgb_model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=len(classes),
                learning_rate=0.1,
                n_estimators=100,
                random_state=1705,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)

            # Predict labels for test data
            y_test_probs = lgb_model.predict_proba(X_test)  # probability of each class
            y_test_pred = y_test_probs.argmax(axis=1)  # final class chosen based on which has highest prob

            # Predict labels for train data (used later for checking overfitting)
            y_train_probs = lgb_model.predict_proba(X_train)  # probability of each class
            y_train_pred = y_train_probs.argmax(axis=1)

            # Log loss
            # 1. Quantifies performance of classification model by measuring the difference between predicted probabilities
            #    and actual outcomes
            # 2. Unlike Accuracy, LL takes into account uncertainty of predictions and penalizes models for confidently
            #    incorrect predictions
            # 3. Lower LL values indicate better model performance
            loss = log_loss(y_test, y_test_probs)  # input actual test classes and predicted class probabilities
            fold_losses.append(loss)

            # Accuracy
            # 1. One way of checking overfitting is to compare the model accuracy on the test set with the training set.
            # 2. If accuracies are similar, there is little overfitting (because the model generalizes well to test set --
            #    data it hasn't seen before).
            # 3. If the test set accuracy is markedly low relative to train set, then that indicates overfitting.
            # 4. Accuracy can be problematic if labelled dataset is very unbalanced, but in this case each class has
            #    3500 labelled cells so I will assume that folds are about equal too.
            test_accuracy = accuracy_score(y_test_pred, y_test)
            train_accuracy = accuracy_score(y_train_pred, y_train)

            fold_test_accuracies.append(test_accuracy)
            fold_train_accuracies.append(train_accuracy)

            # Per-class F1
            # 1. F1 score is the harmonic mean between precision and recall
            # 2. F1 = 2 * ((precision * recall) / (precision + recall))
            # 3. Since accuracy is sensitive towards unbalanced datasets, F1/precision/recall are better metrics.
            # 4. Final per_class_f1_scores will have a key for each class and a list of n_splits F1 values
            report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
            for idx, class_name in enumerate(classes):
                f1 = report[str(idx)]["f1-score"]  # get F1 for each class...
                per_class_f1_scores[class_name].append(f1)  # and save it

        # For each train_size, save average fold-loss across all n_splits
        log_file.write(f"Mean CV Log Loss: {np.mean(fold_losses):.4f}\n")

        # Report train and test set accuracies
        log_file.write(f"Mean accuracy on train set: {np.mean(fold_train_accuracies):.4f}\n")
        log_file.write(f"Mean accuracy on test set: {np.mean(fold_test_accuracies):.4f}\n")

        # Report F1 scores for each class
        log_file.write("Per-class F1s (mean):\n")
        for c in classes:
            scores = per_class_f1_scores[c][-n_splits:]
            log_file.write(f"  {c}: {np.mean(scores):.3f}\n")

        log_file.write(f"\n\n")


### Function for generating the final LGB model and saving it (+ relevant metrics)
def generate_final_lgb_model(labelled_cells_and_features, output_dir):
    """
    Trains a Light Gradient Boost Model using all labelled cells. Saves model, model predictions, confusion matrix,
    model report, feature importance chart, and visualization of the first tree inside output_dir.

    Args:
        labelled_cells_and_features (pl.DataFrame): dataframe with labelled cells and cell cycle classification features
        output_dir (str): location to save outputs to
    """

    # Shuffle labelled cell dataframe rows
    labelled_cells_and_features = labelled_cells_and_features.to_pandas()
    labelled_cells_and_features = (
        labelled_cells_and_features
        .sample(frac=1, random_state=1705).reset_index(drop=True)
        .rename(columns={"Saved_Label": "Actual_Label"})
    )
    class_counts = Counter(labelled_cells_and_features["Actual_Label"])
    label_names = {0: "G1", 1: "S/G2", 2: "MAT"}

    # Create log file and start by saving label counts
    log_file = open(f"{output_dir}/final_model_performance_metrics.txt", "w")
    log_file.write("Light Gradient Boost Model for Cell Cycle Classification\n\n")
    log_file.write(f"Size of full labelled dataset: {labelled_cells_and_features.shape[0]}\n")
    for label, name in label_names.items():
        log_file.write(f"{name} Labels: {class_counts[label]}\n")
    log_file.write("\n\n")

    # Declare feature vector and target variable
    X = labelled_cells_and_features.drop(["Replicate", "Cell_ID", "ORF", "Name", "Strain_ID", "Actual_Label"], axis=1)  # only keep features
    y = labelled_cells_and_features["Actual_Label"]

    # Split dataset into training and test set
    # test_size=0.2 because when getting validation metrics, I did 5-fold cross validation (4 parts train 1 part test which is
    # same as 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1705)

    log_file.write("Train and Test Set Sizes\n")
    for label, name in label_names.items():
        log_file.write(f"{name} Train: {Counter(y_train)[label]}\n")
    log_file.write("\n\n")

    for label, name in label_names.items():
        log_file.write(f"{name} Test: {Counter(y_test)[label]}\n")
    log_file.write("\n\n\n")

    # Create LGBM model, train, and save
    lgb_model = lgb.LGBMClassifier(
        task="train",
        objective="multiclass",
        boosting="gbdt",
        num_iterations=100,
        learning_rate=0.1,
        seed=1705,
        force_col_wise=False,
        force_row_wise=False,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_model.booster_.save_model(f"{output_dir}/cell_cycle_classification_lgbm_model.txt")

    # Use model to predict labels on training/test sets
    y_train_probs = lgb_model.predict_proba(X_train)  # probability of each class
    y_train_probs_df = pd.DataFrame(
        {"G1_Prob": y_train_probs[:, 0], "S/G2_Prob": y_train_probs[:, 1], "MAT_Prob": y_train_probs[:, 2]})
    y_train_pred = y_train_probs.argmax(axis=1)  # final class chosen based on which has highest prob
    y_train_pred_df = pd.DataFrame({"Predicted_Label": y_train_pred})

    y_test_probs = lgb_model.predict_proba(X_test)
    y_test_probs_df = pd.DataFrame(
        {"G1_Prob": y_test_probs[:, 0], "S/G2_Prob": y_test_probs[:, 1], "MAT_Prob": y_test_probs[:, 2]})
    y_test_pred = y_test_probs.argmax(axis=1)
    y_test_pred_df = pd.DataFrame({"Predicted_Label": y_test_pred})

    # Save Cell_IDs together with actual labels and predicted labels
    combined_df = (
        pd
        .concat([
            pd.concat([X_train.reset_index(drop=True), y_train_pred_df.reset_index(drop=True),
                       y_train_probs_df.reset_index(drop=True)], axis=1),
            pd.concat([X_test.reset_index(drop=True), y_test_pred_df.reset_index(drop=True),
                       y_test_probs_df.reset_index(drop=True)], axis=1)
        ], axis=0))
    joined_df = (
        pd
        .merge(labelled_cells_and_features, combined_df, on=list(X_train.columns))
        .replace({"Predicted_Label": label_names, "Actual_Label": label_names})
    )[["Replicate", "Cell_ID", "ORF", "Name", "Strain_ID", "Actual_Label", "Predicted_Label", "G1_Prob", "S/G2_Prob", "MAT_Prob"]]
    joined_df.to_csv(f"{output_dir}/actual_and_predicted_labels.csv", index=False)

    # Record test/training accuracy
    train_accuracy = accuracy_score(y_train_pred, y_train)
    test_accuracy = accuracy_score(y_test_pred, y_test)

    log_file.write("Accuracy Scores\n")
    log_file.write(f"Train set accuracy: {train_accuracy:.4f}\n")
    log_file.write(f"Test set accuracy: {test_accuracy:.4f}\n\n\n")

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_matrix = pd.DataFrame(
        data=cm,
        columns=['Predicted G1', 'Predicted S/G2', 'Predicted MAT'],
        index=['Actual G1', 'Actual S/G2', 'Actual MAT']
    )
    cm_plot = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    figure = cm_plot.get_figure()
    figure.savefig(f"{output_dir}/confusion_matrix.pdf")
    plt.close(cm_plot.figure)

    # Save classification metrics
    log_file.write(f"{classification_report(y_test, y_test_pred)}\n\n\n")

    # Feature importance
    lgb.plot_importance(lgb_model, max_num_features=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=500)
    plt.close()

    # First decision tree example
    lgb.plot_tree(lgb_model, tree_index=0, figsize=(15, 5),
                  show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
    plt.title("First Decision Tree for Cell Cycle Classification")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/decision_tree.png", bbox_inches='tight', dpi=500)
    plt.close()

    log_file.close()

    return joined_df


### Function for checking label mismatches
def generate_sct_inputs_for_checking_label_mismatches(cell_coords, classified_cells, output_dir):
    """
    Generates SingleCellTool inputs for cells that have different actual and predicted labels.

    Args:
        cell_coords (str): path to file with Cell_ID, Image_Path, Center_X, and Center_Y cells for all labelled cells
        classified_cells (str): file with Cell_ID, Actual_Label, and Predicted_Label for all labelled cells
        output_dir (str): path to output directory for saving sct files
    """

    coords = pl.read_csv(cell_coords)

    classified_cells = (
        pl
        .from_pandas(classified_cells)
        .filter(pl.col("Actual_Label") != pl.col("Predicted_Label"))
        .join(coords, on="Cell_ID", how="left")
        .select(["Cell_ID", "Image_Path", "Center_X", "Center_Y", "Actual_Label", "Predicted_Label"])
    )

    classified_cells.write_csv(f"{output_dir}/mismatched_labels.csv")

    (
        classified_cells
        .filter(
            (pl.col("Actual_Label") == "G1") & (pl.col("Predicted_Label") == "S/G2")
        )
        .select(["Cell_ID", "Image_Path", "Center_X", "Center_Y"])
        .write_csv(
            f"{output_dir}/actual_g1_predicted_sg2.csv")
    )

    (
        classified_cells
        .filter(
            (pl.col("Actual_Label") == "S/G2") & (pl.col("Predicted_Label") == "G1")
        )
        .select(["Cell_ID", "Image_Path", "Center_X", "Center_Y"])
        .write_csv(
            f"{output_dir}/actual_sg2_predicted_g1.csv")
    )

    (
        classified_cells
        .filter(
            (pl.col("Actual_Label") == "S/G2") & (pl.col("Predicted_Label") == "MAT")
        )
        .select(["Cell_ID", "Image_Path", "Center_X", "Center_Y"])
        .write_csv(
            f"{output_dir}/actual_sg2_predicted_mat.csv")
    )

    (
        classified_cells
        .filter(
            (pl.col("Actual_Label") == "MAT") & (pl.col("Predicted_Label") == "S/G2")
        )
        .select(["Cell_ID", "Image_Path", "Center_X", "Center_Y"])
        .write_csv(
            f"{output_dir}/actual_mat_predicted_sg2.csv")
    )


if __name__ == '__main__':

    if not os.path.exists(args.cell_cycle_directory):
        os.makedirs(args.cell_cycle_directory)

    if not os.path.exists(f"{args.cell_cycle_directory}/classification_model"):
        os.makedirs(f"{args.cell_cycle_directory}/classification_model")

    if not os.path.exists(f"{args.cell_cycle_directory}/classification_model/sct_inputs_for_mismatched_cells"):
        os.makedirs(f"{args.cell_cycle_directory}/classification_model/sct_inputs_for_mismatched_cells")

    classification_features = [
        "Replicate", "Cell_ID", "ORF", "Name", "Strain_ID",
        "Cell_AreaShape_Compactness", "Cell_AreaShape_Eccentricity",
        "Cell_AreaShape_FormFactor", "Cell_AreaShape_MajorAxisLength", "Cell_AreaShape_MinorAxisLength",
        "Cell_AreaShape_Solidity", "Cell_Children_Nuclei_Count",
        "Cell_Mean_Nuclei_AreaShape_Compactness", "Cell_Mean_Nuclei_AreaShape_Eccentricity",
        "Cell_Mean_Nuclei_AreaShape_FormFactor", "Cell_Mean_Nuclei_AreaShape_MajorAxisLength",
        "Cell_Mean_Nuclei_AreaShape_MinorAxisLength", "Cell_Mean_Nuclei_AreaShape_Solidity",
        "Cell_Mean_Nuclei_Distance_Centroid_Cell", "Cell_MajorAL_Over_Minor_AL",
        "Nucleus_MajorAL_Over_MinorAL", "Nuclear_Distance", "NucDist_Over_CellMajAL"
    ]

    labelled_cells_and_features = make_train_test_dataset(
        classification_features=classification_features,
        labelled_data_path=args.labelled_data_path,
        cell_cycle_features_path=args.cell_cycle_features_path,
        output_dir=args.cell_cycle_directory)

    compare_train_test_sizes_and_model_performance(
        labelled_cells_and_features=labelled_cells_and_features,
        output_dir=f"{args.cell_cycle_directory}/classification_model",
        training_testing_sizes=None)

    classified_cells = generate_final_lgb_model(
        labelled_cells_and_features=labelled_cells_and_features,
        output_dir=f"{args.cell_cycle_directory}/classification_model")

    generate_sct_inputs_for_checking_label_mismatches(
        cell_coords=args.cell_coordinates,
        classified_cells=classified_cells,
        output_dir=f"{args.cell_cycle_directory}/classification_model/sct_inputs_for_mismatched_cells")

    print("Complete.")

