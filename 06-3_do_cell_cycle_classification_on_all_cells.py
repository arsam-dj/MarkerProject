import argparse
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import polars as pl
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to database with cells to be classified.')
parser.add_argument('-p', '--plate', default='', help='Plate identifier.')
parser.add_argument('-c', '--cell_cycle_directory', default='', help='Directory for writing cell cycle output files.')
parser.add_argument('-f', '--cell_cycle_features_path', default='', help='Path to file with all cells and their cell cycle features.')
parser.add_argument('-m', '--lgb_model', default='', help='Path to LGB model for cell cycle classification.')


args = parser.parse_args()


### Function for doing cell cycle classification on the full dataset
def do_cell_cycle_classification_on_full_data(lgb_model_path, cells_and_features, db_path, plate, output_dir):
    """
    Does cell cycle classification on all cells in a given database using Light Gradient Boost. Modifies database by
    adding new 'CC_Pred', 'G1_Prob', 'S/G2_Prob', 'MAT_Prob', and 'Max_Prop' (max of G1/SG2/MAT probs) columns. The
    CC Stage with the highest probability is chosen as the final prediction.

    Args:
        lgb_model_path (str): path to a pre-trained light gradient boost model
        cells_and_features (pl.DataFrame): dataframe with all cells and their cell cycle features to be used for classification
        db_path (str): path to database to be modified
        plate (str): plate identifier for saving file with cell classifications
        output_dir (str): where to save cell IDs and classification labels as csv
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    lgb_model = lgb.Booster(model_file=lgb_model_path)

    # Subset all cells to only include those in dataframe
    db_cells = (
        pl
        .read_database(query="SELECT Cell_ID FROM Per_Cell;", connection=conn)
        .to_series()
        .to_list()
    )
    cells_and_features = (
        cells_and_features
        .filter(
            pl.col("Cell_ID").is_in(db_cells)
        )
    )

    # Separate cell info from features, turn to pandas
    cell_ids = cells_and_features.select(["Replicate", "Cell_ID", "ORF", "Name", "Strain_ID"]).to_pandas()
    cell_features = cells_and_features.drop(["Replicate", "Cell_ID", "ORF", "Name", "Strain_ID"]).to_pandas()
    cell_features = cell_features.apply(pd.to_numeric, errors="coerce") # ensure all features are float

    # Use model to predict labels
    cell_pred_probs = lgb_model.predict(cell_features) # probability of each class
    cell_pred_probs_df = pd.DataFrame({"G1_Prob": cell_pred_probs[:, 0],
                                       "SG2_Prob": cell_pred_probs[:, 1],
                                       "MAT_Prob": cell_pred_probs[:, 2],
                                       "Max_Prob": np.max(cell_pred_probs, axis=1)}
                                      )
    cell_preds = cell_pred_probs.argmax(axis=1) # final class chosen based on which has highest prob
    cell_preds_df = pd.DataFrame({"Predicted_Label": cell_preds})

    # Re-attach Cell_IDs with their predictions, add to database
    # 1. Insert information as a new temp table
    combined_df = (
        pd
        .concat([cell_ids, cell_preds_df, cell_pred_probs_df], axis=1)
        .replace({"Predicted_Label": {0: "G1", 1: "SG2", 2: "MAT"}})
        )
    combined_df.to_csv(f"{output_dir}/{plate}_classified_cells.csv", index=False)
    combined_df.to_sql("temp_table", conn, index=False, if_exists="replace")

    # 2. Create new columns in Per_Cell that exist in combined_df
    missing_columns = [x for x in combined_df.columns if x not in {"Replicate", "Cell_ID", "ORF", "Name", "Strain_ID"}]
    for col in missing_columns:
        datatype = "TEXT"
        if combined_df[col].dtype == "float64" or combined_df[col].dtype == "float32":
            datatype = "FLOAT"
        cursor.execute(f'ALTER TABLE Per_Cell ADD COLUMN "{col}" {datatype}')

    # Step 3: Create indexes on join keys (if they don't already exist)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_Per_Cell ON Per_Cell (Cell_ID)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_temp_table ON temp_table (Cell_ID)")
    conn.commit()

    # Step 4: Use a CTE for faster updates
    update_clause = ", ".join(
        f"{col} = (SELECT temp_table.{col} FROM temp_table WHERE temp_table.Cell_ID = Per_Cell.Cell_ID)"
        for col in missing_columns
    )
    update_query = f"""
        UPDATE Per_Cell
        SET {update_clause}
        WHERE EXISTS (
            SELECT 1 FROM temp_table
            WHERE temp_table.Cell_ID = Per_Cell.Cell_ID
        )
    """
    cursor.execute(update_query)

    # Commit and close connection
    cursor.execute("DROP TABLE temp_table;")
    cursor.execute("DROP INDEX idx_Per_Cell;")
    conn.commit()
    conn.close()


if __name__ == '__main__':

    if not os.path.exists(args.cell_cycle_directory):
        os.makedirs(args.cell_cycle_directory)

    # Open file with all cells and cell cycle features to use for classification
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

    labelled_cells_and_features = (
        pl
        .read_csv(args.cell_cycle_features_path)
        .select(classification_features)
        .unique()
    )

    # Do classification on all cells
    do_cell_cycle_classification_on_full_data(
        lgb_model_path=args.lgb_model,
        cells_and_features=labelled_cells_and_features,
        db_path=args.database_path,
        plate=args.plate,
        output_dir=args.cell_cycle_directory
    )

    print("Complete.")

