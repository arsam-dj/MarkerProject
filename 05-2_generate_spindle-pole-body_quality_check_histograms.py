import argparse
import os
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sqlite3

from GEN_quality_check_functions import feature_distributions_matrix

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--qc_directory', default='', help='Path to directory to write quality check files to.')
parser.add_argument('-d', '--database_directory', default='', help='Path to directory with all databases.')
parser.add_argument('-x', '--cell_coordinates', default='', help='Path to file with all cell overlay coordinates.')

args = parser.parse_args()


# Function for creating dataframes with QC features of interest and paths to segmentation masks for manual assessments
def create_qc_raw_scaled_df(database_directory, coordinates_path, qc_directory):
    """
    Create a dataframe with cells from all three reps, as well as QC features of interest. Create
    a second dataframe with scaled feature values (z-scores). Include the segmentation mask paths so they can be viewed with Single Cell Tool.

    Args:
        database_directory (str): path to directory with all plate databases
        coordinates_path (str): path to file with cell overlay coordinates
        qc_directory (str): path to directory to write qc files to
    """
    qc_features = [
        "SPBs_AreaShape_Eccentricity",
        "SPBs_AreaShape_Perimeter",
        "SPBs_AreaShape_MajorAxisLength",
        "SPBs_AreaShape_Solidity",
        "SPBs_Distance_Minimum_Cell"
    ]

    # Read all plate databases and get cell info + qc feature columns, combine
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]
    qc_dfs = []

    for db_path in databases:
        conn = sqlite3.connect(db_path)

        plate_qc_df = pl.read_database(
            query=f"""
                    SELECT 
                        Per_SPBs.Cell_ID, 
                        SPBs_Number_Object_Number, 
                        {', '.join(qc_features)}, 
                        SPBs_AreaShape_Area / Cell_AreaShape_Area AS SPB_Area_Over_Cell_Area,
                        SPBs_AreaShape_MajorAxisLength / SPBs_AreaShape_MinorAxisLength AS SPB_MajAL_Over_MinAL
                    FROM Per_SPBs
                    JOIN (SELECT Cell_ID, Cell_AreaShape_Area FROM Per_Cell) pc
                        ON Per_SPBs.Cell_ID = pc.Cell_ID;
                    """,
            connection=conn)
        qc_dfs.append(plate_qc_df)

        conn.close()

    qc_df = (
        pl
        .concat(items=qc_dfs, how="vertical")
    )

    # Add segmentation mask paths
    qc_df = (
        pl
        .read_csv(coordinates_path)
        .join(qc_df, on=["Cell_ID"])
    )
    qc_df.write_csv(file=f"{qc_directory}/raw_spb_qc_features.csv")

    # Standardize features so all features are on the same scale. Here, imputation is done
    # by replacing NaNs with the column mean (otherwise, StandardScaler will not work).
    qc_features = qc_features + ["SPB_Area_Over_Cell_Area", "SPB_MajAL_Over_MinAL"]

    cell_info_df = qc_df.select(
        ["Image_Path", "Center_X", "Center_Y"] + ["Cell_ID", "SPBs_Number_Object_Number"])
    qc_features_df = qc_df.select(qc_features).to_numpy()

    # Impute missing values (replace NaNs with column mean)
    imputer = SimpleImputer(strategy="mean")
    qc_features_df = pl.DataFrame(imputer.fit_transform(qc_features_df))
    qc_features_df.columns = qc_features

    # Standardize features
    scaler = StandardScaler()
    scaled_data = pl.DataFrame(scaler.fit_transform(qc_features_df))
    scaled_data.columns = qc_features_df.columns

    # Save standardized features
    qc_df_scaled = pl.concat([cell_info_df, scaled_data], how='horizontal')
    qc_df_scaled.write_csv(file=f"{qc_directory}/scaled_spb_qc_features.csv")

    return qc_df, qc_df_scaled


if __name__ == '__main__':
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    qc_df_raw, _ = create_qc_raw_scaled_df(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)

    features_to_plot = (
        qc_df_raw
        .with_columns(
            (
                pl
                .when(pl.col("SPB_Area_Over_Cell_Area") > 1)
                .then(1)
                .otherwise(pl.col("SPB_Area_Over_Cell_Area"))
            ).alias("SPB_Area_Over_Cell_Area")
        )
        .rename({
            "SPBs_AreaShape_Perimeter": "SPBs_AreaShape_Perimeter (Raw)",
            "SPBs_AreaShape_Solidity": "SPBs_AreaShape_Solidity (Raw)",
            "SPBs_Distance_Minimum_Cell": "SPBs_Distance_Minimum_Cell (Raw)" ,
            "SPB_Area_Over_Cell_Area": "SPB_Area_Over_Cell_Area (Raw)",
            "SPB_MajAL_Over_MinAL": "SPB_MajAL_Over_MinAL (Raw)",
            "SPBs_AreaShape_Eccentricity": "SPBs_AreaShape_Eccentricity (Raw)",
            "SPBs_AreaShape_MajorAxisLength": "SPBs_AreaShape_MajorAxisLength (Raw)"
        })
        .drop(["Cell_ID", "Image_Path", "Center_X", "Center_Y", "SPBs_Number_Object_Number"])
    )

    feature_distributions_matrix(
        qc_features=features_to_plot,
        qc_directory=args.qc_directory,
        output_figure_name='SPBs_qc_feature_distributions')

    print("Complete.")