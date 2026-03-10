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
def create_qc_raw_df(database_directory, coordinates_path, qc_directory):
    """
    Create a dataframe with cells from all three reps, as well as QC features of interest. Create a second dataframe
    with scaled feature values (z-scores). Include the segmentation mask paths so they can be viewed with Single Cell Tool.

    Args:
        database_directory (str): path to directory with all plate databases
        coordinates_path (str): path to file with cell overlay coordinates
        qc_directory (str): path to directory to write qc files to
    """
    qc_features = [
        "Peroxisomes_AreaShape_Area",
        "Peroxisomes_AreaShape_Perimeter",
        "Peroxisomes_AreaShape_Eccentricity",
        "Peroxisomes_AreaShape_Extent",
        "Peroxisomes_Intensity_IntegratedIntensity_GFP"
    ]

    # Read all plate databases and get cell info + qc feature columns, combine
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]
    qc_dfs = []

    for db_path in databases:
        conn = sqlite3.connect(db_path)

        plate_qc_df = pl.read_database(
            query=f"""
                    SELECT 
                        Per_Peroxisomes.Cell_ID, 
                        Per_Peroxisomes.Peroxisomes_AreaShape_Center_X AS Center_X,
                        Per_Peroxisomes.Peroxisomes_AreaShape_Center_Y AS Center_Y,
                        Peroxisomes_Number_Object_Number, 
                        Peroxisomes_Intensity_IntegratedIntensity_GFP / Peroxisomes_AreaShape_Area AS IInt_Norm,
                        {', '.join(qc_features)}
                    FROM Per_Peroxisomes
                    JOIN (SELECT Cell_ID, Cell_AreaShape_Area FROM Per_Cell) pc
                        ON Per_Peroxisomes.Cell_ID = pc.Cell_ID;
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
        .drop(["Center_X", "Center_Y"]) # interested in peroxisome centers, not cell
        .join(qc_df, on=["Cell_ID"])
    )
    qc_df.write_csv(file=f"{qc_directory}/raw_peroxisomes_qc_features.csv")

    return qc_df


if __name__ == '__main__':
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    qc_df_raw = create_qc_raw_df(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)

    # I have to create each histogram individually rather than as a matrix because the long df generated in feature_distributions_matrix
    # gets so massive that it crashes the script.
    for feature in ["Peroxisomes_AreaShape_Area", "Peroxisomes_AreaShape_Perimeter",
                    "Peroxisomes_AreaShape_Eccentricity", "Peroxisomes_AreaShape_Extent",
                    "Peroxisomes_Intensity_IntegratedIntensity_GFP", "IInt_Norm"]:
        features_to_plot = qc_df_raw.select([feature])

        feature_distributions_matrix(
            qc_features=features_to_plot,
            qc_directory=args.qc_directory,
            output_figure_name=f"{feature}_distributions")

    print("Complete.")
