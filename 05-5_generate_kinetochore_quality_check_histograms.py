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
def create_qc_raw_df_kins(database_directory, coordinates_path, qc_directory):
    """
    Create a dataframe with kinetochores from all three reps, as well as QC features of interest. Includes the
    segmentation mask paths so they can be viewed with Single Cell Tool.

    Args:
        database_directory (str): path to directory with all plate databases
        coordinates_path (str): path to file with cell overlay coordinates
        qc_directory (str): path to directory to write qc files to
    """
    qc_features = [
        "Kinetochores_AreaShape_Area",
        "Kinetochores_AreaShape_Perimeter",
        "Kinetochores_AreaShape_Eccentricity",
        "Kinetochores_AreaShape_MajorAxisLength",
        "Kinetochores_AreaShape_Solidity",
        "Kinetochores_Intensity_IntegratedIntensity_GFP"
    ]

    # Read all plate databases and get cell info + qc feature columns, combine
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]
    qc_dfs = []

    for db_path in databases:
        conn = sqlite3.connect(db_path)

        plate_qc_df = pl.read_database(
            query=f"""
                    SELECT 
                        Per_Kinetochores.Cell_ID, 
                        Per_Kinetochores.Kinetochores_AreaShape_Center_X AS Center_X,
                        Per_Kinetochores.Kinetochores_AreaShape_Center_Y AS Center_Y,
                        Kinetochores_Number_Object_Number, 
                        Kinetochores_Intensity_IntegratedIntensity_GFP / Kinetochores_AreaShape_Area AS IInt_Norm,
                        {', '.join(qc_features)}
                    FROM Per_Kinetochores
                    JOIN (SELECT Cell_ID, Cell_AreaShape_Area FROM Per_Cell) pc
                        ON Per_Kinetochores.Cell_ID = pc.Cell_ID;
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
        .drop(["Center_X", "Center_Y"]) # interested in kinetochore centers, not cell
        .join(qc_df, on=["Cell_ID"])
    )
    qc_df.write_csv(file=f"{qc_directory}/raw_kinetochore_qc_features.csv")

    return qc_df


def create_qc_raw_df_cells(database_directory, coordinates_path, qc_directory):
    """
    Create a dataframe with cells from all three reps, as well as QC features of interest. Includes the
    segmentation mask paths so they can be viewed with Single Cell Tool.

    Args:
        database_directory (str): path to directory with all plate databases
        coordinates_path (str): path to file with cell overlay coordinates
        qc_directory (str): path to directory to write qc files to
    """
    # Read all plate databases and get cell info + qc feature columns, combine
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]
    qc_dfs = []

    for db_path in databases:
        conn = sqlite3.connect(db_path)

        plate_qc_df = pl.read_database(
            query=f"""
                    WITH kin_areas AS (
	                        SELECT
		                        Cell_ID,
		                        SUM(Kinetochores_AreaShape_Area) AS Total_Kinetochore_Area
	                        FROM Per_Kinetochores
	                        GROUP BY Cell_ID)

                    SELECT
                    	Per_Cell.Cell_ID,
                    	Cell_Children_Kinetochores_Count AS Num_Kinetochores,
                    	Total_Kinetochore_Area / Cell_AreaShape_Area AS Kinetochore_Coverage
                    FROM Per_Cell
                    JOIN kin_areas
                    ON Per_Cell.Cell_ID = kin_areas.Cell_ID;
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
    qc_df.write_csv(file=f"{qc_directory}/raw_kinetochore_qc_features_cell.csv")

    return qc_df


if __name__ == '__main__':
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    # Per-kinetochore mask QC
    qc_df_raw_kins = create_qc_raw_df_kins(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)

    for feature in ["Kinetochores_AreaShape_Area", "Kinetochores_AreaShape_Perimeter",
                    "Kinetochores_Intensity_IntegratedIntensity_GFP", "IInt_Norm",
                    "Kinetochores_AreaShape_Eccentricity", "Kinetochores_AreaShape_MajorAxisLength",
                    "Kinetochores_AreaShape_Solidity"]:
        features_to_plot = qc_df_raw_kins.select([feature])

        feature_distributions_matrix(
            qc_features=features_to_plot,
            qc_directory=args.qc_directory,
            output_figure_name=f"{feature}_distributions")


    # Per-cell QC
    qc_df_raw_cells = create_qc_raw_df_cells(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)

    for feature in ["Num_Kinetochores", "Kinetochore_Coverage"]:
        features_to_plot = qc_df_raw_cells.select([feature])

        feature_distributions_matrix(
            qc_features=features_to_plot,
            qc_directory=args.qc_directory,
            output_figure_name=f"{feature}_distributions")

    print("Complete.")
