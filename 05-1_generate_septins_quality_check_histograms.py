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
def create_qc_raw_df_septins(database_directory, coordinates_path, qc_directory):
    """
    Create a dataframe with septins from all three reps, as well as QC features of interest. Includes the
    segmentation mask paths so they can be viewed with Single Cell Tool.

    Args:
        database_directory (str): path to directory with all plate databases
        coordinates_path (str): path to file with cell overlay coordinates
        qc_directory (str): path to directory to write qc files to
    """
    qc_features = [
        "Septins_AreaShape_Area",
        "Septins_AreaShape_Perimeter",
        "Septins_AreaShape_Eccentricity",
        "Septins_AreaShape_Extent"
    ]

    # Read all plate databases and get cell info + qc feature columns, combine
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]
    qc_dfs = []

    for db_path in databases:
        conn = sqlite3.connect(db_path)

        plate_qc_df = pl.read_database(
            query=f"""
                    SELECT 
                        Cell_ID, 
                        Septins_AreaShape_Center_X AS Center_X,
                        Septins_AreaShape_Center_Y AS Center_Y,
                        Septins_Number_Object_Number,
                        {', '.join(qc_features)}
                    FROM Per_Septins;
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
        .drop(["Center_X", "Center_Y"]) # interested in septin centers, not cell
        .join(qc_df, on=["Cell_ID"])
    )
    qc_df.write_csv(file=f"{qc_directory}/raw_septin_qc_features.csv")

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
                    WITH septin_areas AS (
	                        SELECT
		                        Cell_ID,
		                        SUM(Septins_AreaShape_Area) AS Total_Septin_Area
	                        FROM Per_Septins
	                        GROUP BY Cell_ID)

                    SELECT
                    	Per_Cell.Cell_ID,
                    	Cell_Children_Septins_Count AS Num_Septins,
                    	Total_Septin_Area / Cell_AreaShape_Area AS Septin_Coverage_Cell
                    FROM Per_Cell
                    JOIN septin_areas
                    ON Per_Cell.Cell_ID = septin_areas.Cell_ID
                    JOIN Per_Nuclei
                    ON Per_Cell.Cell_ID = Per_Nuclei.Cell_ID;
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
    qc_df.write_csv(file=f"{qc_directory}/raw_septin_qc_features_cell.csv")

    return qc_df


def create_septin_nuclear_position_table(database_directory, coordinates_path, qc_directory):
    """
    Create a dataframe with cells that have two nuclei and a septin that is not positioned between the two nuclei.

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
                    WITH cell_nuc AS (
                        SELECT
                        	ROW_NUMBER() OVER (PARTITION BY Per_Nuclei.Cell_ID) AS Nuc_Num,
                        	Per_Nuclei.Cell_ID,
                        	Nuclei_AreaShape_Center_X AS Nuclei_Center_X,
                        	Nuclei_AreaShape_Center_Y AS Nuclei_Center_Y
                        FROM Per_Nuclei
                        JOIN (SELECT Cell_ID, Cell_Children_Nuclei_Count, Cell_Children_Septins_Count FROM Per_Cell) pc
                        	ON Per_Nuclei.Cell_ID = pc.Cell_ID
                        WHERE (Cell_Children_Nuclei_Count = 2) AND (Cell_Children_Septins_Count = 1)
                        ORDER BY Per_Nuclei.Cell_ID),

                    cell_nuc_pivot AS (
                    SELECT 
                    	Cell_ID,
                    	MAX(CASE WHEN Nuc_Num = 1 THEN Nuclei_Center_X END) AS Nuc_X1,
                    	MAX(CASE WHEN Nuc_Num = 1 THEN Nuclei_Center_Y END) AS Nuc_Y1,
                    	MAX(CASE WHEN Nuc_Num = 2 THEN Nuclei_Center_X END) AS Nuc_X2,
                    	MAX(CASE WHEN Nuc_Num = 2 THEN Nuclei_Center_Y END) AS Nuc_Y2
                    FROM cell_nuc
                    GROUP BY Cell_ID)

                    SELECT
                    	cell_nuc_pivot.Cell_ID,
                        Septins_Number_Object_Number,
                    	Nuc_X1,
                    	Nuc_Y1,
                    	Nuc_X2,
                    	Nuc_Y2,
                    	Septin_X,
                    	Septin_Y
                    FROM cell_nuc_pivot
                    JOIN (SELECT Cell_ID, Septins_Number_Object_Number, Septins_AreaShape_Center_X AS Septin_X, Septins_AreaShape_Center_Y AS Septin_Y FROM Per_Septins) ps
                    	ON cell_nuc_pivot.Cell_ID = ps.Cell_ID
                    WHERE 
                        (Septin_X NOT BETWEEN MIN(Nuc_X1, Nuc_X2) AND MAX(Nuc_X1, Nuc_X2)) 
                    AND (Septin_Y NOT BETWEEN MIN(Nuc_Y1, Nuc_Y2) AND MAX(Nuc_Y1, Nuc_Y2));
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
    qc_df.write_csv(file=f"{qc_directory}/nuclear_septin_coords.csv")

    return qc_df


if __name__ == '__main__':
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    # Per-Septin mask QC
    qc_df_raw_septins = create_qc_raw_df_septins(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)

    for feature in ["Septins_AreaShape_Area", "Septins_AreaShape_Perimeter",
                    "Septins_AreaShape_Eccentricity", "Septins_AreaShape_Extent"]:
        features_to_plot = qc_df_raw_septins.select([feature])

        feature_distributions_matrix(
            qc_features=features_to_plot,
            qc_directory=args.qc_directory,
            output_figure_name=f"{feature}_distributions")


    # Per-cell QC
    qc_df_raw_cells = create_qc_raw_df_cells(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)

    for feature in ["Num_Septins", "Septin_Coverage_Cell"]:
        features_to_plot = qc_df_raw_cells.select([feature])

        feature_distributions_matrix(
            qc_features=features_to_plot,
            qc_directory=args.qc_directory,
            output_figure_name=f"{feature}_distributions")


    # Table with septins not between two nuclei
    create_septin_nuclear_position_table(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)
    
    print("Complete.")
