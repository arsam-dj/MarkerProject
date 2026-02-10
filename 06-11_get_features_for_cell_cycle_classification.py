import argparse
import os
import polars as pl
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_directory', default='', help='Path to directory with all databases.')
parser.add_argument('-c', '--cell_cycle_directory', default='', help='Directory for writing cell cycle output files.')
parser.add_argument('-x', '--cell_coordinates', default='', help='Path to file with all cell overlay coordinates.')

args = parser.parse_args()


### Function for getting cell and nuclear dataframes with features relevant for cell cycle classification
def get_nuclear_distances(db_conn):
    """
    For cells that have one nucleus, the major axis length of the nuclear compartment is the same as the one nucleus's
    major axis length. For cells that have two nuclei, the 'major axis length' is calculated as the distance between the
    midpoints of the two nuclei. This is to allow a Nuclear_MajAL / Cell_MajAL calculation for cell cycle classification.

    Args:
        db_conn (sqlite3.connection): connection to a database
    """
    one_nucleus = (
        pl
        .read_database(query="""SELECT
	                                Cell_ID,
	                                Nuclei_AreaShape_MajorAxisLength AS Nuclear_Distance
                                FROM Per_Nuclei
                                GROUP BY Cell_ID
                                HAVING COUNT(*) = 1""",
                       connection=db_conn))
    two_nuclei = (
        pl
        # First read cell coordinates for all nuclei coming from cells that only have two nuclear masks
        # ROW_NUMBER() is used for pivoting from long to wide
        .read_database(
            query=f"""
                    SELECT 
                        ROW_NUMBER() OVER (PARTITION BY Cell_ID) AS Nuc_Num,
                        Cell_ID,
                        Nuclei_AreaShape_Center_X AS Center_X, 
                        Nuclei_AreaShape_Center_Y AS Center_Y
                    FROM Per_Nuclei
                    WHERE Cell_ID IN (
                        SELECT Cell_ID
                        FROM Per_Cell
                        WHERE Cell_Children_Nuclei_Count = 2
                        );
                    """,
            connection=db_conn
            )
        # So far, there are two rows for each Cell_ID (one for each nucleus)
        # Here, we pivot from long to wide on Nuc_Num so that each Cell_ID now has one row
        # Each row will have Center_X_1, Center_X_2, Center_Y_1, and Center_Y_2 columns that will
        # make calculating distances easy
        .pivot(
        on=["Nuc_Num"],
        index=["Cell_ID"]
        )
        # Now create a new column that actually calculates the distance between two nuclei
        .with_columns(
            (
            ((pl.col("Center_X_2") - pl.col("Center_X_1"))**2 + (pl.col("Center_Y_2") - pl.col("Center_Y_1"))**2)**0.5)
            .alias("Nuclear_Distance")
            )
        # Select relevant columns
        .select(["Cell_ID", "Nuclear_Distance"])
    )

    more_than_two_nuclei = (
        pl
        .read_database(query="""SELECT
	                                Cell_ID,
	                                -1.0 AS Nuclear_Distance 
                                FROM Per_Nuclei
                                GROUP BY Cell_ID
                                HAVING COUNT(*) > 2""",
                       connection=db_conn))

    # Merge everything
    merged_dfs = (
        pl
        .concat(items=[one_nucleus, two_nuclei, more_than_two_nuclei], how="vertical")
        .unique()
    )

    return merged_dfs

def get_cell_cycle_dfs(database_directory, cell_features, nuclear_features, coords_file, output_directory):
    """
    Creates a combined cell+nuclear dataframe with cell and nuclear features to be used in cell cycle
    classification.

    Args:
        database_directory (str): path to directory with all plate databases
        cell_features (list(str)): list of feature names to get from cell database
        nuclear_features (list(str)): list of feature names to get from nuclear database
        coords_file (str): path to a file with each cell's x/y coordinates and image path
        output_directory (str): where to save file to

    """
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]
    cell_nuclear_feature_dfs = []

    # Get cell and nuclear features from each database
    for db_path in databases:
        conn = sqlite3.connect(db_path)
        cell_df1 = (
            pl
            .read_database(query=f"""SELECT 
                                        {', '.join(cell_features)},
                                        CAST(Cell_AreaShape_MajorAxisLength AS numeric) / Cell_AreaShape_MinorAxisLength AS Cell_MajorAL_Over_Minor_AL 
                                     FROM Per_Cell""",
                           connection=conn)
            .rename({"Cell_AreaShape_Center_X": "Center_X", "Cell_AreaShape_Center_Y": "Center_Y"})
        )
        cell_df2 = get_nuclear_distances(db_conn=conn)
        cell_df = (
            cell_df1
            .join(cell_df2, on=["Cell_ID"], how="left")
            .with_columns((pl.col('Nuclear_Distance') / pl.col('Cell_AreaShape_MajorAxisLength')).alias('NucDist_Over_CellMajAL'))
        )

        nuclear_df = pl.read_database(query=f"SELECT {', '.join(nuclear_features)} FROM Per_Nuclei", connection=conn)

        ### DELETE THIS ###
        dumbass_df = (
            pl
            .read_database(query=f"SELECT {', '.join(dumbass_features)} FROM Per_Nuclei", connection=conn)
            .group_by("Cell_ID")
            .agg(
                pl.col("Nuclei_AreaShape_Center_X").mean().alias("Cell_Mean_Nuclei_AreaShape_Center_X"),
                pl.col("Nuclei_AreaShape_Center_Y").mean().alias("Cell_Mean_Nuclei_AreaShape_Center_Y"),
                pl.col("Nuclei_AreaShape_Compactness").mean().alias("Cell_Mean_Nuclei_AreaShape_Compactness"),
                pl.col("Nuclei_AreaShape_Eccentricity").mean().alias("Cell_Mean_Nuclei_AreaShape_Eccentricity"),
                pl.col("Nuclei_AreaShape_FormFactor").mean().alias("Cell_Mean_Nuclei_AreaShape_FormFactor"),
                pl.col("Nuclei_AreaShape_MajorAxisLength").mean().alias("Cell_Mean_Nuclei_AreaShape_MajorAxisLength"),
                pl.col("Nuclei_AreaShape_MinorAxisLength").mean().alias("Cell_Mean_Nuclei_AreaShape_MinorAxisLength"),
                pl.col("Nuclei_AreaShape_Solidity").mean().alias("Cell_Mean_Nuclei_AreaShape_Solidity"),
                pl.col("Nuclei_Distance_Centroid_Cell").mean().alias("Cell_Mean_Nuclei_Distance_Centroid_Cell"),
            )
            .with_columns(
                (pl.col("Cell_Mean_Nuclei_AreaShape_MajorAxisLength") / pl.col("Cell_Mean_Nuclei_AreaShape_MinorAxisLength")).alias("Nucleus_MajorAL_Over_MinorAL")
            )
        )
        # Merge dataframes and calculate the ratio of cell and nuclear major axis lengths
        cell_nuclear_df = cell_df.join(nuclear_df, on="Cell_ID", how="left")

        ### DELETE THIS ###
        cell_nuclear_df = cell_nuclear_df.join(dumbass_df, on="Cell_ID", how="left")

        cell_nuclear_feature_dfs.append(cell_nuclear_df)

        conn.close()

    # Merge all cell_nuclear_dfs, add coordinate paths
    merged_df = pl.concat(cell_nuclear_feature_dfs, how="vertical")
    coords = (
        pl
        .read_csv(coords_file)
        .join(merged_df, on=["Cell_ID", "Center_X", "Center_Y"], how="right")
    )

    # Infer which string columns can be cast to float by trying a cast on the first non-null value
    float_candidates = []
    for col in coords.columns:
        if coords.schema[col] == pl.Utf8:
            # Try casting first non-null value
            sample = coords.select(pl.col(col).drop_nulls().limit(1)).to_series(0).item(0)
            try:
                float(sample)
                float_candidates.append(col)
            except (ValueError, TypeError):
                pass

    # Cast those columns to Float64 and save
    (
        coords
        .with_columns([pl.col(c).cast(pl.Float64) for c in float_candidates])
        .unique()
        .write_csv(f"{output_directory}/all_cell_and_nuclear_features_for_cell_cycle_classification.csv")
    )




if __name__ == '__main__':
    if not os.path.exists(args.cell_cycle_directory):
        os.makedirs(args.cell_cycle_directory)

    # Get cell-nuclear dataframe
    #cell_features = [
    #    "Replicate", "Cell_ID", "ORF", "Name", "Strain_ID",
    #    "Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y",
    #    "Cell_Mean_Nuclei_AreaShape_Center_X", "Cell_Mean_Nuclei_AreaShape_Center_Y",
    #    "Cell_AreaShape_Compactness", "Cell_AreaShape_Eccentricity",
    #    "Cell_AreaShape_FormFactor", "Cell_AreaShape_MajorAxisLength", "Cell_AreaShape_MinorAxisLength",
    #    "Cell_AreaShape_Solidity", "Cell_Children_Nuclei_Count",
    #    "Cell_Mean_Nuclei_AreaShape_Compactness", "Cell_Mean_Nuclei_AreaShape_Eccentricity",
    #    "Cell_Mean_Nuclei_AreaShape_FormFactor", "Cell_Mean_Nuclei_AreaShape_MajorAxisLength",
    #    "Cell_Mean_Nuclei_AreaShape_MinorAxisLength", "Cell_Mean_Nuclei_AreaShape_Solidity",
    #    "Cell_Mean_Nuclei_Distance_Centroid_Cell"
    #]

    cell_features = [
        "Replicate", "Cell_ID", "ORF", "Name", "Strain_ID",
        "Cell_AreaShape_Center_X", "Cell_AreaShape_Center_Y",
        #"Cell_Mean_Nuclei_AreaShape_Center_X", "Cell_Mean_Nuclei_AreaShape_Center_Y",
        "Cell_AreaShape_Compactness", "Cell_AreaShape_Eccentricity",
        "Cell_AreaShape_FormFactor", "Cell_AreaShape_MajorAxisLength", "Cell_AreaShape_MinorAxisLength",
        "Cell_AreaShape_Solidity", "Cell_Children_Nuclei_Count"
    ]

    nuclear_features = [
        "Cell_ID",
        "Nuclei_AreaShape_Compactness", "Nuclei_AreaShape_Eccentricity", "Nuclei_AreaShape_FormFactor",
        "Nuclei_AreaShape_MajorAxisLength", "Nuclei_AreaShape_MinorAxisLength", "Nuclei_AreaShape_Solidity"]

    ### DELETE THIS ###
    dumbass_features = ["Cell_ID", "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
                        "Nuclei_AreaShape_Compactness", "Nuclei_AreaShape_Eccentricity",
                        "Nuclei_AreaShape_FormFactor", "Nuclei_AreaShape_MajorAxisLength",
                        "Nuclei_AreaShape_MinorAxisLength", "Nuclei_AreaShape_Solidity",
                        "Nuclei_Distance_Centroid_Cell"]

    get_cell_cycle_dfs(
        database_directory=args.database_directory,
        cell_features=cell_features,
        nuclear_features=nuclear_features,
        coords_file=args.cell_coordinates,
        output_directory=args.cell_cycle_directory)

    print("Complete.")

