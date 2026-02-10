import argparse
import os
import polars as pl
import sqlite3
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cell_outliers', default='', help='Path to csv file with all cell phenotype outliers.')
parser.add_argument('-s', '--subcellular_outliers', default='', help='Path to csv file with all subcellular compartment outliers.')
parser.add_argument('-o', '--output_dir', default='', help='Where to save output file with combined penetrances.')
parser.add_argument('-d', '--database_dir', default='', help='Path to directory with all screen databases.')

args = parser.parse_args()


def get_total_cell_counts(database_dir):
    """
    Creates a table with the total cell count for every strain across all databases.

    Args:
        database_dir (str): path to directory where all databases are kept.

    Returns:
        pl.Dataframe with strain information and total cell count for all strains
    """

    databases = os.listdir(database_dir)
    cell_counts_dfs = []
    for database in databases:
        conn = sqlite3.connect(f"{database_dir}/{database}")
        cell_counts = (
            pl
            .read_database(
                query="""
                        SELECT
	                        ORF,
	                        Name,
	                        Strain_ID,
	                        COUNT(Cell_ID) AS Total_Cells
                        FROM
                        	Per_Cell
                        GROUP BY ORF, Name, Strain_ID
                      """,
                connection=conn
            )
            .with_columns(
                (
                    pl
                    .when(pl.col("Name").is_null())
                    .then(pl.lit(""))
                    .otherwise(pl.col("Name"))
                ).alias("Name")
            )
        )
        conn.close()

        cell_counts_dfs.append(cell_counts)

    combined_cell_counts = (
        pl
        .concat(cell_counts_dfs, how="vertical")
        .group_by(["ORF", "Name", "Strain_ID"])
        .agg(pl.col("Total_Cells").sum().alias("Total_Cells")) # Combines all wildtypes together
    )

    return combined_cell_counts


def combine_penetrances(cell_outliers_path, subcellular_outliers_path, total_cell_counts, output_dir):
    """
    Given whole-cell and subcellular compartment penetrances, combines them and calculates a final overall penetrance

    Args:
        cell_outliers_path (str): path to csv file with all cell phenotype outliers
        subcellular_outliers_path (str): path to csv file with all subcellular outliers
        total_cell_counts (pl.DataFrame): dataframe with total cell counts for all strains
        output_dir (str): where to save output file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load outlier dataframes
    cell_outliers = (
        pl
        .read_csv(cell_outliers_path)
        .select(["ORF", "Name", "Strain_ID", "Cell_ID"])
        .with_columns(
            (
                pl
                .when(pl.col("Name").is_null())
                .then(pl.lit(""))
                .otherwise(pl.col("Name"))
            ).alias("Name")
        )
    )

    subcellular_outliers = (
        pl
        .read_csv(subcellular_outliers_path)
        .select(["ORF", "Name", "Strain_ID", "Cell_ID"])
        .with_columns(
            (
                pl
                .when(pl.col("Name").is_null())
                .then(pl.lit(""))
                .otherwise(pl.col("Name"))
            ).alias("Name")
        )
    )

    all_outliers = (
        pl
        .concat([cell_outliers, subcellular_outliers], how="vertical")
        .unique()
    )
    
    # Calculate cell, subcellular, and combined penetrances
    cell_pens = (
        cell_outliers
        .group_by(["ORF", "Name", "Strain_ID"])
        .len(name="Cell_Outliers")
        .join(total_cell_counts, on=["ORF", "Name", "Strain_ID"], how="left")
        .with_columns(
            (pl.col("Cell_Outliers") / pl.col("Total_Cells") * 100)
            .alias("Cell_Penetrance")
        )
        .select(["ORF", "Name", "Strain_ID", "Cell_Penetrance"])
    )

    subcell_pens = (
        subcellular_outliers
        .group_by(["ORF", "Name", "Strain_ID"])
        .len(name="Subcellular_Outliers")
        .join(total_cell_counts, on=["ORF", "Name", "Strain_ID"], how="left")
        .with_columns(
            (pl.col("Subcellular_Outliers") / pl.col("Total_Cells") * 100)
            .alias("Subcellular_Penetrance")
        )
        .select(["ORF", "Name", "Strain_ID", "Subcellular_Penetrance"])
    )

    overall_pens = (
        all_outliers
        .group_by(["ORF", "Name", "Strain_ID"])
        .len(name="All_Outliers")
        .join(total_cell_counts, on=["ORF", "Name", "Strain_ID"], how="left")
        .with_columns(
            (pl.col("All_Outliers") / pl.col("Total_Cells") * 100)
            .alias("Combined_Penetrance")
        )
        .select(["ORF", "Name", "Strain_ID", "Combined_Penetrance"])
    )

    # Join all penetrances
    combined_pens = reduce(
        lambda left, right: left.join(right, on=["ORF", "Name", "Strain_ID"], how="left"),
        [cell_pens, subcell_pens, overall_pens]
    )

    (
        combined_pens
        .sort(["Combined_Penetrance"], descending=True)
        .write_csv(f"{output_dir}/cell_subcell_combined_pens.csv"))
    

if __name__ == '__main__':

    total_cell_counts = get_total_cell_counts(args.database_dir)

    combine_penetrances(
        cell_outliers_path=args.cell_outliers,
        subcellular_outliers_path=args.subcellular_outliers,
        total_cell_counts=total_cell_counts,
        output_dir=args.output_dir)

    print("Complete.")
