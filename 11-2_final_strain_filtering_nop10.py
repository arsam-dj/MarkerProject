import argparse
import os
import pandas as pd
import polars as pl
import polars.selectors as cs
import sqlite3
from functools import reduce
from itertools import chain
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--phenotype_directory', default='', help='Path to directory with all phenotype information.')
parser.add_argument('-c', '--cell_counts', default='', help='Path to spreadsheet with R1, R2, and R3 cell counts for every strain in a screen.')
parser.add_argument('-p', '--penetrances', default='', help='Path to spreadsheet with overall and per-CC penetraces for every phenotype for every strain in a screen.')
parser.add_argument('-m', '--min_num_cells', default=51, help='Specify minimum cell count for filtering; defaults to 51.')
parser.add_argument('-s', '--screen', default='', help='Name of screen for filtering strain files.')
parser.add_argument('-O', '--outlier_cells', default='', help='Path to spreadsheet with all outlier cells and labelled defects.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save spreadsheets with replicate-replicate distances.')

args = parser.parse_args()

# Sheet A: everything
# Sheet B: no WT, no ORFs near marker/HTA2/CAN1/LYP1, no dubious ORFs, no strains with info from only one rep
# Sheet C: above + no cases with < 51 total cell count + no cases with < 20 cells in any of their reps + no cases with too much variation in rep-rep pens
# Sheet D: above + no nonsig strains + min. penetrance is greater than 95th perc. of wt strains


def get_tsv2_mapping_sheet():
    """
    Helper function for loading the TSv2 mapping sheet (reduce clutter).
    
    Returns:
        pl.DataFrame with strain coordinates.
    """
    mapping_coords = (
        pl.read_csv("/home/alex/alex_files/markerproject_redux/array_mapping_files/TS-Array-Morphology-v2-384.csv")
        .select(["Plate", "Row", "Column", "Strain_ID"])
        .filter(pl.col("Plate").is_in([1, 2, 3]))
        .with_columns(
            pl.col("Plate")
            .cast(pl.String)
            .str.zfill(2)
            .alias("Plate")
        )
        .unique()
    )

    mapping_coords_26C = (
        mapping_coords
        .with_columns(
            (pl.col("Strain_ID") + "-26C")
            .alias("Strain_ID")
            )
    )

    mapping_coords_37C = (
        mapping_coords
        .with_columns(
            (pl.col("Strain_ID") + "-37C")
            .alias("Strain_ID")
            )
    )

    mapping_coords = pl.concat([mapping_coords_26C, mapping_coords_37C])
    
    return mapping_coords
    

def recalculate_penetrances_for_nop10_strains(outlier_cells_path, penetrance_path):
    """
    The rest of this script requires joining the Distance spreadsheet with the aggregate Penetrance spreadsheet based on 
    strain coordinates. However, in the previous script some TS1 strains had their coordinates artificially changed so
    they can't be merged with the Penetrance spreadsheet containing real coordinates. Therefore, penetrances need to be
    re-calculated for TS strains and given the same artifical coordinates before being joined.
    
    Args:
        outlier_cells_path (str): path to spreadsheet with all outlier cell IDs and their respective defect(s)
        penetrance_path (str): path to spreadsheet with aggregated penetrance data produced during OD
        
    Returns:
        pl.DataFrame with re-calculated penetrances and artificial strain coordinates
    """
    
    tsa_databases = ["Nop10_TSA_26C_Plate01.db", "Nop10_TSA_26C_Plate02.db", "Nop10_TSA_26C_Plate03.db", "Nop10_TSA_26C_Plate04.db", "Nop10_TSA_26C_Plate05.db",
                         "Nop10_TSA_37C_Plate01.db", "Nop10_TSA_37C_Plate02.db", "Nop10_TSA_37C_Plate03.db", "Nop10_TSA_37C_Plate04.db", "Nop10_TSA_37C_Plate05.db"] 
    query = """
            SELECT
                Plate,
                Replicate,
                ORF,
                Name,
                Strain_ID, 
                Row,
                Column,
                COUNT() AS Total_Num_Cells,
                SUM(CASE WHEN Predicted_Label = 'G1' THEN 1 ELSE 0 END) AS Num_G1_Cells,
                SUM(CASE WHEN Predicted_Label = 'SG2' THEN 1 ELSE 0 END) AS Num_SG2_Cells,
                SUM(CASE WHEN Predicted_Label = 'MAT' THEN 1 ELSE 0 END) AS Num_MAT_Cells
            FROM Per_Cell
            GROUP BY Plate, Replicate, ORF, Name, Strain_ID, Row, Column;
            """

    # First, get total and per-CC cell counts from databases
    mapping_coords = get_tsv2_mapping_sheet()

    total_and_per_cc_cell_counts = []
    for tsa_database in tsa_databases:
        conn = sqlite3.connect(f"/home/alex/alex_files/markerproject_redux/screens/Nop10/{tsa_database}")
        database_df = (
            pl
            .read_database(query=query, connection=conn)
            .with_columns(
                (
                    pl
                    .when(pl.col("Name").is_null())
                    .then(pl.lit(""))
                    .otherwise(pl.col("Name"))
                    ).alias("Name")
                )
            )
        total_and_per_cc_cell_counts.append(database_df)
        conn.close()
    
    total_and_per_cc_cell_counts = (
        pl
        .concat(total_and_per_cc_cell_counts, how="vertical")
        .drop(["Plate", "Row", "Column"])
        .join(mapping_coords, on="Strain_ID", how="left")
        .with_columns(pl.col("Plate").cast(pl.String).str.zfill(2))
        .group_by(["Plate", "Row", "Column", "ORF", "Name", "Strain_ID"])
        .agg(
            pl.col("Total_Num_Cells").sum(),
            pl.col("Num_G1_Cells").sum(),
            pl.col("Num_SG2_Cells").sum(),
            pl.col("Num_MAT_Cells").sum()
            )
        )
    
    # Then load the penetrance data (for DMA strains only)
    penetrance_df_dma = (
    pl
    .read_csv(penetrance_path)
    .filter(pl.col("Strain_ID").str.contains("dma"))
    .with_columns(
        (pl
        .when(pl.col("Name").is_null())
        .then(pl.lit(""))
        .otherwise(pl.col("Name"))).alias("Name"),
        cs.exclude(pl.String).cast(pl.Float64),
        )
    .with_columns(
        pl.col("Plate").cast(pl.Int64).cast(pl.String).str.zfill(2),
        pl.col("Row").cast(pl.Int64),
        pl.col("Column").cast(pl.Int64)
        )
    )

    # Now load outlier cell data (for TSA strains only)
    outlier_cells = (
    pl
    .read_csv(outlier_cells_path)
    .with_columns(
        (
            pl
            .when(pl.col("Name").is_null())
            .then(pl.lit(""))
            .otherwise(pl.col("Name"))
            ).alias("Name")
    )
    .drop(["Plate", "Row", "Column"])
    .join(mapping_coords, on="Strain_ID", how="left")
    .filter(pl.col("Replicate").is_in(["TS1", "TS2", "TS3"])))
    
    unique_defects = set(chain.from_iterable([defect.split(" | ") for defect in outlier_cells["Cell_Phenotype"].unique()]))
    
    # Finally recalculate penetrances by using artificial coordinates
    agg_outliers_overall = (
    outlier_cells
    .group_by(["Plate", "Row", "Column", "ORF", "Name", "Strain_ID"])
    .agg(
        pl.len().alias("Total_Outliers"),
        (pl.col("Predicted_Label") == "G1").sum().alias("Num_G1_Outliers"),
        (pl.col("Predicted_Label") == "SG2").sum().alias("Num_SG2_Outliers"),
        (pl.col("Predicted_Label") == "MAT").sum().alias("Num_MAT_Outliers"),
    )
    .with_columns(pl.col("Plate").cast(pl.String).str.zfill(2))
    .join(total_and_per_cc_cell_counts, on=["Plate", "Row", "Column", "ORF", "Name", "Strain_ID"], how="left")
    .with_columns(
        (pl.col("Total_Outliers") / pl.col("Total_Num_Cells")).alias("Overall_Penetrance"),
        (pl.col("Num_G1_Outliers") / pl.col("Num_G1_Cells")).alias("Overall_G1_Penetrance"),
        (pl.col("Num_SG2_Outliers") / pl.col("Num_SG2_Cells")).alias("Overall_SG2_Penetrance"),
        (pl.col("Num_MAT_Outliers") / pl.col("Num_MAT_Cells")).alias("Overall_MAT_Penetrance"),
        )
    .select(["Plate", "ORF", "Name", "Strain_ID", "Row", "Column", "Overall_Penetrance", "Overall_G1_Penetrance", "Overall_SG2_Penetrance", "Overall_MAT_Penetrance"])
    )
    
    agg_outliers_defect_dfs = []
    for unique_defect in unique_defects:
        agg_outliers_defect = (
            outlier_cells
            .filter(pl.col("Cell_Phenotype").str.contains(unique_defect))
            .group_by(["Plate", "Row", "Column", "ORF", "Name", "Strain_ID"])
            .agg(
                pl.len().alias("Total_Outliers"),
                (pl.col("Predicted_Label") == "G1").sum().alias("Num_G1_Outliers"),
                (pl.col("Predicted_Label") == "SG2").sum().alias("Num_SG2_Outliers"),
                (pl.col("Predicted_Label") == "MAT").sum().alias("Num_MAT_Outliers"),
            )
            .with_columns(pl.col("Plate").cast(pl.String).str.zfill(2))
            .join(total_and_per_cc_cell_counts, on=["Plate", "Row", "Column", "ORF", "Name", "Strain_ID"], how="left")
            .with_columns(
                (pl.col("Total_Outliers") / pl.col("Total_Num_Cells")).alias(f"Overall_{unique_defect}_Penetrance"),
                (pl.col("Num_G1_Outliers") / pl.col("Num_G1_Cells")).alias(f"{unique_defect}_G1_Penetrance"),
                (pl.col("Num_SG2_Outliers") / pl.col("Num_SG2_Cells")).alias(f"{unique_defect}_SG2_Penetrance"),
                (pl.col("Num_MAT_Outliers") / pl.col("Num_MAT_Cells")).alias(f"{unique_defect}_MAT_Penetrance"),
            )
            .select(["Plate", "ORF", "Name", "Strain_ID", "Row", "Column", f"Overall_{unique_defect}_Penetrance", f"{unique_defect}_G1_Penetrance", f"{unique_defect}_SG2_Penetrance", f"{unique_defect}_MAT_Penetrance"])
        )
        agg_outliers_defect_dfs.append(agg_outliers_defect)
    agg_outliers_defect_dfs.append(agg_outliers_overall)

    # Merge the original DMA penetrances with the re-calculated TSA penetrances and return
    penetrance_df_tsa = reduce(
        lambda left, right: left.join(
            right,
            on=["Plate", "ORF", "Name", "Strain_ID", "Row", "Column"],
            how="full",
            coalesce=True,
        ),
        agg_outliers_defect_dfs,
    )

    penetrance_df_tsa = penetrance_df_tsa.select(penetrance_df_dma.columns)

    final_penetrance_df = pl.concat([penetrance_df_dma, penetrance_df_tsa], how="vertical")

    return final_penetrance_df


def combine_cell_counts_with_penetrances(per_rep_cell_count_path, penetrance_df):
    """
    Merges per-phenotype penetrances with total cell counts per strain.

    Args:
        per_rep_cell_count_path (str): path to spreadsheet with R1, R2, and R3 cell counts for every strain in a screen
        penetrance_df (pl.DataFrame): dataframe with overall and per-CC penetraces for every phenotype for every strain in a screen

    Returns:
        pl.DataFrame with merged penetrance and cell count info for every strain in a screen
    """

    penetrance_df = (
        penetrance_df
        .with_columns(
            pl.col("Plate").cast(pl.Int64)
        )
    )
    merged_df = (
        pl
        .read_csv(per_rep_cell_count_path)
        .select(["Plate", "Row", "Column", "ORF", "Name", "Strain_ID",
                 "Penetrance_R1", "Penetrance_R2", "Penetrance_R3",
                 "Total_Num_Cells_R1", "Total_Num_Cells_R2", "Total_Num_Cells_R3",
                 "Distance_R1-R2", "Distance_R1-R3", "Distance_R2-R3"])
        .with_columns(
            (
                pl
                .when(pl.col("Name").is_null())
                .then(pl.lit(""))
                .otherwise(pl.col("Name"))
            ).alias("Name"),
            pl.sum_horizontal(["Total_Num_Cells_R1", "Total_Num_Cells_R2", "Total_Num_Cells_R3"]).alias("Total_Num_Cells"),
            pl.sum_horizontal(
                pl.col(["Distance_R1-R2", "Distance_R1-R3", "Distance_R2-R3"]).is_null()
            ).alias("Num_Null_Dists")
        )
        .join(penetrance_df, on=["Plate", "Row", "Column", "ORF", "Name", "Strain_ID"])
    )

    return merged_df


def get_all_strain_hits(phenotype_dir):
    paths = Path(phenotype_dir).rglob("all_hit_strains.csv")
    good_paths = [pl.read_csv(p) for p in paths if pl.read_csv(p).shape[0] > 0] # only keep dataframes that have at least one row

    sig_strains = (
        pl
        .concat(
            good_paths, how="vertical"
        )
        .filter(~pl.col("ORF").is_in(["YOR202W", "YMR271C"]))
        .select("Strain_ID")
        .unique()
    )

    return list(sig_strains["Strain_ID"])


if __name__ == '__main__':

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    wildtype_strains = (
        pl
        .read_csv("/home/alex/alex_files/markerproject_redux/strain_filtering/filtered_strains/all_wt_strains.csv")
        .filter(pl.col("Marker") == args.screen)
    )

    orfs_near_marker = (
        pl
        .read_csv("/home/alex/alex_files/markerproject_redux/strain_filtering/filtered_strains/orfs_near_markers.csv")
        .filter(pl.col("Marker").is_in([args.screen, "Hta2", "Can1", "Lyp1"]))
    )

    all_orfs_to_keep = (
        pl
        .read_csv("/home/alex/alex_files/markerproject_redux/strain_filtering/filtered_strains/verified_and_uncharacterized_ORFs.csv")
    )
    
    penetrance_df = recalculate_penetrances_for_nop10_strains(
        outlier_cells_path=args.outlier_cells, 
        penetrance_path=args.penetrances)

    # Sheet A: everything
    sheetA = combine_cell_counts_with_penetrances(
        per_rep_cell_count_path=args.cell_counts, penetrance_df=penetrance_df)

    # Sheet B: no WT, no ORFs near marker/HTA2/CAN1/LYP1, no dubious ORFs, no strains with info from only one rep
    sheetB = (
        sheetA
        .filter(
            ~pl.col("Strain_ID").is_in(wildtype_strains["Strain"]),
            ~pl.col("ORF").is_in(orfs_near_marker["ORF"]),
            pl.col("ORF").is_in(all_orfs_to_keep["ORF"]),
            pl.col("Num_Null_Dists") != 3
        )
    )


    # Sheet C: above + no cases with < set min. total cell count + no cases with < 20 cells in any of their reps + no cases
    # with too much variation in rep-rep pens (unless all replicates have penetrance >= 30)
    sheetC = (
        sheetB
        .with_columns(pl.concat_list("Distance_R1-R2", "Distance_R1-R3", "Distance_R2-R3").list.std().alias("Std_Distance"))
        .filter(
            (pl.col("Total_Num_Cells") >= int(args.min_num_cells)),
            (pl.col("Total_Num_Cells_R1") >= 20),
            (pl.col("Total_Num_Cells_R2") >= 20),
            (pl.col("Total_Num_Cells_R3") >= 20),
        )
    )

    strains_above_30pen = sheetC.filter(
        (pl.col("Penetrance_R1") >= 30) & (pl.col("Penetrance_R2") >= 30) & (pl.col("Penetrance_R3") >= 30))
    strains_with_high_rep_agreement = sheetC.filter((pl.col("Std_Distance")) <= 5)
    strains_to_keep = pl.concat([strains_above_30pen, strains_with_high_rep_agreement], how="vertical")

    sheetC = (
        sheetC
        .filter(pl.col("Strain_ID").is_in(strains_to_keep["Strain_ID"]))
        .drop("Std_Distance")
    )
    
    # Sheet D: above + min. penetrance is greater than 95th perc. of wt strains
    wt_strains = (
        sheetA
        .filter(
            pl.col("Strain_ID").is_in(wildtype_strains["Strain"]),
            pl.col("Num_Null_Dists") != 3
        )
        .with_columns(pl.concat_list("Distance_R1-R2", "Distance_R1-R3", "Distance_R2-R3").list.std().alias("Std_Distance"))
        .filter(
            (pl.col("Total_Num_Cells") >= int(args.min_num_cells)),
            (pl.col("Total_Num_Cells_R1") >= 20),
            (pl.col("Total_Num_Cells_R2") >= 20),
            (pl.col("Total_Num_Cells_R3") >= 20),
            (pl.col("Std_Distance") <= 5)
        )
    )

    sheetD = (
        sheetC
        .filter(
            (pl.col("Overall_Penetrance") >= wt_strains["Overall_Penetrance"].quantile(0.05))
        )
    )

    # SheetE: above + no strains not sig in any one defect
    sig_strains = get_all_strain_hits(args.phenotype_directory)
    sheetE = (
        sheetD
        .filter(
            (pl.col("Strain_ID").is_in(sig_strains)),
        )
    )

    # Export
    dropped_cols = ["Num_Null_Dists",
                    "Penetrance_R1", "Penetrance_R2", "Penetrance_R3",
                    "Total_Num_Cells_R1", "Total_Num_Cells_R2", "Total_Num_Cells_R3",
                    "Distance_R1-R2", "Distance_R1-R3", "Distance_R2-R3"]

    sheetA = sheetA.drop(dropped_cols)
    sheetB = sheetB.drop(dropped_cols)
    sheetC = sheetC.drop(dropped_cols)
    sheetD = sheetD.drop(dropped_cols)
    sheetE = sheetE.drop(dropped_cols)

    with pd.ExcelWriter(f"{args.output_directory}/{args.screen}_filtered_strains.xlsx", engine="openpyxl") as writer:
        sheetA.to_pandas().to_excel(writer, sheet_name="SheetA", index=False)
        sheetB.to_pandas().to_excel(writer, sheet_name="SheetB", index=False)
        sheetC.to_pandas().to_excel(writer, sheet_name="SheetC", index=False)
        sheetD.to_pandas().to_excel(writer, sheet_name="SheetD", index=False)
        sheetE.to_pandas().to_excel(writer, sheet_name="SheetE", index=False)

    print("Complete.")
