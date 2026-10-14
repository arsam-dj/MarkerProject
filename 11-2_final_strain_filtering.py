import argparse
import os
import pandas as pd
import polars as pl
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--phenotype_directory', default='', help='Path to directory with all phenotype information.')
parser.add_argument('-c', '--cell_counts', default='', help='Path to spreadsheet with R1, R2, and R3 cell counts for every strain in a screen.')
parser.add_argument('-p', '--penetrances', default='', help='Path to spreadsheet with overall and per-CC penetraces for every phenotype for every strain in a screen.')
parser.add_argument('-m', '--min_num_cells', default=51, help='Specify minimum cell count for filtering; defaults to 51.')
parser.add_argument('-s', '--screen', default='', help='Name of screen for filtering strain files.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save spreadsheets with replicate-replicate distances.')

args = parser.parse_args()

# Sheet A: everything
# Sheet B: no WT, no ORFs near marker/HTA2/CAN1/LYP1, no dubious ORFs, no strains with info from only one rep
# Sheet C: above + no cases with < 51 total cell count + no cases with < 20 cells in any of their reps + no cases with too much variation in rep-rep pens
# Sheet D: above + min. penetrance is greater than 95th perc. of wt strains
# Sheet E: above + no strains not sig in any one defect + no strains with low phenotype-specific penetrances

def combine_cell_counts_with_penetrances(per_rep_cell_count_path, penetrance_path):
    """
    Merges per-phenotype penetrances with total cell counts per strain.

    Args:
        per_rep_cell_count_path (str): path to spreadsheet with R1, R2, and R3 cell counts for every strain in a screen
        penetrance_path (str): path to spreadsheet with overall and per-CC penetraces for every phenotype for every strain in a screen

    Returns:
        pl.DataFrame with merged penetrance and cell count info for every strain in a screen
    """

    penetrances = (
        pl
        .read_csv(penetrance_path)
        .with_columns(
            (
                pl
                .when(pl.col("Name").is_null())
                .then(pl.lit(""))
                .otherwise(pl.col("Name"))
            ).alias("Name")
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
                pl.col(["Total_Num_Cells_R1", "Total_Num_Cells_R2", "Total_Num_Cells_R3"]) >= 20
            ).alias("Num_Reps_Over_20_Cells"),
            pl.sum_horizontal(
                pl.col(["Distance_R1-R2", "Distance_R1-R3", "Distance_R2-R3"]).is_null()
            ).alias("Num_Null_Dists"),
        )
        .join(penetrances, on=["Plate", "Row", "Column", "ORF", "Name", "Strain_ID"])
    )

    return merged_df


def get_all_strain_hits(phenotype_dir, sheetA):
    """
    Loads each phenotype's per-CC hit list, then for each hit it determines its overall phenotype penetrance using SheetA.
    If the overall phenotype penetrance is too low (<10%), then that hit is no longer considered a hit and thus removed.
    This removes strains that for example have >20% penetrance in MAT, but only 4% overall penetrance.
    """
    paths = Path(phenotype_dir).rglob("all_hit_strains.csv")
    sheetA = (
        sheetA
        .with_columns(
            (
                pl
                .when(pl.col("Name").is_null())
                .then(pl.lit(""))
                .otherwise(pl.col("Name"))
            ).alias("Name")
        )
    )
    paths_to_columns = pl.read_csv("/home/alex/alex_files/markerproject_redux/misc_files/sig_strain_paths_mapped_to_penetrance_columns.csv")

    for path in paths:
        phenotype_pens = paths_to_columns.filter(pl.col("Path") == str(path))["Phenotype_Penetrance"][0]
        sheetA_phenotype_pens = sheetA.select(["ORF", "Name", "Strain_ID", phenotype_pens])
        sig_strains = (
            pl
            .read_csv(path)
            .with_columns(
                (
                    pl
                    .when(pl.col("Name").is_null())
                    .then(pl.lit(""))
                    .otherwise(pl.col("Name"))
                ).alias("Name")
            )
            .select(["ORF", "Name", "Strain_ID"])
            .unique()
            .join(sheetA_phenotype_pens, on=["ORF", "Name", "Strain_ID"], how="left")
            .filter(pl.col(phenotype_pens) >= 0.1)
            .rename({phenotype_pens: "Penetrance"})
        )
        sig_strains.write_csv(f"{str.split(str(path), '/all_hit_strains.csv')[0]}/all_hit_strains_filtered.csv")

    paths_filtered = Path(phenotype_dir).rglob("all_hit_strains_filtered.csv")
    good_paths = [pl.read_csv(p) for p in paths_filtered if pl.read_csv(p).shape[0] > 0] # only keep dataframes that have at least one row
    all_sig_strains = (
        pl
        .concat(
            good_paths, how="vertical"
        )
        .filter(~pl.col("ORF").is_in(["YOR202W", "YMR271C"]))
        .select("Strain_ID")
        .unique()
    )

    return list(all_sig_strains["Strain_ID"])


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

    # Sheet A: everything
    sheetA = combine_cell_counts_with_penetrances(
        per_rep_cell_count_path=args.cell_counts, penetrance_path=args.penetrances)

    sheetA = (
        sheetA
        .with_columns(
            (pl.concat_list("Distance_R1-R2", "Distance_R1-R3", "Distance_R2-R3").list.std()).alias("Std_Rep_Distance")
        )
    )

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


    # Sheet C: above + no cases with < set min. total cell count + no cases with < 20 cells in 2-3 reps + no cases
    # with too much variation in rep-rep pens (unless all replicates have penetrance >= 30)
    sheetC = (
        sheetB
        .filter(
            (pl.col("Total_Num_Cells") >= int(args.min_num_cells)),
            (pl.col("Num_Reps_Over_20_Cells") >= 2)
            #(pl.col("Total_Num_Cells_R1") >= 20),
            #(pl.col("Total_Num_Cells_R2") >= 20),
            #(pl.col("Total_Num_Cells_R3") >= 20),
        )
    )

    strains_above_30pen = sheetC.filter(
        (pl.col("Penetrance_R1") >= 30) & (pl.col("Penetrance_R2") >= 30) & (pl.col("Penetrance_R3") >= 30))
    strains_with_high_rep_agreement = sheetC.filter(pl.col("Std_Rep_Distance") <= 10)
    strains_to_keep = pl.concat([strains_above_30pen, strains_with_high_rep_agreement], how="vertical")

    sheetC = (
        sheetC
        .filter(pl.col("Strain_ID").is_in(strains_to_keep["Strain_ID"]))
    )
    
    # Sheet D: above + min. penetrance is greater than 95th perc. of wt strains
    wt_strains = (
        sheetA
        .filter(
            pl.col("Strain_ID").is_in(wildtype_strains["Strain"]),
            pl.col("Num_Null_Dists") != 3
        )
        .with_columns(pl.concat_list("Distance_R1-R2", "Distance_R1-R3", "Distance_R2-R3").list.std().alias("Std_Rep_Distance"))
        .filter(
            (pl.col("Total_Num_Cells") >= int(args.min_num_cells)),
            (pl.col("Num_Reps_Over_20_Cells") >= 2),
            #(pl.col("Total_Num_Cells_R1") >= 20),
            #(pl.col("Total_Num_Cells_R2") >= 20),
            #(pl.col("Total_Num_Cells_R3") >= 20),
            (pl.col("Std_Rep_Distance") <= 10)
        )
    )

    sheetD = (
        sheetC
        .filter(
            (pl.col("Overall_Penetrance") >= wt_strains["Overall_Penetrance"].quantile(0.95))
        )
    )

    # Sheet E: above + no strains not sig in any one defect + no strains with low phenotype-specific penetrances
    sig_strains = get_all_strain_hits(args.phenotype_directory, sheetA)
    phenotype_pen_cols = [
        column for column in sheetD.columns if
        ("Overall" in column) and
        (("G1" not in column) and ("SG2" not in column) and ("MAT" not in column) and ("Overall_Penetrance" not in column))
    ]

    sheetE = (
        sheetD
        .filter(
            (pl.col("Strain_ID").is_in(sig_strains))
        )
        .filter(
            pl.any_horizontal(
                pl.col(phenotype_pen_cols) >= 0.1
            )
        )
    )

    # Export
    dropped_cols = ["Num_Null_Dists", "Num_Reps_Over_20_Cells",
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
