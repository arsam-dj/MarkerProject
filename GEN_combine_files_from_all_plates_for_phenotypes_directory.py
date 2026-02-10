import argparse
import os
import polars as pl
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', default='', help='Path to directory with phenotype data.')

args = parser.parse_args()


def combine_files(directory, file_suffix, output_name="", save_file=True):
    """
    Given a path to directory with files for each plate, combines them into one single file.

    Args:
        directory (str): path to directory with files to be combined
        suffix (str): file name suffix for all files to be combined
        output_name (str, Optional): what to name output combined file
        save_file (bool, Optional): if True, saves the combined file without returning it. Otherwise, returns it without saving
    """
    combined_file = pl.read_csv(f"{directory}/{file_suffix}.csv")

    if save_file:
        combined_file.write_csv(f"{directory}/{output_name}.csv")

    else:
        return combined_file

    
if __name__ == '__main__':

    # Merge aggregated cell outlier and penetrance files
    combine_files(
        directory=str(next(Path(args.directory).rglob("aggregated_cell_outlier_data"))),
        file_suffix="*_aggregated_cell_outlier_data",
        output_name="all_aggregated_cell_outlier_data"
    )

    combine_files(
        directory=str(next(Path(args.directory).rglob("aggregated_penetrance_data"))),
        file_suffix="*_aggregated_penetrance_data",
        output_name="all_aggregated_penetrance_data"
    )

    # Merge cell counts
    cell_count_dirs = [str(p) for p in Path(args.directory).rglob("cell_counts")]
    for cell_count_dir in cell_count_dirs:
        dma_df = combine_files(
            directory=cell_count_dir,
            file_suffix=f"DMA_*",
            save_file=False
        )

        tsa_df = combine_files(
            directory=cell_count_dir,
            file_suffix=f"TSA_*",
            save_file=False
        )

        tsa_df = (
            tsa_df
            .rename(
                {"Total_TS1_Cells": "Total_R1_Cells",
                 "Total_TS1_Outlier_Cells": "Total_R1_Outlier_Cells",
                 "Total_TS2_Cells": "Total_R2_Cells",
                 "Total_TS2_Outlier_Cells": "Total_R2_Outlier_Cells",
                 "Total_TS3_Cells": "Total_R3_Cells",
                 "Total_TS3_Outlier_Cells": "Total_R3_Outlier_Cells",
                 "TS1_G1_Cells": "R1_G1_Cells",
                 "TS1_G1_Outlier_Cells": "R1_G1_Outlier_Cells",
                 "TS2_G1_Cells": "R2_G1_Cells",
                 "TS2_G1_Outlier_Cells": "R2_G1_Outlier_Cells",
                 "TS3_G1_Cells": "R3_G1_Cells",
                 "TS3_G1_Outlier_Cells": "R3_G1_Outlier_Cells",
                 "TS1_SG2_Cells": "R1_SG2_Cells",
                 "TS1_SG2_Outlier_Cells": "R1_SG2_Outlier_Cells",
                 "TS2_SG2_Cells": "R2_SG2_Cells",
                 "TS2_SG2_Outlier_Cells": "R2_SG2_Outlier_Cells",
                 "TS3_SG2_Cells": "R3_SG2_Cells",
                 "TS3_SG2_Outlier_Cells": "R3_SG2_Outlier_Cells",
                 "TS1_MAT_Cells": "R1_MAT_Cells",
                 "TS1_MAT_Outlier_Cells": "R1_MAT_Outlier_Cells",
                 "TS2_MAT_Cells": "R2_MAT_Cells",
                 "TS2_MAT_Outlier_Cells": "R2_MAT_Outlier_Cells",
                 "TS3_MAT_Cells": "R3_MAT_Cells",
                 "TS3_MAT_Outlier_Cells": "R3_MAT_Outlier_Cells",
                 }
            )
        )

        (
            pl
            .concat([dma_df, tsa_df], how="vertical")
            .write_csv(f"{cell_count_dir}/all_strain_cell_counts.csv")
        )

    # Merge outlier cells
    outlier_cells_dirs = [str(p) for p in Path(args.directory).rglob("outlier_cells")]
    for outlier_cells_dir in outlier_cells_dirs:
        combine_files(
            directory=outlier_cells_dir,
            file_suffix="*_outlier_cells",
            output_name="all_strain_outlier_cells"
        )

    # Merge penetrances
    penetrance_dirs = [str(p) for p in Path(args.directory).rglob("penetrances")]
    for penetrance_dir in penetrance_dirs:
        dma_df = combine_files(
            directory=penetrance_dir,
            file_suffix=f"DMA_*",
            save_file=False
        )

        tsa_df = combine_files(
            directory=penetrance_dir,
            file_suffix=f"TSA_*",
            save_file=False
        )

        tsa_df = (
            tsa_df
            .rename(
                {"Overall_TS1_Penetrance": "Overall_R1_Penetrance",
                 "Overall_TS2_Penetrance": "Overall_R2_Penetrance",
                 "Overall_TS3_Penetrance": "Overall_R3_Penetrance",

                 "TS1_G1_Penetrance": "R1_G1_Penetrance",
                 "TS2_G1_Penetrance": "R2_G1_Penetrance",
                 "TS3_G1_Penetrance": "R3_G1_Penetrance",

                 "TS1_SG2_Penetrance": "R1_SG2_Penetrance",
                 "TS2_SG2_Penetrance": "R2_SG2_Penetrance",
                 "TS3_SG2_Penetrance": "R3_SG2_Penetrance",

                 "TS1_MAT_Penetrance": "R1_MAT_Penetrance",
                 "TS2_MAT_Penetrance": "R2_MAT_Penetrance",
                 "TS3_MAT_Penetrance": "R3_MAT_Penetrance",
                 }
            )
        )

        (
            pl
            .concat([dma_df, tsa_df], how="vertical")
            .write_csv(f"{penetrance_dir}/all_strain_penetrances.csv")
        )

    # Merge strain hits
    strain_hits_dirs = [str(p) for p in Path(args.directory).rglob("strain_hits")]
    for strain_hits_dir in strain_hits_dirs:
        combine_files(
            directory=strain_hits_dir,
            file_suffix="*_hit_strains",
            output_name="all_hit_strains"
        )

        combine_files(
            directory=strain_hits_dir,
            file_suffix="*_thresholds_used_for_getting_strain_hits",
            output_name="all_thresholds_used_for_getting_strain_hits"
        )

    print("Complete.")
