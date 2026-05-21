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
            .select(pl.all().name.replace(r"TS", "R")) # replace TS in column names with R
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
            .select(pl.all().name.replace(r"TS", "R"))
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
