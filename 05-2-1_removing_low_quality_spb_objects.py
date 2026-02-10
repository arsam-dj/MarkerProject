import argparse
import os
import pandas as pd
import polars as pl
import sqlite3

from GEN_quality_check_functions import delete_problematic_compartment_masks

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-q', '--qc_directory', default='', help='Path to directory to write quality check files to.')
parser.add_argument('-c', '--qc_compartment_features', default='', help='Path to file with Cell_IDs and scaled QC features.')
parser.add_argument('-p', '--plate', default='', help='Number for identifying plate being processed.')
parser.add_argument('-x', '--delete_all_comps', default='False', help='Specify if all other compartments in a cell with a single problematic mask should also be deleted (True). False by default.')

args = parser.parse_args()


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    # Load SPB features and get problematic SPB masks
    spb_to_remove = (
        pl
        .read_csv(args.qc_compartment_features)
        .filter(
            (pl.col('SPBs_AreaShape_Eccentricity') >= 0.95) |
            (pl.col('SPBs_AreaShape_Perimeter') >= 75) |
            (pl.col('SPBs_Distance_Minimum_Cell') <= 4) |
            (pl.col('SPB_Area_Over_Cell_Area') >= 0.15) |
            (pl.col('SPBs_AreaShape_MajorAxisLength') >= 25) |
            (pl.col('SPB_MajAL_Over_MinAL') >= 4)
        )
        .select(["Cell_ID", "SPBs_Number_Object_Number"])
    )

    # 1) save mean and median proportion of compartments removed for each cell
    # 2) percentage of cells that got filtered out for each strain
    delete_problematic_compartment_masks(
        db_path=args.database_path,
        filtered_comps=spb_to_remove,
        comp_name="SPBs",
        output_dir=args.qc_directory,
        plate=args.plate,
        delete_all_comp_masks=args.delete_all_comps,
        replace_comp_num_with=-1,
        save_csv="True")


    print("Complete.")