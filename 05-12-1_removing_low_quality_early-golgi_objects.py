import argparse
import os
import pandas as pd
import polars as pl
import sqlite3

from GEN_quality_check_functions import delete_problematic_compartment_masks

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-q', '--qc_directory', default='', help='Path to directory to write quality check files to.')
parser.add_argument('-c', '--qc_compartment_features', default='', help='Path to file with Cell_IDs and per-compartment QC features.')
parser.add_argument('-C', '--qc_cell_features', default='', help='Path to file with Cell_IDs and per-cell QC features.')
parser.add_argument('-p', '--plate', default='', help='Number for identifying plate being processed.')
parser.add_argument('-x', '--delete_all_comps', default='False', help='Specify if all other compartments in a cell with a single problematic mask should also be deleted (True). False by default.')

args = parser.parse_args()


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    # Load early golgi features and get problematic early golgi masks
    eg_to_remove1 = (  # this file is on the EG level
        pl
        .read_csv(args.qc_compartment_features)
        .filter(
            (pl.col('EGs_Intensity_IntegratedIntensity_GFP') >= 2) |
            (pl.col('IInt_Norm') >= 0.008)
        )
        .select(["Cell_ID", "EGs_Number_Object_Number"])
    )

    eg_to_remove2 = ( # this file is on the cell level, so I need to get the EG numbers from database
        pl
        .read_csv(args.qc_cell_features)
        .filter(
            (pl.col('EG_Coverage') >= 0.8)
        )
        .select(["Cell_ID"])
        .to_series()
        .to_list()
    )

    conn = sqlite3.connect(args.database_path)
    cells = ",".join(f"'{item}'" for item in eg_to_remove2)
    query = f"""
            SELECT
                Cell_ID,
                EGs_Number_Object_Number
            FROM Per_EGs
            WHERE Cell_ID IN ({cells});"""

    eg_to_remove2 = (
        pl
        .read_database(query=query, connection=conn)
    )
    conn.close()

    eg_to_remove = pl.concat([eg_to_remove1, eg_to_remove2], how="vertical")



    # 1) save mean and median proportion of compartments removed for each cell and
    # 2) percentage of cells that got filtered out for each strain
    delete_problematic_compartment_masks(
        db_path=args.database_path,
        filtered_comps=eg_to_remove,
        comp_name="EGs",
        output_dir=args.qc_directory,
        plate=args.plate,
        delete_all_comp_masks=args.delete_all_comps,
        replace_comp_num_with=-1,
        save_csv="True")

    print("Complete.")
