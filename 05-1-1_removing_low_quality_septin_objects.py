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
parser.add_argument('-n', '--qc_septin_nuc_coords', default='', help='Path to file with Cell_IDs septin positions relative to nuclei.')
parser.add_argument('-p', '--plate', default='', help='Number for identifying plate being processed.')
parser.add_argument('-x', '--delete_all_comps', default='False', help='Specify if all other compartments in a cell with a single problematic mask should also be deleted (True). False by default.')

args = parser.parse_args()


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    # Load septin features and get problematic septin masks
    septins_to_remove1 = ( 
        pl
        .read_csv(args.qc_compartment_features)
        .filter(
            (pl.col('Septins_AreaShape_Area') >= 175) |
            (pl.col('Septins_AreaShape_Area') <= 30) |
            (pl.col('Septins_AreaShape_Perimeter') >= 85) |
            (pl.col('Septins_AreaShape_Perimeter') <= 20) |
            (pl.col('Septins_AreaShape_Extent') <= 0.25)
        )
        .select(["Cell_ID", "Septins_Number_Object_Number"])
    )
    
    large_septins = ( 
        pl
        .read_csv(args.qc_cell_features)
        .filter(
            (pl.col('Septin_Coverage_Cell') >= 0.175)
        )
        .select(["Cell_ID"])
    )
    
    conn = sqlite3.connect(args.database_path)
    septins_to_remove2 = (
        pl
        .read_database(query="SELECT Cell_ID, Septins_Number_Object_Number FROM Per_Septins", connection=conn)
        .filter(pl.col("Cell_ID").is_in(large_septins["Cell_ID"]))
    )
    conn.close()
    
    septins_to_remove3 = ( 
        pl
        .read_csv(args.qc_septin_nuc_coords)
        .select(["Cell_ID", "Septins_Number_Object_Number"])
    )
    
    septins_to_remove = pl.concat([septins_to_remove1, septins_to_remove2, septins_to_remove3], how="vertical").unique()

    # 1) save mean and median proportion of compartments removed for each cell and
    # 2) percentage of cells that got filtered out for each strain
    delete_problematic_compartment_masks(
        db_path=args.database_path,
        filtered_comps=septins_to_remove,
        comp_name="Septins",
        output_dir=args.qc_directory,
        plate=args.plate,
        delete_all_comp_masks=args.delete_all_comps,
        replace_comp_num_with=-1,
        save_csv="True")

    print("Complete.")
