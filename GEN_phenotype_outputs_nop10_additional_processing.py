import argparse
import os
from pathlib import Path
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--phenotypes_directory', default='', help='Path to directory with Cell/Compartment features.')
parser.add_argument('-c', '--sample_cell_count_file', default='', help='Path to file with all expected cell count columns.')
parser.add_argument('-p', '--sample_penetrance_file', default='', help='Path to file with all expected penetrance columns.')

args = parser.parse_args()

def add_additional_columns(sub_dir, sample_cols, fill_with):
    """
    Phenotype file outputs for Nop10 are inconsistent for its TSA plates 4 and 5, because those strains only have one 
    replicate. Given a cell count sub-directory containing these inconsistent files, the function inserts columns filled
    with a specified value pertaining to TS2 and TS3 so these files can be merged with other files.
    
    Args:
        sub_dir (str): path to directory with per-plate output files
        sample_cols (list): list of column names that should be present in the fixed file
        fill_with (Any): specify value that should fill in the newly added columns
    """
    
    
    inconsistent_files = [
        inconsistent_file for inconsistent_file in os.listdir(sub_dir) if 
        ("TSA_26C_Plate04" in inconsistent_file) or 
        ("TSA_26C_Plate05" in inconsistent_file) or 
        ("TSA_37C_Plate04" in inconsistent_file) or 
        ("TSA_37C_Plate05" in inconsistent_file)
        ]
    
    for inconsistent_file in inconsistent_files:
        file_df = pl.read_csv(f"{sub_dir}/{inconsistent_file}")
        
        missing_cols = [missing_col for missing_col in sample_cols if missing_col not in file_df.columns]
        for missing_col in missing_cols:
            file_df = (
                file_df
                .with_columns(
                    pl.lit(fill_with).alias(missing_col)
                )
            )
        
        (
            file_df
            .select(sample_cols)
            .write_csv(f"{sub_dir}/{inconsistent_file}")
        )


if __name__ == '__main__':
    cell_count_cols = pl.read_csv(args.sample_cell_count_file).columns
    cell_count_dirs = [str(p) for p in Path(args.phenotypes_directory).rglob("cell_counts")]
    for cell_count_dir in cell_count_dirs:
        add_additional_columns(sub_dir=cell_count_dir, sample_cols=cell_count_cols, fill_with=0)
        
    penetrance_cols = pl.read_csv(args.sample_penetrance_file).columns
    penetrance_dirs = [str(p) for p in Path(args.phenotypes_directory).rglob("penetrances")]
    for penetrance_dir in penetrance_dirs:
        add_additional_columns(sub_dir=penetrance_dir, sample_cols=penetrance_cols, fill_with=-1)


    print("Complete.")