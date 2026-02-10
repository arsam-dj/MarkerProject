import argparse
import os
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', default='', help='Path to directory with per-plate files.')
parser.add_argument('-s', '--file_suffix', default='', help='Suffix in file name of all files that are to be combined.')
parser.add_argument('-o', '--output_name', default='', help='What to name output file.')

args = parser.parse_args()


def combine_files(directory, file_suffix, output_name):
    """
    Given a path to directory with files for each plate, combines them into one single file.

    Args:
        directory (str): path to directory with files to be combined
        suffix (str): file name suffix for all files to be combined
        output_name (str): what to name output combined file
    """
    (
        pl
        .read_csv(f"{directory}/*_{file_suffix}.csv")
        .write_csv(f"{directory}/{output_name}.csv")
    )

    
if __name__ == '__main__':

    if f"{args.output_name}.csv" in os.listdir(args.directory):
        print("Combined file already exists.")

    else:
        combine_files(
            directory=args.directory,
            file_suffix=args.file_suffix,
            output_name=args.output_name
        )

        print("Complete.")