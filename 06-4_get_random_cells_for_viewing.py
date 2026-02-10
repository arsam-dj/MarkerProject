import argparse
import os
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--classified_cells_directory', default='', help='Directory with files that have all cells and their cell cycle prediction for each plate.')
parser.add_argument('-c', '--combined_cells', default='', help='Path to file with all cells and their cell cycle classification.')
parser.add_argument('-o', '--output_directory', default='', help='Directory for writing output files.')
parser.add_argument('-x', '--cell_coordinates', default='', help='Path to file with all cell overlay coordinates.')

args = parser.parse_args()


### Check classifications of random cells
def get_random_classified_cell_images(all_classified_cells, path_coords_file, cc_stage, num_cells, output_dir, output_file_name, pred_prob=1.0):
    """
    Creates and saves an input file for singlecelltool to look at cells classified into a
    given cell cycle stage. Done to ensure classification works well.

    Args:
        all_classified_cells (str): path to file with all cells and their assigned cell cycle label
        path_coords_file (str): path to file with image paths and cell x/y coordinates
        cc_stage (str): classified cell cycle stage to filter cells by
        num_cells (int): number of random cells filtered by cc_stage to pull from each database
        output_dir (str): where to save sct input files
        output_file_name (str): what to save sct input files as
        pred_prob (float): value to filter cells by their predicted label probability - defaults to 1.0 (no filtering)
    """
    image_paths = pl.read_csv(path_coords_file)

    (
        pl
        .read_csv(all_classified_cells)
        .filter(
            (pl.col("Predicted_Label") == cc_stage) & (pl.col(f"{cc_stage}_Prob") <= pred_prob)
        )
        .sample(n=num_cells, with_replacement=False, shuffle=True, seed=1705)
        .select(["Cell_ID"])
        .join(image_paths, on=["Cell_ID"], how="left")
        .write_csv(f"{output_dir}/{output_file_name}")
    )


if __name__ == '__main__':
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    get_random_classified_cell_images(
        all_classified_cells=args.combined_cells,
        path_coords_file=args.cell_coordinates,
        cc_stage="G1",
        num_cells=5000,
        output_dir=args.output_directory,
        output_file_name="random_G1_cells.csv"
    )

    get_random_classified_cell_images(
        all_classified_cells=args.combined_cells,
        path_coords_file=args.cell_coordinates,
        cc_stage="SG2",
        num_cells=5000,
        output_dir=args.output_directory,
        output_file_name="random_SG2_cells.csv"
    )

    get_random_classified_cell_images(
        all_classified_cells=args.combined_cells,
        path_coords_file=args.cell_coordinates,
        cc_stage="MAT",
        num_cells=5000,
        output_dir=args.output_directory,
        output_file_name="random_MAT_cells.csv"
    )

    print("Complete.")

