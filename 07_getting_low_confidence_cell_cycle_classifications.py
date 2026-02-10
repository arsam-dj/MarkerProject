import argparse
import os
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--pred_file', default='', help='Path to csv file with all cells, their predicted label, and predication probability.')
parser.add_argument('-o', '--output_directory', default='', help='Path to directory to write output files to.')
parser.add_argument('-p', '--plate', default='', help='Number for identifying plate being processed.')
parser.add_argument('-g', '--g1_thresh', default='0', help='Probability threshold for filtering out G1 cells.')
parser.add_argument('-s', '--sg2_thresh', default='0', help='Probability threshold for filtering out SG2 cells.')
parser.add_argument('-m', '--mat_thresh', default='0', help='Probability threshold for filtering out MAT cells.')

args = parser.parse_args()


def save_filtered_objects(cells_and_preds, g1_thresh, sg2_thresh, mat_thresh, output_path, plate):
    """
    First identifies and saves cells to be filtered out along with their gene/strain information. Then, saves the
    percentage of cells that got filtered out for each strain.

    Args:
        cells_and_preds (path): path to file with all cells and their cell cycle predictions for plate
        g1_thresh (float): threshold for filtering cells classified as G1
        sg2_thresh (float): threshold for filtering cells classified as SG2
        mat_thresh (float): threshold for filtering cells classified as MAT
        output_path (str): where to save list of filtered cells
        plate (str): plate identifier for saving output files
    """
    cells_and_preds_df = pl.read_csv(cells_and_preds)

    # Get filtered objects
    cells_and_preds_filtered = (
        cells_and_preds_df
        .filter(
            ((pl.col('Predicted_Label') == 'G1') & (pl.col('Max_Prob') < g1_thresh)) |
            ((pl.col('Predicted_Label') == 'SG2') & (pl.col('Max_Prob') < sg2_thresh)) |
            ((pl.col('Predicted_Label') == 'MAT') & (pl.col('Max_Prob') < mat_thresh))
        )
    )

    # Save filtered objects for each plate
    cells_and_preds_filtered.write_csv(f"{output_path}/{plate}_cells_classified_with_low_confidence.csv")

    # Getting percentage of cells filtered out for each strain
    total_cells = (
        cells_and_preds_df
        .group_by(["ORF", "Name", "Strain_ID"])
        .len()
        .rename({"len": "Total_Cells"})
    )

    total_filtered_cells = (
        cells_and_preds_filtered
        .group_by(["ORF", "Name", "Strain_ID"])
        .len()
        .rename({"len": "Dropped_Cells"})
    )

    (
        total_cells
        .join(total_filtered_cells, on=["ORF", "Name", "Strain_ID"], how="left")
        .with_columns(
            pl.col("Dropped_Cells").fill_null(strategy="zero"),
            ((pl.col("Dropped_Cells") / pl.col("Total_Cells")) * 100).alias("Percent_Dropped")
        )
        .with_columns(pl.col("Percent_Dropped").fill_null(strategy="zero"))
        .sort("Percent_Dropped", descending=True)
        .write_csv(f"{output_path}/{plate}_percentage_of_cells_dropped.csv")
    )


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Save:
    #   1) filtered objects for each plate
    #   2) percentage of cells that got filtered out for each strain
    save_filtered_objects(
        cells_and_preds=args.pred_file,
        g1_thresh=float(args.g1_thresh),
        sg2_thresh=float(args.sg2_thresh),
        mat_thresh=float(args.mat_thresh),
        output_path=args.output_directory,
        plate=args.plate
    )

    print("Complete.")

