import argparse
import os
import polars as pl
import sqlite3
from natsort import natsorted
from plotnine import ggplot, aes, geom_point, scale_x_discrete, scale_fill_manual, ylim, labs, theme_bw, theme, element_text, element_rect, ggsave

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_directory', default='', help='Path to directory with all feature databases.')
parser.add_argument('-c', '--outlier_cells_directory', default='', help='Path to directory with spreadsheet containing all outlier cells for all strains.')
parser.add_argument('-s', '--screen', default='', help='Name of screen for filtering strain files.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save spreadsheets with replicate-replicate distances.')

args = parser.parse_args()


def get_per_rep_pens_cell_counts_and_distances(database_directory, outlier_cells_directory, screen, output_directory):
    """
    Creates a spreadsheet that has overall per-replicate penetrances, cell counts, and replicate-replicate distances for
    all strains.

    Args:
        database_directory: path to directory with all feature databases
        outlier_cells: path to spreadsheet with all outlier cell IDs
        screen (str): name of screen being processed
        output_directory (str): where to save per-replicate penetrance and distance spreadsheet

    Returns:
         pl.DataFrame with per-replicate penetrance, cell counts, and replicate-replicate distances for all strains
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get number of outlier cells for each strain
    outlier_cells_files = [f"{outlier_cells_directory}/{file_name}" for file_name in os.listdir(outlier_cells_directory)]
    outlier_cells_files.remove(f"{outlier_cells_directory}/all_aggregated_cell_outlier_data.csv")

    outlier_cells_dfs = []
    for outlier_cells_file in outlier_cells_files:
        plate_num = outlier_cells_file.split("Plate")[1][0:2]
        outlier_cells_df = (
            pl
            .read_csv(outlier_cells_file)
            .with_columns(
                (pl.lit(plate_num).alias("Plate")),
                (
                    pl
                    .when(pl.col("Name").is_null())
                    .then(pl.lit(""))
                    .otherwise(pl.col("Name"))
                ).alias("Name")
            )
        )
        outlier_cells_dfs.append(outlier_cells_df)

    outlier_cell_counts = (
        pl
        .concat(outlier_cells_dfs, how="vertical")
        .select(["Plate", "Replicate", "Row", "Column", "ORF", "Name", "Strain_ID", "Cell_ID"])
        .group_by(["Plate", "Replicate", "Row", "Column", "ORF", "Name", "Strain_ID"])
        .len(name="Num_Outliers")
    )

    # Get total number of cells for each strain
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]

    cell_count_dfs = []
    for database in databases:
        conn = sqlite3.connect(database)

        cell_count_df = (
            pl
            .read_database(
                query=f"""
                       SELECT
	                        Plate,
	                        Replicate,
	                        CAST(Row AS NUMERIC) AS Row,
	                        CAST(Column AS NUMERIC) AS Column,
	                        ORF,
	                        Name,
	                        Strain_ID,
	                        COUNT (Cell_ID) AS Total_Num_Cells
                       FROM Per_Cell
                       GROUP BY Plate, Replicate, Row, Column, ORF, Name, Strain_ID;
                       """,
                connection=conn
            )
            .with_columns(
                (
                    pl
                    .when(pl.col("Name").is_null())
                    .then(pl.lit(""))
                    .otherwise(pl.col("Name"))
                ).alias("Name")
            )
        )
        cell_count_dfs.append(cell_count_df)
        conn.close()

    merged_cell_counts = (
        pl
        .concat(cell_count_dfs, how="vertical")
        .join(outlier_cell_counts, on=["Plate", "Replicate", "Row", "Column", "ORF", "Name", "Strain_ID"])
        .with_columns(
            ((pl.col("Num_Outliers") / pl.col("Total_Num_Cells")) * 100).alias("Penetrance"),
            pl.col("Replicate").replace({"TS1": "R1", "TS2": "R2", "TS3": "R3"})
        )
        .pivot(
            on=["Replicate"],
            index=["Plate", "Row", "Column", "ORF", "Name", "Strain_ID"],
            values=["Num_Outliers", "Total_Num_Cells", "Penetrance"]
        )
        .with_columns(
            (
                    (pl.col("Penetrance_R1") - pl.col("Penetrance_R2")).abs()
                    * pl.max_horizontal("Penetrance_R1", "Penetrance_R2")
                    / pl.mean_horizontal("Penetrance_R1", "Penetrance_R2")
            ).alias("Distance_R1-R2"),
            (
                    (pl.col("Penetrance_R1") - pl.col("Penetrance_R3")).abs()
                    * pl.max_horizontal("Penetrance_R1", "Penetrance_R3")
                    / pl.mean_horizontal("Penetrance_R1", "Penetrance_R3")
            ).alias("Distance_R1-R3"),
            (
                    (pl.col("Penetrance_R2") - pl.col("Penetrance_R3")).abs()
                    * pl.max_horizontal("Penetrance_R2", "Penetrance_R3")
                    / pl.mean_horizontal("Penetrance_R2", "Penetrance_R3")
            ).alias("Distance_R2-R3")
        )
    )

    merged_cell_counts.write_csv(f"{output_directory}/{screen}_per_replicate_penetrances_and_distances.csv")
    
    return merged_cell_counts


def rep_binner(distance_df, rep_numx, rep_numy, lower_bound, upper_bound):
    """
    From the full distance dataframe, this function returns rows whose total replicate cell count for RepX and/or RepY
    falls within the range indicated by lower_bound and upper_bound.

    Args:
        distance_df (pl.DataFrame): full dataframe with all strains and their penetrance/cell count/replicate distance info
        rep_numx (int): number of replicate X
        rep_numy (int): number of replicate Y
        lower_bound (int): lower boundary of replicate cell count for filtering
        upper_bound (int): upper boundary of replicate cell count for filtering

    Returns:
        pl.DataFrame that is a filtered version of the distance_df based on reps X/Y and lower/upper bound replicate cell count
    """

    repx_df = (
        distance_df
        .filter(
            pl.col(f"Total_Num_Cells_R{rep_numx}").is_between(lower_bound, upper_bound, closed="both")
        )
    )

    repy_df = (
        distance_df
        .filter(
            pl.col(f"Total_Num_Cells_R{rep_numy}").is_between(lower_bound, upper_bound, closed="both")
        )
    )

    repxy_bin = (
        pl
        .concat([repx_df, repy_df], how="vertical")
        .unique()
        .with_columns(
            pl.lit(f"{lower_bound}-{upper_bound}").alias("Bin")
        )
    )

    return repxy_bin


def combine_rep_bins(distance_df, rep_numx, rep_numy, max_lower_bound=3201, max_upper_bound=float('inf'), bin_size=200):
    """
    For given replicates X and Y, creates a dataframe with all replicate cell count bins. Bin sizes and starting/stopping
    points for bin generation can be specified.

    Args:
        distance_df (pl.DataFrame): dataframe with all replicate distances
        rep_numx (int): number of replicate X
        rep_numy (int): number of replicate Y
        max_lower_bound (int): the lower boundary of the last bin; defaults to 3201 (so the last bin would start from 3001)
        max_upper_bound (int): the upper boundary of the last bin; defaults to inf
        bin_size (int): how big each bin should be; defaults to 200

    Returns:
        pl.DataFrame with each row assigned to a possible cell count bin for a given replicate pair
    """

    lower_bounds = [0] + list(range(bin_size+1, max_lower_bound, bin_size))
    upper_bounds = list(range(bin_size, max_lower_bound-1, bin_size)) + [max_upper_bound]

    rxry_bins = []
    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        rxry_bin = rep_binner(distance_df, rep_numx, rep_numy, lower_bound, upper_bound)
        rxry_bins.append(rxry_bin)

    rxry_bins_concat = pl.concat(rxry_bins, how="vertical")

    return rxry_bins_concat


def aggregate_distances(cell_count_bin_df, metric):
    """
    Calculates either the median or the variance of each cell count bin.

    Args:
        cell_count_bin_df (pl.DataFrame): dataframe that has the cell count bin and replicate-replicate distances for all strains
        metric (str): specify either "median" or "variance" to determine the aggregation performed on replicate-replicate distances per bin

    Returns:
        pl.DataFrame that has two columns; cell count bin and the metric/variance for that bin
    """

    if metric == "Median":
        cell_count_bin_df_agg = (
            cell_count_bin_df
            .group_by("Bin")
            .agg(pl.col("Rep_Distance").median().alias("Bin_Median"))
        )

    else:
        cell_count_bin_df_agg = (
            cell_count_bin_df
            .group_by("Bin")
            .agg(pl.col("Rep_Distance").var().alias("Bin_Variance"))
        )
    
    return cell_count_bin_df_agg


def generate_distance_plot(distance_df, max_lower_bound, max_upper_bound, bin_size, screen, output_directory, metric="Median"):
    """
    Given a dataframe with replicate-replicate distances, generates a plot of distance medians or variances per specified cell count
    bin.

    Args:
        distance_df (pl.DataFrame): dataframe with replicate-replicate distances and per-replicate cell counts
        max_lower_bound (int): lower bound of the last cell count bin + bin_size (e.g., for max lower bound of 1001, enter 1051 if bin_size is 50
        max_upper_bound (int, float): upper bound of the last cell count bin; can be int but enter float('inf') to encompass all cell counts
        bin_size (int): how big each bin should be
        screen (str): name of screen being processed
        output_directory (str): where to save resulting plot
        metric (str): determine whether plot should display distance Medians or Variances; defaults to Median
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    r1r2_bins = (
        combine_rep_bins(distance_df, 1, 2, max_lower_bound, max_upper_bound, bin_size)
        .drop(["Distance_R1-R3", "Distance_R2-R3"])
        .rename({"Distance_R1-R2": "Rep_Distance"})
    )

    r1r3_bins = (
        combine_rep_bins(distance_df, 1, 3, max_lower_bound, max_upper_bound, bin_size)
        .drop(["Distance_R1-R2", "Distance_R2-R3"])
        .rename({"Distance_R1-R3": "Rep_Distance"})
    )

    r2r3_bins = (
        combine_rep_bins(distance_df, 2, 3, max_lower_bound, max_upper_bound, bin_size)
        .drop(["Distance_R1-R2", "Distance_R1-R3"])
        .rename({"Distance_R2-R3": "Rep_Distance"})
    )

    r1r2_agg = aggregate_distances(r1r2_bins, metric)
    r1r3_agg = aggregate_distances(r1r3_bins, metric)
    r2r3_agg = aggregate_distances(r2r3_bins, metric)

    r1r2_plot = r1r2_agg.with_columns(pl.lit("R1–R2").alias("Comparison"))
    r1r3_plot = r1r3_agg.with_columns(pl.lit("R1–R3").alias("Comparison"))
    r2r3_plot = r2r3_agg.with_columns(pl.lit("R2–R3").alias("Comparison"))

    plot_df = pl.concat([r1r2_plot, r1r3_plot, r2r3_plot]).to_pandas()

    # Get max distance for creating plot
    y_max = max(
        r1r2_agg.select(pl.col(f"Bin_{metric}").max()).item(),
        r1r3_agg.select(pl.col(f"Bin_{metric}").max()).item(),
        r2r3_agg.select(pl.col(f"Bin_{metric}").max()).item()
    )
    y_upper = y_max * 1.1

    # Set bin order
    bins = natsorted(
        r1r2_agg.select("Bin").to_series().to_list()
    )

    full_plot = (
            ggplot(plot_df, aes("Bin", f"Bin_{metric}", fill="Comparison")) + \
            geom_point(
                color="black",
                size=3.5,
                stroke=1.5
            ) + \
            scale_fill_manual(
                values={
                    "R1–R2": "#86bbd8",
                    "R1–R3": "#f6ae2d",
                    "R2–R3": "#f26419",
                }
            ) + \
            ylim(0, y_upper) + \
            scale_x_discrete(limits=bins) + \
            labs(
                title=f"{metric} of Replicate-Replicate Distances for each Cell Count Bin",
                x="Cell Count Bin",
                y=f"Distance {metric}"
            ) + \
            theme_bw() + \
            theme(
                axis_text=element_text(size=10),
                axis_title=element_text(size=12, weight="bold"),
                plot_title=element_text(size=16, weight="bold"),
                axis_text_x=element_text(angle=45, ha="right"),
                legend_position=(0.975, 0.975),  # (0, 0) places legend in bottom left, (1, 1) in top right
                legend_background=element_rect(fill="white", alpha=0.8),
                legend_title=element_text(size=10, weight="bold"),
                legend_text=element_text(size=9)
            )
    )

    full_plot.save(
        filename=f"{output_directory}/{screen}_replicate_distance_{metric.lower()}.pdf",
        bbox_inches="tight"
    )


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

    # Get distances for all strains
    distances = get_per_rep_pens_cell_counts_and_distances(
        database_directory=args.database_directory,
        outlier_cells_directory=args.outlier_cells_directory,
        screen=args.screen,
        output_directory=f"{args.output_directory}/per_replicate_penentrances_and_distances")

    # Remove wildtypes and ORFs too close to marker/HTA2/CAN1/LYP1
    distances = (
        distances
        .filter(
            ~pl.col("Strain_ID").is_in(wildtype_strains["Strain"].to_list()),
            ~pl.col("ORF").is_in(orfs_near_marker["ORF"].to_list())
        )
        .sort(["Plate", "Row", "Column"], descending=False)
        .with_row_index(name="Case_Num", offset=1)
    )

    generate_distance_plot(
        distance_df=distances,
        max_lower_bound=1051,
        max_upper_bound=float('inf'),
        bin_size=50,
        screen=args.screen,
        output_directory=f"{args.output_directory}/median_distance_plots",
        metric="Median")

    generate_distance_plot(
        distance_df=distances,
        max_lower_bound=1051,
        max_upper_bound=float('inf'),
        bin_size=50,
        screen=args.screen,
        output_directory=f"{args.output_directory}/variance_distance_plots",
        metric="Variance")

    print("Complete.")