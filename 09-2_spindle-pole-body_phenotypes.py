import argparse
import math
import os
import polars as pl
import sqlite3

from GEN_outlier_detection_functions import (scale_compartment_feature,
                                             identify_outlier_cells,
                                             calculate_strain_penetrances,
                                             tabulate_strain_cell_counts,
                                             get_strain_hits,
                                             run_all_functions,
                                             combine_output_phenotypes_from_plate,
                                             tabulate_compartment_masks_per_strain)

from GEN_quality_check_functions import (delete_problematic_compartment_masks)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save phenotype information.')
parser.add_argument('-p', '--plate', default='', help='Plate identifier for saving files.')

args = parser.parse_args()


# Function for getting SPBs likely to be unseparated
def get_unseparated_spbs(db_path, output_dir, plate):
    """
    Gets SPBs that are likely to be unseparated (two SPBs close enough to each other that they get segmented as a single unit).

    Args:
        db_path (str): path to database with compartment and cell information
        output_dir (str): where to write output table
        plate (str): plate identifier for saving output file

    Returns:
        pl.DataFrame with all SPB masks likely to be unseparated SPBs
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conn = sqlite3.connect(db_path)

    likely_unseparated_spbs = (
        pl
        .read_database(
            query=f"""SELECT 
                        Per_SPBs.Replicate,
                        Per_SPBs.Condition,
                        Per_SPBs.Row,
                        Per_SPBs.Column,
                        Per_SPBs.Cell_ID,
                        Per_SPBs.SPBs_Number_Object_Number,
                        Per_SPBs.ORF,
                        Per_SPBs.Name,
                        Per_SPBs.Strain_ID,
                        pc.Predicted_Label
                      FROM 
                        Per_SPBs
                      JOIN (SELECT Cell_ID, Predicted_Label, Cell_Children_SPBs_Count FROM Per_Cell) pc
                        ON Per_SPBs.Cell_ID = pc.Cell_ID
                      WHERE
                        (SPBs_AreaShape_Solidity <= 0.85) AND 
                        (pc.Cell_Children_SPBs_Count = 1);""",
            connection=conn)
    )
    likely_unseparated_spbs.write_csv(f"{output_dir}/{plate}_SPBs_likely_unseparated.csv")
    conn.close()

    return likely_unseparated_spbs


# Function for getting cells that have too many/too few SPB given their CC stage
def get_cells_with_abnormal_spb_count(db_path, output_dir, plate, spb_count="too_many", unseparated_spbs=""):
    """
    Gets cells that have too many or too few SPBs given their cell cycle stage.

    Args:
        db_path (str): path to database with compartment and cell information
        output_dir (str): where to write output table
        plate (str): plate identifier for saving output file
        spb_count (str): specify if looking for cells that have too many SPBs ('too_many') or too few ('too_few'); defaults to too_many
        unseparated_spbs (Optional(str, pl.DataFrame)): if given, filters out these cells from the outlier dataframe before returning

    Returns:
        pl.DataFrame with cells that have an abnormal SPB count
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conn = sqlite3.connect(db_path)

    query = """
            SELECT
                Replicate,
                Condition,
                Row,
                Column, 
                Cell_ID, 
                ORF, 
                Name, 
                Strain_ID, 
                Predicted_Label
            FROM Per_Cell
            WHERE 
                (Predicted_Label = 'G1' AND Cell_Children_SPBs_Count > 1) OR 
                (Predicted_Label = 'SG2' AND Cell_Children_SPBs_Count > 2) OR 
                (Predicted_Label = 'MAT' AND Cell_Children_SPBs_Count > 2);
            """

    if spb_count == "too_few":
        query = """
                SELECT
                    Replicate,
                    Condition,
                    Row,
                    Column, 
                    Cell_ID, 
                    ORF, 
                    Name, 
                    Strain_ID, 
                    Predicted_Label
                FROM Per_Cell
                WHERE 
                    (Predicted_Label = 'G1' AND Cell_Children_SPBs_Count = 0) OR 
                    (Predicted_Label = 'SG2' AND Cell_Children_SPBs_Count IN (0, 1)) OR 
                    (Predicted_Label = 'MAT' AND Cell_Children_SPBs_Count IN (0, 1));
                """

    outlier_cells = pl.read_database(query=query, connection=conn)
    conn.close()

    if isinstance(unseparated_spbs, pl.DataFrame):
        outlier_cells = outlier_cells.filter(~pl.col("Cell_ID").is_in(unseparated_spbs["Cell_ID"]))

    outlier_cells.write_csv(f"{output_dir}/{plate}_SPBs_outlier_cells.csv")

    return outlier_cells


# Function for generating SPB size table where unseparated SPBs are split into half
def generate_spb_size_table(db_path, unseparated_spbs):
    """
    Creates a table with cell/SPB info and SPBs_AreaShape_Area for all SPBs so they can be scaled later. For unseparated
    SPBs, halves their total area (i.e., artificially splits them) so they don't affect
    size scaling calculations later.

    Args:
        db_path (str): path to database with compartment and cell information
        unseparated_spbs (pl.DataFrame): dataframe with all likely unseparated SPBs

    Returns:
        pl.DataFrame with SPBs_AreaShape_Area feature for all SPBs
    """
    conn = sqlite3.connect(db_path)
    all_spb_areas = (
        pl
        .read_database(
            query=f"""SELECT 
                        Replicate, 
                        Condition, 
                        Row, 
                        Column, 
                        Per_SPBs.Cell_ID,
                        SPBs_Number_Object_Number, 
                        ORF, 
                        Name, 
                        Strain_ID, 
                        Predicted_Label, 
                        SPBs_AreaShape_Area 
                      FROM Per_SPBs
                      JOIN (SELECT Cell_ID, Predicted_Label FROM Per_Cell) pc
                        ON Per_SPBs.Cell_ID = pc.Cell_ID;""",
            connection=conn
        )
    )
    conn.close()

    separated_spbs_subset = (
        all_spb_areas
        .filter(
            (~pl.col("Cell_ID").is_in(unseparated_spbs["Cell_ID"])) &
            (~pl.col("SPBs_Number_Object_Number").is_in(unseparated_spbs["SPBs_Number_Object_Number"]))
        )
    )

    unseparated_spbs_subset = (
        all_spb_areas
        .filter(
            (pl.col("Cell_ID").is_in(unseparated_spbs["Cell_ID"])) &
            (pl.col("SPBs_Number_Object_Number").is_in(unseparated_spbs["SPBs_Number_Object_Number"]))
        )
        .with_columns((pl.col("SPBs_AreaShape_Area") / 2).alias("SPBs_AreaShape_Area"))
    )

    all_spb_areas = pl.concat([separated_spbs_subset, unseparated_spbs_subset], how="vertical")

    return all_spb_areas


# Function for getting SPB distances and orientations
def calculate_spb_distances_and_orientations(db_path, plate, output_dir, cell_cycle_stages=["G1", "SG2", "MAT"]):
    """
    For cells that have two SPBs, calculates the distance between their centers, normalizes values to cell's major axis
    length, then scales according to wildtype cells for each cell cycle stage. Calculates one-sided pvalues. In addition,
    calculates the orientation of SPBs and makes them comparable with cell orientation.

    Args:
        db_path (str): path to database with cell/compartment information
        plate (str): plate identifier for saving output file
        output_dir (str): where to save output tables
        cell_cycle_stages (list, optional): cell cycle stages to filter by; defaults to ["G1", "SG2", "MAT"]

    Returns:
        pl.DataFrame with SPB distances and orientations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conn = sqlite3.connect(db_path)

    # Read SPB data for cells that have 2 SPBs, calculate distance using distance formula
    # dist = sqrt((x2-x1)**2 + (y2-y1)**2)
    cell_and_spb_distance_orientation = (
        pl
        # First read cell coordinates for all SPBs coming from cells that only have two SPB masks
        # ROW_NUMBER() is used for pivoting from long to wide
        .read_database(
            query=f"""
                    SELECT 
                        ROW_NUMBER() OVER (PARTITION BY Per_SPBs.Cell_ID) AS SPB_Num,
                        Replicate,
                        Condition,
                        Row,
                        Column,
                        ORF,
                        Name,
                        Strain_ID,
                        Per_SPBs.Cell_ID, 
                        Predicted_Label, 
                        SPBs_AreaShape_Center_X AS SPB_Center_X, 
                        SPBs_AreaShape_Center_Y AS SPB_Center_Y
                    FROM Per_SPBs
                    JOIN (SELECT Cell_ID, Cell_Children_SPBs_Count, Predicted_Label FROM Per_Cell) pc
                        ON Per_SPBs.Cell_ID = pc.Cell_ID
                    WHERE 
                        (Cell_Children_SPBs_Count = 2) AND 
                        (Predicted_Label IN ({",".join(f"'{cc_stage}'" for cc_stage in cell_cycle_stages)}));
                    """,
            connection=conn
        )
        # So far, there are two rows for each Cell_ID (one for each SPB)
        # Here, we pivot from long to wide on SPB_Num so that each Cell_ID now has one row
        # Each row will have SPB_Center_X_1, SPB_Center_X_2, SPB_Center_Y_1, and SPB_Center_Y_2 columns that will
        # make calculating distances easy
        .pivot(
            on=["SPB_Num"],
            index=["Replicate", "Condition", "Row", "Column", "ORF", "Name", "Strain_ID", "Cell_ID", "Predicted_Label"]
        )
        # Now create a new column that actually calculates the distance between two SPBs
        .with_columns(
            (
                (
                    (pl.col("SPB_Center_X_2") - pl.col("SPB_Center_X_1")) ** 2 +
                    (pl.col("SPB_Center_Y_2") - pl.col("SPB_Center_Y_1")) ** 2
                ) ** 0.5
            )
            .alias("SPB_Distance_Unscaled")
        )
        # Here, use the x/y coordinates for spindles to calculate their orientation in a way that
        # allows them to be compared against their cell's orientation. Orientations are limited
        # to 0-180 degrees.
        .with_columns(
            (
                180 - (
                        (
                            pl.arctan2(
                                (pl.col("SPB_Center_Y_2") - pl.col("SPB_Center_Y_1")),
                                (pl.col("SPB_Center_X_2") - pl.col("SPB_Center_X_1"))
                            ) * (180 / math.pi)
                        ) % 360)
            ).alias("SPB_Orientation")
        )
        .with_columns(
            (
                pl
                .when(pl.col("SPB_Orientation") > 180)
                .then(pl.col("SPB_Orientation") - 180)
                .when(pl.col("SPB_Orientation") < 0)
                .then(pl.col("SPB_Orientation") + 180)
                .otherwise(pl.col("SPB_Orientation")))
            .alias("SPB_Orientation")
        )
        # Add in the Cell orientations (make them go from 0-180 as well) AND include the Cell's major axis length for
        # normalizing SPB distance
        .join(
            pl
            .read_database(
                query="""
                        SELECT 
                            Cell_ID, 
                            Cell_AreaShape_Orientation AS Cell_Orientation,
                            Cell_AreaShape_MajorAxisLength AS Cell_MajorAL
                        FROM Per_Cell""", connection=conn),
            on="Cell_ID", how="left")
        .with_columns(
            (pl.col("Cell_Orientation") + 90).alias("Cell_Orientation"),
            (pl.col("SPB_Distance_Unscaled") / pl.col("Cell_MajorAL")).alias("SPB_Distance_Normalized")
        )
        # To make interpreting orientation easier, put them into categorical groups for cell and spindles
        .with_columns(
            (
                pl
                .when((pl.col("Cell_Orientation") >= 0) & (pl.col("Cell_Orientation") < 45))
                .then(pl.lit("I"))
                .when((pl.col("Cell_Orientation") >= 45) & (pl.col("Cell_Orientation") < 90))
                .then(pl.lit("II"))
                .when((pl.col("Cell_Orientation") >= 90) & (pl.col("Cell_Orientation") < 135))
                .then(pl.lit("III"))
                .when((pl.col("Cell_Orientation") >= 135) & (pl.col("Cell_Orientation") <= 180))
                .then(pl.lit("IV"))
                .otherwise(None)
            ).alias("Cell_Orientation_Class")
        )
        .with_columns(
            (
                pl
                .when((pl.col("SPB_Orientation") >= 0) & (pl.col("SPB_Orientation") < 45))
                .then(pl.lit("I"))
                .when((pl.col("SPB_Orientation") >= 45) & (pl.col("SPB_Orientation") < 90))
                .then(pl.lit("II"))
                .when((pl.col("SPB_Orientation") >= 90) & (pl.col("SPB_Orientation") < 135))
                .then(pl.lit("III"))
                .when((pl.col("SPB_Orientation") >= 135) & (pl.col("SPB_Orientation") <= 180))
                .then(pl.lit("IV"))
                .otherwise(None)
            ).alias("SPB_Orientation_Class")
        )
        # Use orientation difference for getting outliers later
        .with_columns(
            (pl.col("SPB_Orientation") - pl.col("Cell_Orientation")).abs().alias("Orientation_Difference")
        )
    )
    conn.close()

    cell_and_spb_distance_orientation.write_csv(f"{output_dir}/{plate}_spb_distances_and_orientations.csv")

    return cell_and_spb_distance_orientation


if __name__ == '__main__':

    conn = sqlite3.connect(args.database_path)
    all_cells = (
        pl
        .read_database(
            query="SELECT Replicate, Condition, Row, Column, Cell_ID, ORF, Name, Strain_ID, Predicted_Label FROM Per_Cell;",
            connection=conn
        )
    )
    conn.close()

# ============================== UNSEPARATED SPBs ==============================
# When SPBs are too close together (like early S/G2, they get segmented as a single unit. This poses issues when looking
# for SPB size/number defects. Ex., two unseparated (but normal-sized) SPBs can be mistaken for a large SPB. Therefore,
# the first step before characterizing SPB phenotypes is to identify these cases.

# From looking at SPB images, I've determined that SPBs with solidity <= 0.85 are likely to be unseparated
# SPBs. In addition, I will only consider cells with only one SPB/unseparated SPB mask for additional
# robustness.
    likely_unseparated_spbs = get_unseparated_spbs(
            db_path=args.database_path,
            output_dir=f"{args.output_directory}/unseparated_spbs/likely_unseparated",
            plate=args.plate)

    # Cells with unseparated SPBs
    likely_unseparated_spbs_filtered = (
        likely_unseparated_spbs
        .drop(["SPBs_Number_Object_Number"])
        .unique()
    )

    penetrance_table_unseparated_spb = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=likely_unseparated_spbs_filtered,
        output_dir=f"{args.output_directory}/unseparated_spbs/penetrances",
        plate=args.plate,
        compartment_name="SPBs",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
            all_cells=all_cells,
            all_outlier_cells=likely_unseparated_spbs_filtered,
            output_dir=f"{args.output_directory}/unseparated_spbs/cell_counts",
            plate=args.plate,
            compartment_name="SPBs",
            cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=likely_unseparated_spbs_filtered,
        penetrance_table=penetrance_table_unseparated_spb,
        output_dir=f"{args.output_directory}/unseparated_spbs/strain_hits",
        wt_pens_dir=f"{args.output_directory}/unseparated_spbs/per_well_wt_pens",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)


# ============================== NUMBER OF SPBs IN CELL (TOO FEW/TOO MANY) ==============================
    # Before starting, I need to identify diffuse SPBs that get segmented as massive mask(s) almost as big as the
    # nucleus. These will be deleted and the SPB count for the respective cells will be changed to 0. The phenotype
    # will be treated as 'absent/dim/diffuse SPBs'.
    conn = sqlite3.connect(args.database_path)

    diffuse_spbs = (
        pl
        .read_database(
            query="""
                  WITH nuc AS (
                        SELECT
                        	Cell_ID,
                        	Condition,
                        	Name,
                        	SUM(Nuclei_AreaShape_Area) AS Nuclear_Area
                        FROM Per_Nuclei
                        GROUP BY Cell_ID),

                  spb AS (
                        SELECT
                        	Cell_ID,
                        	SUM(SPBs_AreaShape_Area) AS SPB_Area
                        FROM Per_SPBs
                        GROUP BY Cell_ID)
                  
                  SELECT
                  	nuc.Cell_ID,
                  	SPBs_Number_Object_Number
                  FROM nuc
                  JOIN spb ON nuc.Cell_ID = spb.Cell_ID
                  JOIN Per_SPBs ON nuc.Cell_ID = Per_SPBs.Cell_ID
                  WHERE SPB_Area >= Nuclear_Area * 0.75;
                  """,
            connection=conn
        )
        .filter(~pl.col("Cell_ID").is_in(likely_unseparated_spbs_filtered["Cell_ID"])) # leave out masks likely to be unseparated SPBs
    )

    conn.close()

    delete_problematic_compartment_masks(
        db_path=args.database_path,
        filtered_comps=diffuse_spbs,
        comp_name="SPBs",
        output_dir="",
        plate=args.plate,
        delete_all_comp_masks="True",
        replace_comp_num_with=0,
        save_csv="False")

    tabulate_compartment_masks_per_strain(
            db_path=args.database_path,
            compartment_name="SPBs",
            plate=args.plate,
            output_directory=f"{args.output_directory}/abnormal_spb_count/spb_count_tables")

    # many SPB (fractured SPB, weird SPB foci, or extra SPB due to stuff like CC arrest)
    too_many_outliers = get_cells_with_abnormal_spb_count(
        db_path=args.database_path,
        output_dir=f"{args.output_directory}/abnormal_spb_count/many_spb/outlier_cells",
        plate=args.plate,
        spb_count="too_many",
        unseparated_spbs=likely_unseparated_spbs_filtered
    )

    penetrance_table = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=too_many_outliers,
        output_dir=f"{args.output_directory}/abnormal_spb_count/many_spb/penetrances",
        plate=args.plate,
        compartment_name="SPBs",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=too_many_outliers,
        output_dir=f"{args.output_directory}/abnormal_spb_count/many_spb/cell_counts",
        plate=args.plate,
        compartment_name="SPBs",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=too_many_outliers,
        penetrance_table=penetrance_table,
        output_dir=f"{args.output_directory}/abnormal_spb_count/many_spb/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_spb_count/many_spb/per_well_wt_pens",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)

    # few SPB
    too_few_outliers = get_cells_with_abnormal_spb_count(
        db_path=args.database_path,
        output_dir=f"{args.output_directory}/abnormal_spb_count/few_spb/outlier_cells",
        plate=args.plate,
        spb_count="too_few",
        unseparated_spbs=""
    )

    penetrance_table = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=too_few_outliers,
        output_dir=f"{args.output_directory}/abnormal_spb_count/few_spb/penetrances",
        plate=args.plate,
        compartment_name="SPBs",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=too_few_outliers,
        output_dir=f"{args.output_directory}/abnormal_spb_count/few_spb/cell_counts",
        plate=args.plate,
        compartment_name="SPBs",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=too_few_outliers,
        penetrance_table=penetrance_table,
        output_dir=f"{args.output_directory}/abnormal_spb_count/few_spb/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_spb_count/few_spb/per_well_wt_pens",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)


# ============================== UNEXPECTED SPB SIZE (TOO SMALL/TOO LARGE) ==============================
    spb_size_table = generate_spb_size_table(
        db_path=args.database_path,
        unseparated_spbs=likely_unseparated_spbs)

    # large SPB
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="SPBs_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_spb_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_spb_size/large_spb/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_spb_size/large_spb/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_spb_size/large_spb/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_spb_size/large_spb/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_spb_size/large_spb/per_well_wt_pens",
        plate=args.plate,
        compartment_name="SPBs",
        feature_table=spb_size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # small SPB
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="SPBs_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_spb_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_spb_size/small_spb/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_spb_size/small_spb/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_spb_size/small_spb/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_spb_size/small_spb/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_spb_size/small_spb/per_well_wt_pens",
        plate=args.plate,
        compartment_name="SPBs",
        feature_table=spb_size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== ABNORMAL DISTANCE AND ORIENTATION ==============================
    distance_and_orientation_table = calculate_spb_distances_and_orientations(
        db_path=args.database_path,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        plate=args.plate,
        output_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/distance_orientation_tables"
    )

    # abnormally close
    run_all_functions(
            db_path=args.database_path,
            all_cells=all_cells,
            compartment_table_name="",
            feature_name="SPB_Distance_Normalized",
            scaled_feature_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_close/scaled_features",
            outlier_objects_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_close/outlier_cells",
            penetrance_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_close/penetrances",
            cell_count_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_close/cell_counts",
            strain_hits_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_close/strain_hits",
            wt_pens_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_close/per_well_wt_pens",
            plate=args.plate,
            compartment_name="SPBs",
            excluded_outlier_cc_stages=["G1"],  # SPBs being close in G1 is fully expected
            feature_table=distance_and_orientation_table,
            cell_cycle_stages=["G1", "SG2", "MAT"],
            outlier_pval_cutoff=0.05,
            right_sided_outliers=False,
            percentile_cutoff=0.95)

    # abnormally far
    run_all_functions(
            db_path=args.database_path,
            all_cells=all_cells,
            compartment_table_name="",
            feature_name="SPB_Distance_Normalized",
            scaled_feature_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_far/scaled_features",
            outlier_objects_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_far/outlier_cells",
            penetrance_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_far/penetrances",
            cell_count_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_far/cell_counts",
            strain_hits_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_far/strain_hits",
            wt_pens_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_far/per_well_wt_pens",
            plate=args.plate,
            compartment_name="SPBs",
            excluded_outlier_cc_stages=[],
            feature_table=distance_and_orientation_table,
            cell_cycle_stages=["G1", "SG2", "MAT"],
            outlier_pval_cutoff=0.05,
            right_sided_outliers=True,
            percentile_cutoff=0.95)

    # spb misaligned with cell
    misalignment_scaled = scale_compartment_feature(
        db_path=args.database_path,
        table_name="Per_SPBs",
        feature_name="Orientation_Difference",
        output_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/scaled_features",
        plate=args.plate,
        compartment_name="SPBs",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        feature_table=distance_and_orientation_table)

    misalignment_outliers = (
        # First get the outliers
        identify_outlier_cells(
            feature_pvals=misalignment_scaled,
            scaled_col_name="Orientation_Difference_Scaled",
            output_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/outlier_cells",
            plate=args.plate,
            compartment_name="SPBs",
            pval_cutoff=0.05,
            right_sided_outliers=True,
            excluded_cc_stages=["G1"])
        # For misalignment, objects whose orientation is categorized as I are about the same as those with IV classification.
        # Therefore, they're considered 'far' in terms of orientation difference, but they're not actually true outliers.
        # These will be removed.
        .filter(
            ~pl.struct(
                ["Cell_Orientation_Class", "SPB_Orientation_Class"]
            ).is_in([
                {"Cell_Orientation_Class": "I", "SPB_Orientation_Class": "IV"},
                {"Cell_Orientation_Class": "IV", "SPB_Orientation_Class": "I"}
            ]
            )
        )
    )
    misalignment_outliers.write_csv(f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/outlier_cells/{args.plate}_SPBs_outlier_cells.csv")

    misalignment_penetrances = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=misalignment_outliers,
        output_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/penetrances",
        plate=args.plate,
        compartment_name="SPBs",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=misalignment_outliers,
        output_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/cell_counts",
        plate=args.plate,
        compartment_name="SPBs",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=misalignment_outliers,
        penetrance_table=misalignment_penetrances,
        wt_pens_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/per_well_wt_pens",
        output_dir=f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/strain_hits",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "DimDiffuse": f"{args.output_directory}/abnormal_spb_count/few_spb/outlier_cells/{args.plate}_SPBs_outlier_cells.csv",
            "ExtraSPB": f"{args.output_directory}/abnormal_spb_count/many_spb/outlier_cells/{args.plate}_SPBs_outlier_cells.csv",
            "TooClose": f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_close/outlier_cells/{args.plate}_SPBs_outlier_cells.csv",
            "TooFar": f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_spb_distance/spb_too_far/outlier_cells/{args.plate}_SPBs_outlier_cells.csv",
            "Misaligned": f"{args.output_directory}/abnormal_spb_distance_orientation/abnormal_orientation/outlier_cells/{args.plate}_SPBs_outlier_cells.csv",
            "Unseparated": f"{args.output_directory}/unseparated_spbs/likely_unseparated/{args.plate}_SPBs_likely_unseparated.csv",
            "TooSmall": f"{args.output_directory}/abnormal_spb_size/small_spb/outlier_cells/{args.plate}_SPBs_outlier_cells.csv",
            "TooLarge": f"{args.output_directory}/abnormal_spb_size/large_spb/outlier_cells/{args.plate}_SPBs_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)

    print("Complete")
