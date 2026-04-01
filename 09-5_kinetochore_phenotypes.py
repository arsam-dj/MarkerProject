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


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save phenotype information.')
parser.add_argument('-p', '--plate', default='', help='Plate identifier for saving files.')

args = parser.parse_args()


# Function for getting kinetochores likely to be unseparated
def get_unseparated_kinetochores(db_path, output_dir, plate):
    """
    Gets Kinetochores that are likely to be unseparated (two Kinetochores close enough to each other that they get segmented as a single unit).

    Args:
        db_path (str): path to database with compartment and cell information
        output_dir (str): where to write output table
        plate (str): plate identifier for saving output file

    Returns:
        pl.DataFrame with all kinetochore masks likely to be unseparated Kinetochores
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conn = sqlite3.connect(db_path)

    likely_unseparated_kinetochores = (
        pl
        .read_database(
            query=f"""SELECT 
                        Per_Kinetochores.Replicate,
                        Per_Kinetochores.Condition,
                        Per_Kinetochores.Row,
                        Per_Kinetochores.Column,
                        Per_Kinetochores.Cell_ID,
                        Per_Kinetochores.Kinetochores_Number_Object_Number,
                        Per_Kinetochores.ORF,
                        Per_Kinetochores.Name,
                        Per_Kinetochores.Strain_ID,
                        pc.Predicted_Label
                      FROM 
                        Per_Kinetochores
                      JOIN (SELECT Cell_ID, Predicted_Label, Cell_Children_Kinetochores_Count FROM Per_Cell) pc
                        ON Per_Kinetochores.Cell_ID = pc.Cell_ID
                      WHERE 
                        (pc.Cell_Children_Kinetochores_Count = 1) AND
                        (Kinetochores_AreaShape_Solidity <= 0.85) AND
                        (Kinetochores_AreaShape_Eccentricity >= 0.6) AND
                        (Kinetochores_AreaShape_Area BETWEEN 30 AND 160) AND
                        (Kinetochores_Intensity_IntegratedIntensity_GFP >= 0.03);""",
            connection=conn)
    )
    likely_unseparated_kinetochores.write_csv(f"{output_dir}/{plate}_Kinetochores_likely_unseparated.csv")
    conn.close()

    return likely_unseparated_kinetochores


# Function for getting cells that have too many/too few kinetochore given their CC stage
def get_cells_with_abnormal_kinetochore_count(db_path, output_dir, plate, kinetochore_count="too_many", unseparated_kinetochores=""):
    """
    Gets cells that have too many or too few kinetochores given their cell cycle stage.

    Args:
        db_path (str): path to database with compartment and cell information
        output_dir (str): where to write output table
        plate (str): plate identifier for saving output file
        kinetochore_count (str): specify if looking for cells that have too many kinetochores ('too_many') or too few ('too_few'); defaults to too_many
        unseparated_kinetochores (Optional(str, pl.DataFrame)): if given, filters out these cells from the outlier dataframe before returning

    Returns:
        pl.DataFrame with cells that have an abnormal kinetochore count
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
                (Predicted_Label = 'G1' AND Cell_Children_Kinetochores_Count > 1) OR 
                (Predicted_Label = 'SG2' AND Cell_Children_Kinetochores_Count > 2) OR 
                (Predicted_Label = 'MAT' AND Cell_Children_Kinetochores_Count > 2);
            """

    if kinetochore_count == "too_few":
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
                    (Predicted_Label = 'SG2' AND Cell_Children_Kinetochores_Count = 0) OR 
                    (Predicted_Label = 'MAT' AND Cell_Children_Kinetochores_Count IN (0, 1));
                """

    outlier_cells = pl.read_database(query=query, connection=conn)
    conn.close()

    if isinstance(unseparated_kinetochores, pl.DataFrame):
        outlier_cells = outlier_cells.filter(~pl.col("Cell_ID").is_in(unseparated_kinetochores["Cell_ID"]))

    outlier_cells.write_csv(f"{output_dir}/{plate}_Kinetochores_outlier_cells.csv")

    return outlier_cells


# Function for generating kinetochore size table where unseparated kinetochores are split into half
def generate_kinetochore_size_table(db_path, unseparated_kinetochores):
    """
    Creates a table with cell/kinetochore info and Kinetochores_AreaShape_Area for all kinetochores so they can be scaled later. 
    For unseparated kinetochores, halves their total area (i.e., artificially splits them) so they don't affect
    size scaling calculations later.

    Args:
        db_path (str): path to database with compartment and cell information
        unseparated_kinetochores (pl.DataFrame): dataframe with all likely unseparated kinetochores

    Returns:
        pl.DataFrame with Kinetochores_AreaShape_Area feature for all kinetochores
    """
    conn = sqlite3.connect(db_path)
    all_kinetochore_areas = (
        pl
        .read_database(
            query=f"""SELECT 
                        Replicate, 
                        Condition, 
                        Row, 
                        Column, 
                        Per_kinetochores.Cell_ID,
                        Kinetochores_Number_Object_Number, 
                        ORF, 
                        Name, 
                        Strain_ID, 
                        Predicted_Label, 
                        Kinetochores_AreaShape_Area 
                      FROM Per_kinetochores
                      JOIN (SELECT Cell_ID, Predicted_Label FROM Per_Cell) pc
                        ON Per_kinetochores.Cell_ID = pc.Cell_ID
                      WHERE Kinetochores_Parent_Nuclei != 0;""", # Kinetochores screen has a lot of cells with diffuse, 
                                                                 # non-kinetochore signal in cytoplasm that get segmented.
                                                                 # These non-real kinetochore masks are
            connection=conn
        )
    )
    conn.close()

    separated_kinetochores_subset = (
        all_kinetochore_areas
        .filter(
            (~pl.col("Cell_ID").is_in(unseparated_kinetochores["Cell_ID"])) &
            (~pl.col("Kinetochores_Number_Object_Number").is_in(unseparated_kinetochores["Kinetochores_Number_Object_Number"]))
        )
    )

    unseparated_kinetochores_subset = (
        all_kinetochore_areas
        .filter(
            (pl.col("Cell_ID").is_in(unseparated_kinetochores["Cell_ID"])) &
            (pl.col("Kinetochores_Number_Object_Number").is_in(unseparated_kinetochores["Kinetochores_Number_Object_Number"]))
        )
        .with_columns((pl.col("Kinetochores_AreaShape_Area") / 2).alias("Kinetochores_AreaShape_Area"))
    )

    all_kinetochore_areas = pl.concat([separated_kinetochores_subset, unseparated_kinetochores_subset], how="vertical")

    return all_kinetochore_areas


# Function for getting kinetochore distances and orientations
def calculate_kinetochore_distances_and_orientations(db_path, plate, output_dir, cell_cycle_stages=["G1", "SG2", "MAT"]):
    """
    For cells that have two kinetochores, calculates the distance between their centers, normalizes values to cell's major axis
    length, then scales according to wildtype cells for each cell cycle stage. Calculates one-sided pvalues. In addition,
    calculates the orientation of kinetochores and makes them comparable with cell orientation.

    Args:
        db_path (str): path to database with cell/compartment information
        plate (str): plate identifier for saving output file
        output_dir (str): where to save output tables
        cell_cycle_stages (list, optional): cell cycle stages to filter by; defaults to ["G1", "SG2", "MAT"]

    Returns:
        pl.DataFrame with kinetochores distances and orientations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conn = sqlite3.connect(db_path)

    # Read kinetochore data for cells that have 2 kinetochores, calculate distance using distance formula
    # dist = sqrt((x2-x1)**2 + (y2-y1)**2)
    cell_and_kinetochore_distance_orientation = (
        pl
        # First read cell coordinates for all kinetochores coming from cells that only have two kinetochore masks
        # ROW_NUMBER() is used for pivoting from long to wide
        .read_database(
            query=f"""
                    SELECT 
                        ROW_NUMBER() OVER (PARTITION BY Per_Kinetochores.Cell_ID) AS Kinetochore_Num,
                        Replicate,
                        Condition,
                        Row,
                        Column,
                        ORF,
                        Name,
                        Strain_ID,
                        Per_Kinetochores.Cell_ID, 
                        Predicted_Label, 
                        Kinetochores_AreaShape_Center_X AS Kinetochore_Center_X, 
                        Kinetochores_AreaShape_Center_Y AS Kinetochore_Center_Y
                    FROM Per_Kinetochores
                    JOIN (SELECT Cell_ID, Cell_Children_Kinetochores_Count, Predicted_Label FROM Per_Cell) pc
                        ON Per_Kinetochores.Cell_ID = pc.Cell_ID
                    WHERE 
                        (Cell_Children_Kinetochores_Count = 2) AND 
                        (Predicted_Label IN ({",".join(f"'{cc_stage}'" for cc_stage in cell_cycle_stages)}));
                    """,
            connection=conn
        )
        # So far, there are two rows for each Cell_ID (one for each kinetochore)
        # Here, we pivot from long to wide on Kinetochore_Num so that each Cell_ID now has one row
        # Each row will have Kinetochore_Center_X_1, Kinetochore_Center_X_2, Kinetochore_Center_Y_1, and Kinetochore_Center_Y_2 columns that will
        # make calculating distances easy
        .pivot(
            on=["Kinetochore_Num"],
            index=["Replicate", "Condition", "Row", "Column", "ORF", "Name", "Strain_ID", "Cell_ID", "Predicted_Label"]
        )
        # Now create a new column that actually calculates the distance between two kinetochores
        .with_columns(
            (
                (
                    (pl.col("Kinetochore_Center_X_2") - pl.col("Kinetochore_Center_X_1")) ** 2 +
                    (pl.col("Kinetochore_Center_Y_2") - pl.col("Kinetochore_Center_Y_1")) ** 2
                ) ** 0.5
            )
            .alias("Kinetochore_Distance_Unscaled")
        )
        # Here, use the x/y coordinates for kinetochores to calculate their orientation in a way that
        # allows them to be compared against their cell's orientation. Orientations are limited
        # to 0-180 degrees.
        .with_columns(
            (
                180 - (
                        (
                            pl.arctan2(
                                (pl.col("Kinetochore_Center_Y_2") - pl.col("Kinetochore_Center_Y_1")),
                                (pl.col("Kinetochore_Center_X_2") - pl.col("Kinetochore_Center_X_1"))
                            ) * (180 / math.pi)
                        ) % 360)
            ).alias("Kinetochore_Orientation")
        )
        .with_columns(
            (
                pl
                .when(pl.col("Kinetochore_Orientation") > 180)
                .then(pl.col("Kinetochore_Orientation") - 180)
                .when(pl.col("Kinetochore_Orientation") < 0)
                .then(pl.col("Kinetochore_Orientation") + 180)
                .otherwise(pl.col("Kinetochore_Orientation")))
            .alias("Kinetochore_Orientation")
        )
        # Add in the Cell orientations (make them go from 0-180 as well) AND include the Cell's major axis length for
        # normalizing kinetochore distance
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
            (pl.col("Kinetochore_Distance_Unscaled") / pl.col("Cell_MajorAL")).alias("Kinetochore_Distance_Normalized")
        )
        # To make interpreting orientation easier, put them into categorical groups for cell and kinetochore
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
                .when((pl.col("Kinetochore_Orientation") >= 0) & (pl.col("Kinetochore_Orientation") < 45))
                .then(pl.lit("I"))
                .when((pl.col("Kinetochore_Orientation") >= 45) & (pl.col("Kinetochore_Orientation") < 90))
                .then(pl.lit("II"))
                .when((pl.col("Kinetochore_Orientation") >= 90) & (pl.col("Kinetochore_Orientation") < 135))
                .then(pl.lit("III"))
                .when((pl.col("Kinetochore_Orientation") >= 135) & (pl.col("Kinetochore_Orientation") <= 180))
                .then(pl.lit("IV"))
                .otherwise(None)
            ).alias("Kinetochore_Orientation_Class")
        )
        # Use orientation difference for getting outliers later
        .with_columns(
            (pl.col("Kinetochore_Orientation") - pl.col("Cell_Orientation")).abs().alias("Orientation_Difference")
        )
    )
    conn.close()

    cell_and_kinetochore_distance_orientation.write_csv(f"{output_dir}/{plate}_Kinetochores_distances_and_orientations.csv")

    return cell_and_kinetochore_distance_orientation


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
    

# ============================== UNSEPARATED KINETOCHORES ==============================
# When kinetochores are too close together (like early S/G2, they get segmented as a single unit. This poses issues when looking
# for kinetochore size/number defects. Ex., two unseparated (but normal-sized) kinetochores can be mistaken for a large kinetochore. 
# Therefore, the first step before characterizing kinetochore phenotypes is to identify these cases.

    likely_unseparated_kinetochores = get_unseparated_kinetochores(
            db_path=args.database_path,
            output_dir=f"{args.output_directory}/unseparated_kinetochores/likely_unseparated",
            plate=args.plate)

    # Cells with unseparated kinetochores
    likely_unseparated_kinetochores_filtered = (
        likely_unseparated_kinetochores
        .drop(["Kinetochores_Number_Object_Number"])
        .unique()
    )

    penetrance_table_unseparated_kinetochore = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=likely_unseparated_kinetochores_filtered,
        output_dir=f"{args.output_directory}/unseparated_kinetochores/penetrances",
        plate=args.plate,
        compartment_name="Kinetochores",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
            all_cells=all_cells,
            all_outlier_cells=likely_unseparated_kinetochores_filtered,
            output_dir=f"{args.output_directory}/unseparated_kinetochores/cell_counts",
            plate=args.plate,
            compartment_name="Kinetochores",
            cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=likely_unseparated_kinetochores_filtered,
        penetrance_table=penetrance_table_unseparated_kinetochore,
        output_dir=f"{args.output_directory}/unseparated_kinetochores/strain_hits",
        wt_pens_dir=f"{args.output_directory}/unseparated_kinetochores/per_well_wt_pens",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)


# ============================== NUMBER OF KINETOCHORES IN CELL (TOO FEW/TOO MANY) ==============================
    tabulate_compartment_masks_per_strain(
            db_path=args.database_path,
            compartment_name="Kinetochores",
            plate=args.plate,
            output_directory=f"{args.output_directory}/abnormal_kinetochore_count/kinetochore_count_tables")

    # many kinetochore (fractured kinetochore, weird kinetochore foci, or extra kinetochore due to stuff like CC arrest)
    too_many_outliers = get_cells_with_abnormal_kinetochore_count(
        db_path=args.database_path,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_count/many_kinetochore/outlier_cells",
        plate=args.plate,
        kinetochore_count="too_many"
    )

    penetrance_table = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=too_many_outliers,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_count/many_kinetochore/penetrances",
        plate=args.plate,
        compartment_name="Kinetochores",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=too_many_outliers,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_count/many_kinetochore/cell_counts",
        plate=args.plate,
        compartment_name="Kinetochores",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=too_many_outliers,
        penetrance_table=penetrance_table,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_count/many_kinetochore/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_kinetochore_count/many_kinetochore/per_well_wt_pens",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)

    # few kinetochore
    too_few_outliers = get_cells_with_abnormal_kinetochore_count(
        db_path=args.database_path,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_count/few_kinetochore/outlier_cells",
        plate=args.plate,
        kinetochore_count="too_few"
    )

    penetrance_table = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=too_few_outliers,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_count/few_kinetochore/penetrances",
        plate=args.plate,
        compartment_name="Kinetochores",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=too_few_outliers,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_count/few_kinetochore/cell_counts",
        plate=args.plate,
        compartment_name="Kinetochores",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=too_few_outliers,
        penetrance_table=penetrance_table,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_count/few_kinetochore/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_kinetochore_count/few_kinetochore/per_well_wt_pens",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)


# ============================== UNEXPECTED KINETOCHORE SIZE (TOO SMALL/TOO LARGE) ==============================
    kinetochore_size_table = generate_kinetochore_size_table(
        db_path=args.database_path,
        unseparated_kinetochores=likely_unseparated_kinetochores)

    # large kinetochore
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Kinetochores_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_kinetochore_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_kinetochore_size/large_kinetochore/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_kinetochore_size/large_kinetochore/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_kinetochore_size/large_kinetochore/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_kinetochore_size/large_kinetochore/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_kinetochore_size/large_kinetochore/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Kinetochores",
        feature_table=kinetochore_size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # small kinetochore
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Kinetochores_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_kinetochore_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_kinetochore_size/small_kinetochore/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_kinetochore_size/small_kinetochore/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_kinetochore_size/small_kinetochore/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_kinetochore_size/small_kinetochore/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_kinetochore_size/small_kinetochore/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Kinetochores",
        feature_table=kinetochore_size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== ABNORMAL DISTANCE AND ORIENTATION ==============================
    distance_and_orientation_table = calculate_kinetochore_distances_and_orientations(
        db_path=args.database_path,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        plate=args.plate,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/distance_orientation_tables"
    )

    # abnormally close
    run_all_functions(
            db_path=args.database_path,
            all_cells=all_cells,
            compartment_table_name="",
            feature_name="Kinetochore_Distance_Normalized",
            scaled_feature_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_close/scaled_features",
            outlier_objects_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_close/outlier_cells",
            penetrance_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_close/penetrances",
            cell_count_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_close/cell_counts",
            strain_hits_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_close/strain_hits",
            wt_pens_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_close/per_well_wt_pens",
            plate=args.plate,
            compartment_name="Kinetochores",
            excluded_outlier_cc_stages=["G1"],  # kinetochores being close in G1 is fully expected
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
            feature_name="Kinetochore_Distance_Normalized",
            scaled_feature_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_far/scaled_features",
            outlier_objects_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_far/outlier_cells",
            penetrance_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_far/penetrances",
            cell_count_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_far/cell_counts",
            strain_hits_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_far/strain_hits",
            wt_pens_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_far/per_well_wt_pens",
            plate=args.plate,
            compartment_name="Kinetochores",
            excluded_outlier_cc_stages=[],
            feature_table=distance_and_orientation_table,
            cell_cycle_stages=["G1", "SG2", "MAT"],
            outlier_pval_cutoff=0.05,
            right_sided_outliers=True,
            percentile_cutoff=0.95)

    # kinetochore misaligned with cell
    misalignment_scaled = scale_compartment_feature(
        db_path=args.database_path,
        table_name="Per_Kinetochores",
        feature_name="Orientation_Difference",
        output_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/scaled_features",
        plate=args.plate,
        compartment_name="Kinetochores",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        feature_table=distance_and_orientation_table)

    misalignment_outliers = (
        # First get the outliers
        identify_outlier_cells(
            feature_pvals=misalignment_scaled,
            scaled_col_name="Orientation_Difference_Scaled",
            output_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/outlier_cells",
            plate=args.plate,
            compartment_name="Kinetochores",
            pval_cutoff=0.05,
            right_sided_outliers=True,
            excluded_cc_stages=["G1"])
        # For misalignment, objects whose orientation is categorized as I are about the same as those with IV classification.
        # Therefore, they're considered 'far' in terms of orientation difference, but they're not actually true outliers.
        # These will be removed.
        .filter(
            ~pl.struct(
                ["Cell_Orientation_Class", "Kinetochore_Orientation_Class"]
            ).is_in([
                {"Cell_Orientation_Class": "I", "Kinetochore_Orientation_Class": "IV"},
                {"Cell_Orientation_Class": "IV", "Kinetochore_Orientation_Class": "I"}
            ]
            )
        )
    )
    misalignment_outliers.write_csv(f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/outlier_cells/{args.plate}_Kinetochores_outlier_cells.csv")

    misalignment_penetrances = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=misalignment_outliers,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/penetrances",
        plate=args.plate,
        compartment_name="Kinetochores",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=misalignment_outliers,
        output_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/cell_counts",
        plate=args.plate,
        compartment_name="Kinetochores",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=misalignment_outliers,
        penetrance_table=misalignment_penetrances,
        wt_pens_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/per_well_wt_pens",
        output_dir=f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/strain_hits",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "DimDiffuseAbsent": f"{args.output_directory}/abnormal_kinetochore_count/few_kinetochore/outlier_cells/{args.plate}_Kinetochores_outlier_cells.csv",
            "Extrakinetochore": f"{args.output_directory}/abnormal_kinetochore_count/many_kinetochore/outlier_cells/{args.plate}_Kinetochores_outlier_cells.csv",
            "TooClose": f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_close/outlier_cells/{args.plate}_Kinetochores_outlier_cells.csv",
            "TooFar": f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_kinetochore_distance/kinetochore_too_far/outlier_cells/{args.plate}_Kinetochores_outlier_cells.csv",
            "Misaligned": f"{args.output_directory}/abnormal_kinetochore_distance_orientation/abnormal_orientation/outlier_cells/{args.plate}_Kinetochores_outlier_cells.csv",
            "Unseparated": f"{args.output_directory}/unseparated_kinetochores/likely_unseparated/{args.plate}_Kinetochores_likely_unseparated.csv",
            "TooSmall": f"{args.output_directory}/abnormal_kinetochore_size/small_kinetochore/outlier_cells/{args.plate}_Kinetochores_outlier_cells.csv",
            "TooLarge": f"{args.output_directory}/abnormal_kinetochore_size/large_kinetochore/outlier_cells/{args.plate}_Kinetochores_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)

    print("Complete")
