import argparse
import os
import polars as pl
import sqlite3

from GEN_outlier_detection_functions import (scale_compartment_feature,
                                             calculate_strain_penetrances,
                                             tabulate_strain_cell_counts,
                                             get_strain_hits,
                                             run_all_functions,
                                             generate_compartment_feature_table,
                                             combine_output_phenotypes_from_plate)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save phenotype information.')
parser.add_argument('-p', '--plate', default='', help='Plate identifier for saving files.')

args = parser.parse_args()


def identify_orientation_outlier_cells(feature_pvals, output_dir, plate, compartment_name, pval_cutoff=0.05, excluded_cc_stages=[]):
    """
    Obtain and save outlier cells from a given cell population based on their calculated p-values.

    Args:
        feature_pvals (pl.DataFrame): dataframe that has cell cycle stage, Cell ID, and cell size pvalue
        output_dir (str): where to save output files
        plate (str): plate identifier for saving files
        compartment_name (str): name of compartment used in saving files
        pval_cutoff (float, optional): p-value cutoff for identifying outliers; cells with p-value below this are tagged outlier
        right_sided_outliers (bool): when set to true, only looks at cells with positive Z-Scores (indicating that they're above the wt distribution) when looking for outliers
        excluded_cc_stages (list(str, optional)): if there are any cell cycle stages from which no cells are considered to be outliers, exclude them
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Separate outlier cells from rest of population based on cutoff and detection strategy (above or below cutoff)
    outlier_cells = (
        feature_pvals
        .filter(
            (~pl.col("Predicted_Label").is_in(excluded_cc_stages)),
            (pl.col("pval") <= pval_cutoff),
            )
        .filter( # remove cases where cell or septin orientation is in class I/IV (far in terms of orientation difference but not actual outliers)
            ~pl.struct(
                ["Cell_Orientation_Class", "Septin_Orientation_Class"]
            ).is_in([
                {"Cell_Orientation_Class": "I", "Septin_Orientation_Class": "IV"},
                {"Cell_Orientation_Class": "IV", "Septin_Orientation_Class": "I"}
            ]
            )
        )
        )

    # Export all outlier cells
    outlier_cells.write_csv(f"{output_dir}/{plate}_{compartment_name}_outlier_cells.csv")

    return outlier_cells


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

# ============================== DIM/ABSENT SEPTIN ==============================
    conn = sqlite3.connect(args.database_path)
    cell_intensity_table = (
        pl
        .read_database(
            query="""
                    SELECT 
                        Replicate, 
                        Condition, 
                        Row, 
                        Column, 
                        Cell_ID, 
                        ORF, 
                        Name, 
                        Strain_ID, 
                        Predicted_Label, 
                        Cell_Intensity_IntegratedIntensity_GFP
                    FROM Per_Cell
                    WHERE Predicted_Label IN ('SG2', 'MAT');""",
            connection=conn
        )
    )
    conn.close()
    
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_IntegratedIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/dim_septin/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/dim_septin/outlier_cells",
        penetrance_dir=f"{args.output_directory}/dim_septin/penetrances",
        cell_count_dir=f"{args.output_directory}/dim_septin/cell_counts",
        strain_hits_dir=f"{args.output_directory}/dim_septin/strain_hits",
        wt_pens_dir=f"{args.output_directory}/dim_septin/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Septins",
        feature_table=cell_intensity_table,
        cell_cycle_stages=["SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== THICK/THIN SEPTIN ==============================
    min_axis_length_table = generate_compartment_feature_table(db_path=args.database_path, feature="Septins_AreaShape_MinorAxisLength", comp_name="Septins")

    # thick septin
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Septins_AreaShape_MinorAxisLength",
        scaled_feature_dir=f"{args.output_directory}/abnormal_septin_thickness/thick_septin/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_septin_thickness/thick_septin/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_septin_thickness/thick_septin/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_septin_thickness/thick_septin/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_septin_thickness/thick_septin/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_septin_thickness/thick_septin/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Septins",
        feature_table=min_axis_length_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)
    
    # thin septin
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Septins_AreaShape_MinorAxisLength",
        scaled_feature_dir=f"{args.output_directory}/abnormal_septin_thickness/thin_septin/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_septin_thickness/thin_septin/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_septin_thickness/thin_septin/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_septin_thickness/thin_septin/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_septin_thickness/thin_septin/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_septin_thickness/thin_septin/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Septins",
        feature_table=min_axis_length_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)
  
    
# ============================== LARGE/SMALL SEPTIN ==============================
    size_table = generate_compartment_feature_table(db_path=args.database_path, feature="Septins_AreaShape_Area", comp_name="Septins")

    # large septin
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Septins_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_septin_size/large_septin/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_septin_size/large_septin/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_septin_size/large_septin/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_septin_size/large_septin/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_septin_size/large_septin/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_septin_size/large_septin/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Septins",
        feature_table=size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)
    
    # small septin
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Septins_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_septin_size/small_septin/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_septin_size/small_septin/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_septin_size/small_septin/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_septin_size/small_septin/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_septin_size/small_septin/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_septin_size/small_septin/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Septins",
        feature_table=size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)
    
    
# ============================== MISORIENTED SEPTIN ==============================
    conn = sqlite3.connect(args.database_path)
    orientation_table = (
        pl
        .read_database(
            query="""
                SELECT
	                Replicate, 
	                Condition, 
	                Row, 
	                Column, 
	                Per_Cell.Cell_ID, 
	                ORF, 
	                Name, 
	                Strain_ID, 
	                Predicted_Label,
	                Cell_AreaShape_Orientation,
	                Septins_AreaShape_Orientation
                FROM Per_Cell
                JOIN (SELECT Cell_ID, Septins_AreaShape_Orientation FROM Per_Septins) ps
                	ON Per_Cell.Cell_ID = ps.Cell_ID
                WHERE (Cell_Children_Septins_Count == 1) AND (Predicted_Label IN ('SG2', 'MAT'));""",
            connection=conn
        )
        .with_columns(
            (180 - (pl.col("Cell_AreaShape_Orientation") % 360)).alias("Cell_Orientation"),
            (180 - (pl.col("Septins_AreaShape_Orientation") % 360)).alias("Septin_Orientation")
            )
        .with_columns(
            (
                pl
                .when(pl.col("Cell_Orientation") > 180)
                .then(pl.col("Cell_Orientation") - 180)
                .when(pl.col("Cell_Orientation") < 0)
                .then(pl.col("Cell_Orientation") + 180)
                .otherwise(pl.col("Cell_Orientation")))
            .alias("Cell_Orientation")
        )
        .with_columns(
            (
                pl
                .when(pl.col("Septin_Orientation") > 180)
                .then(pl.col("Septin_Orientation") - 180)
                .when(pl.col("Septin_Orientation") < 0)
                .then(pl.col("Septin_Orientation") + 180)
                .otherwise(pl.col("Septin_Orientation")))
            .alias("Septin_Orientation")
        )
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
                .when((pl.col("Septin_Orientation") >= 0) & (pl.col("Septin_Orientation") < 45))
                .then(pl.lit("I"))
                .when((pl.col("Septin_Orientation") >= 45) & (pl.col("Septin_Orientation") < 90))
                .then(pl.lit("II"))
                .when((pl.col("Septin_Orientation") >= 90) & (pl.col("Septin_Orientation") < 135))
                .then(pl.lit("III"))
                .when((pl.col("Septin_Orientation") >= 135) & (pl.col("Septin_Orientation") <= 180))
                .then(pl.lit("IV"))
                .otherwise(None)
            ).alias("Septin_Orientation_Class")
        )
        .with_columns(
            (pl.col("Cell_Orientation") - pl.col("Septin_Orientation")).abs().alias("Orientation_Difference")
        )
        .filter(pl.col("Cell_Orientation_Class") != pl.col("Septin_Orientation_Class"))
    )
    conn.close()
    
    all_objects_scaled = scale_compartment_feature(
        db_path=args.database_path, 
        table_name="", 
        feature_name="Orientation_Difference", 
        output_dir=f"{args.output_directory}/abnormal_septin_orientation/scaled_features", 
        plate=args.plate, 
        compartment_name="Septins",
        cell_cycle_stages=["SG2", "MAT"], 
        feature_table=orientation_table)
    
    outlier_objects = identify_orientation_outlier_cells(
        feature_pvals=all_objects_scaled,
        output_dir=f"{args.output_directory}/abnormal_septin_orientation/outlier_cells",
        plate=args.plate,
        compartment_name="Septins",
        pval_cutoff=0.025,
        excluded_cc_stages=["G1"])

    penetrance_table = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=outlier_objects,
        output_dir=f"{args.output_directory}/abnormal_septin_orientation/penetrances",
        plate=args.plate,
        compartment_name="Septins",
        cell_cycle_stages=["SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=outlier_objects,
        output_dir=f"{args.output_directory}/abnormal_septin_orientation/cell_counts",
        plate=args.plate,
        compartment_name="Septins",
        cell_cycle_stages=["SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=outlier_objects,
        penetrance_table=penetrance_table,
        wt_pens_dir=f"{args.output_directory}/abnormal_septin_orientation/per_well_wt_pens",
        output_dir=f"{args.output_directory}/abnormal_septin_orientation/strain_hits",
        plate=args.plate,
        cc_stages=["SG2", "MAT"],
        percentile_cutoff=0.95)


# ============================== IMPROPERLY FORMED SEPTIN ==============================
    conn = sqlite3.connect(args.database_path)
    outlier_objects = (
        pl
        .read_database(
            query="""
                SELECT
	                Replicate, 
	                Condition, 
	                Row, 
	                Column, 
	                Per_Cell.Cell_ID, 
	                ORF, 
	                Name, 
	                Strain_ID, 
	                Predicted_Label,
	                Cell_Children_Septins_Count
                FROM Per_Cell
                WHERE (Cell_Children_Septins_Count > 1);""",
            connection=conn
        )
    )
    conn.close()
    
    if not os.path.exists(f"{args.output_directory}/septin_fragmentation/outlier_cells"):
        os.makedirs(f"{args.output_directory}/septin_fragmentation/outlier_cells")
    outlier_objects.write_csv(f"{args.output_directory}/septin_fragmentation/outlier_cells/{args.plate}_Septins_outlier_cells.csv")
    
    penetrance_table = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=outlier_objects,
        output_dir=f"{args.output_directory}/septin_fragmentation/penetrances",
        plate=args.plate,
        compartment_name="Septins",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=outlier_objects,
        output_dir=f"{args.output_directory}/septin_fragmentation/cell_counts",
        plate=args.plate,
        compartment_name="Septins",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=outlier_objects,
        penetrance_table=penetrance_table,
        wt_pens_dir=f"{args.output_directory}/septin_fragmentation/per_well_wt_pens",
        output_dir=f"{args.output_directory}/septin_fragmentation/strain_hits",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)
    
    
# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "DimAbsent": f"{args.output_directory}/dim_septin/outlier_cells/{args.plate}_Septins_outlier_cells.csv",
            "ThickSeptin": f"{args.output_directory}/abnormal_septin_thickness/thick_septin/outlier_cells/{args.plate}_Septins_outlier_cells.csv",
            "ThinSeptin": f"{args.output_directory}/abnormal_septin_thickness/thin_septin/outlier_cells/{args.plate}_Septins_outlier_cells.csv",
            "LargeSeptin": f"{args.output_directory}/abnormal_septin_size/large_septin/outlier_cells/{args.plate}_Septins_outlier_cells.csv",
            "SmallSeptin": f"{args.output_directory}/abnormal_septin_size/small_septin/outlier_cells/{args.plate}_Septins_outlier_cells.csv",
            "Misoriented": f"{args.output_directory}/abnormal_septin_orientation/outlier_cells/{args.plate}_Septins_outlier_cells.csv",
            "Misformed": f"{args.output_directory}/septin_fragmentation/outlier_cells/{args.plate}_Septins_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)
    
    print("Complete")