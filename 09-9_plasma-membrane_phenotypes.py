import argparse
import math
import os
import polars as pl
import sqlite3

from GEN_outlier_detection_functions import (run_all_functions,
                                             combine_output_phenotypes_from_plate,
                                             generate_iint_norm_table)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save phenotype information.')
parser.add_argument('-p', '--plate', default='', help='Plate identifier for saving files.')

args = parser.parse_args()


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
    

# ============================== DIM/BRIGHT PM ==============================
    iint_norm_table = generate_iint_norm_table(db_path=args.database_path)

    # bright PM
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="IInt_Norm",
        scaled_feature_dir=f"{args.output_directory}/abnormal_pm_intensity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_pm_intensity/bright_pm/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_pm_intensity/bright_pm/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_pm_intensity/bright_pm/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_pm_intensity/bright_pm/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_pm_intensity/bright_pm/per_well_wt_pens",
        plate=args.plate,
        compartment_name="PM",
        feature_table=iint_norm_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # dim PM
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="IInt_Norm",
        scaled_feature_dir=f"{args.output_directory}/abnormal_pm_intensity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_pm_intensity/dim_pm/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_pm_intensity/dim_pm/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_pm_intensity/dim_pm/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_pm_intensity/dim_pm/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_pm_intensity/dim_pm/per_well_wt_pens",
        plate=args.plate,
        compartment_name="PM",
        feature_table=iint_norm_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== PM STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).

    # low uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_pm_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_pm_uniformity/low_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_pm_uniformity/low_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_pm_uniformity/low_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_pm_uniformity/low_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_pm_uniformity/low_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="PM",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "DimPM": f"{args.output_directory}/abnormal_pm_intensity/dim_pm/outlier_cells/{args.plate}_PM_outlier_cells.csv",
            "BrightPM": f"{args.output_directory}/abnormal_pm_intensity/bright_pm/outlier_cells/{args.plate}_PM_outlier_cells.csv",
            "LowUniform": f"{args.output_directory}/abnormal_pm_uniformity/low_uniformity/outlier_cells/{args.plate}_PM_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)

    print("Complete")
