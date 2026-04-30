import argparse
import math
import os
import polars as pl
from scipy.stats import norm
import sqlite3
from statistics import stdev, mean

from GEN_outlier_detection_functions import (scale_compartment_feature,
                                             identify_outlier_cells,
                                             calculate_strain_penetrances,
                                             tabulate_strain_cell_counts,
                                             get_strain_hits,
                                             run_all_functions,
                                             combine_output_phenotypes_from_plate,
                                             tabulate_compartment_masks_per_strain,
                                             calculate_compartment_coverage,
                                             calculate_compartment_distances,
                                             generate_comp_size_table,
                                             generate_filtered_cell_feature_table)

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

# ============================== EG NUM. (TOO LOW/HIGH) ==============================

    tabulate_compartment_masks_per_strain(
        db_path=args.database_path,
        compartment_name="EGs",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_eg_count/eg_count_tables")

    comp_nums_table = generate_filtered_cell_feature_table(
        db_path=args.database_path,
        feature="Cell_Children_EGs_Count",
        comp_name="EGs"
    )

    # many EGs
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_EGs_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eg_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eg_count/many_eg/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eg_count/many_eg/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eg_count/many_eg/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eg_count/many_eg/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eg_count/many_eg/per_well_wt_pens",
        plate=args.plate,
        compartment_name="EGs",
        feature_table=comp_nums_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # few EGs
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_EGs_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eg_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eg_count/few_eg/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eg_count/few_eg/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eg_count/few_eg/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eg_count/few_eg/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eg_count/few_eg/per_well_wt_pens",
        plate=args.plate,
        compartment_name="EGs",
        feature_table=comp_nums_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== EG COVERAGE (TOO LOW/HIGH) ==============================

    eg_coverage_table = calculate_compartment_coverage(
        db_path=args.database_path,
        compartment_name="EGs",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_eg_coverage/coverage_tables")

    # high coverage
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Coverage",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eg_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eg_coverage/high_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eg_coverage/high_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eg_coverage/high_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eg_coverage/high_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eg_coverage/high_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="EGs",
        feature_table=eg_coverage_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)


# ============================== EG STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).

    # low uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eg_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eg_uniformity/low_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eg_uniformity/low_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eg_uniformity/low_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eg_uniformity/low_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eg_uniformity/low_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="EGs",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # high uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eg_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eg_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eg_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eg_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eg_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eg_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="EGs",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "ManyEGs": f"{args.output_directory}/abnormal_eg_count/many_eg/outlier_cells/{args.plate}_EGs_outlier_cells.csv",
            "FewEGs": f"{args.output_directory}/abnormal_eg_count/few_eg/outlier_cells/{args.plate}_EGs_outlier_cells.csv",
            "HighCoverage": f"{args.output_directory}/abnormal_eg_coverage/high_coverage/outlier_cells/{args.plate}_EGs_outlier_cells.csv",
            "HighUniform": f"{args.output_directory}/abnormal_eg_uniformity/high_uniformity/outlier_cells/{args.plate}_EGs_outlier_cells.csv",
            "LowUniform": f"{args.output_directory}/abnormal_eg_uniformity/low_uniformity/outlier_cells/{args.plate}_EGs_outlier_cells.csv",
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


    print("Complete")
