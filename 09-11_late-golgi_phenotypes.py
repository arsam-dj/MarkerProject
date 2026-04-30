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

# ============================== LG NUM. (TOO LOW/HIGH) ==============================

    tabulate_compartment_masks_per_strain(
        db_path=args.database_path,
        compartment_name="LGs",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_lg_count/lg_count_tables")

    comp_nums_table = generate_filtered_cell_feature_table(
        db_path=args.database_path,
        feature="Cell_Children_LGs_Count",
        comp_name="LGs"
    )

    # many LGs
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_LGs_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_lg_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_lg_count/many_lg/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_lg_count/many_lg/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_lg_count/many_lg/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_lg_count/many_lg/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_lg_count/many_lg/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LGs",
        feature_table=comp_nums_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # few LGs
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_LGs_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_lg_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_lg_count/few_lg/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_lg_count/few_lg/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_lg_count/few_lg/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_lg_count/few_lg/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_lg_count/few_lg/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LGs",
        feature_table=comp_nums_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== LG COVERAGE (TOO LOW/HIGH) ==============================

    lg_coverage_table = calculate_compartment_coverage(
        db_path=args.database_path,
        compartment_name="LGs",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_lg_coverage/coverage_tables")

    # high coverage
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Coverage",
        scaled_feature_dir=f"{args.output_directory}/abnormal_lg_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_lg_coverage/high_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_lg_coverage/high_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_lg_coverage/high_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_lg_coverage/high_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_lg_coverage/high_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LGs",
        feature_table=lg_coverage_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # low coverage
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Coverage",
        scaled_feature_dir=f"{args.output_directory}/abnormal_lg_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_lg_coverage/low_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_lg_coverage/low_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_lg_coverage/low_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_lg_coverage/low_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_lg_coverage/low_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LGs",
        feature_table=lg_coverage_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== LG STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).
    conn = sqlite3.connect(args.database_path)
    std_intensity_table = (
        pl
        .read_database(
            query=f"""
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
                        Cell_Intensity_StdIntensity_GFP
                    FROM Per_Cell
                    WHERE Cell_Children_LGs_Count != -1;
                  """,
            connection=conn
        )
    )
    conn.close()

    # low uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_lg_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_lg_uniformity/low_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_lg_uniformity/low_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_lg_uniformity/low_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_lg_uniformity/low_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_lg_uniformity/low_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LGs",
        feature_table=std_intensity_table,
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_lg_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_lg_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_lg_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_lg_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_lg_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_lg_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LGs",
        feature_table=std_intensity_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "ManyLGs": f"{args.output_directory}/abnormal_lg_count/many_lg/outlier_cells/{args.plate}_LGs_outlier_cells.csv",
            "FewLGs": f"{args.output_directory}/abnormal_lg_count/few_lg/outlier_cells/{args.plate}_LGs_outlier_cells.csv",
            "HighCoverage": f"{args.output_directory}/abnormal_lg_coverage/high_coverage/outlier_cells/{args.plate}_LGs_outlier_cells.csv",
            "LowCoverage": f"{args.output_directory}/abnormal_lg_coverage/low_coverage/outlier_cells/{args.plate}_LGs_outlier_cells.csv",
            "HighUniform": f"{args.output_directory}/abnormal_lg_uniformity/high_uniformity/outlier_cells/{args.plate}_LGs_outlier_cells.csv",
            "LowUniform": f"{args.output_directory}/abnormal_lg_uniformity/low_uniformity/outlier_cells/{args.plate}_LGs_outlier_cells.csv",
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


    print("Complete")
