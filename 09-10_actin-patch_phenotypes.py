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
                                             combine_FracAtD_rings)

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

## ============================== AP NUM. (TOO LOW/HIGH) ==============================
#
#    tabulate_compartment_masks_per_strain(
#        db_path=args.database_path,
#        compartment_name="APs",
#        plate=args.plate,
#        output_directory=f"{args.output_directory}/abnormal_ap_count/ap_count_tables")
#
#    # many APs
#    run_all_functions(
#        db_path=args.database_path,
#        all_cells=all_cells,
#        compartment_table_name="Per_Cell",
#        feature_name="Cell_Children_APs_Count",
#        scaled_feature_dir=f"{args.output_directory}/abnormal_ap_count/scaled_features",
#        outlier_objects_dir=f"{args.output_directory}/abnormal_ap_count/many_ap/outlier_cells",
#        penetrance_dir=f"{args.output_directory}/abnormal_ap_count/many_ap/penetrances",
#        cell_count_dir=f"{args.output_directory}/abnormal_ap_count/many_ap/cell_counts",
#        strain_hits_dir=f"{args.output_directory}/abnormal_ap_count/many_ap/strain_hits",
#        wt_pens_dir=f"{args.output_directory}/abnormal_ap_count/many_ap/per_well_wt_pens",
#        plate=args.plate,
#        compartment_name="APs",
#        feature_table="",
#        cell_cycle_stages=["G1", "SG2", "MAT"],
#        outlier_pval_cutoff=0.05,
#        right_sided_outliers=True,
#        percentile_cutoff=0.95)
#
#    # few APs
#    run_all_functions(
#        db_path=args.database_path,
#        all_cells=all_cells,
#        compartment_table_name="Per_Cell",
#        feature_name="Cell_Children_APs_Count",
#        scaled_feature_dir=f"{args.output_directory}/abnormal_ap_count/scaled_features",
#        outlier_objects_dir=f"{args.output_directory}/abnormal_ap_count/few_ap/outlier_cells",
#        penetrance_dir=f"{args.output_directory}/abnormal_ap_count/few_ap/penetrances",
#        cell_count_dir=f"{args.output_directory}/abnormal_ap_count/few_ap/cell_counts",
#        strain_hits_dir=f"{args.output_directory}/abnormal_ap_count/few_ap/strain_hits",
#        wt_pens_dir=f"{args.output_directory}/abnormal_ap_count/few_ap/per_well_wt_pens",
#        plate=args.plate,
#        compartment_name="APs",
#        feature_table="",
#        cell_cycle_stages=["G1", "SG2", "MAT"],
#        outlier_pval_cutoff=0.05,
#        right_sided_outliers=False,
#        percentile_cutoff=0.95)


## ============================== AP COVERAGE (TOO LOW/HIGH) ==============================
#
#    ap_coverage_table = calculate_compartment_coverage(
#        db_path=args.database_path,
#        compartment_name="APs",
#        plate=args.plate,
#        output_directory=f"{args.output_directory}/abnormal_ap_coverage/coverage_tables")
#
#    # high coverage
#    run_all_functions(
#        db_path=args.database_path,
#        all_cells=all_cells,
#        compartment_table_name="",
#        feature_name="Coverage",
#        scaled_feature_dir=f"{args.output_directory}/abnormal_ap_coverage/scaled_features",
#        outlier_objects_dir=f"{args.output_directory}/abnormal_ap_coverage/high_coverage/outlier_cells",
#        penetrance_dir=f"{args.output_directory}/abnormal_ap_coverage/high_coverage/penetrances",
#        cell_count_dir=f"{args.output_directory}/abnormal_ap_coverage/high_coverage/cell_counts",
#        strain_hits_dir=f"{args.output_directory}/abnormal_ap_coverage/high_coverage/strain_hits",
#        wt_pens_dir=f"{args.output_directory}/abnormal_ap_coverage/high_coverage/per_well_wt_pens",
#        plate=args.plate,
#        compartment_name="APs",
#        feature_table=ap_coverage_table,
#        cell_cycle_stages=["G1", "SG2", "MAT"],
#        outlier_pval_cutoff=0.05,
#        right_sided_outliers=True,
#        percentile_cutoff=0.95)
#
#    # low coverage
#    run_all_functions(
#        db_path=args.database_path,
#        all_cells=all_cells,
#        compartment_table_name="",
#        feature_name="Coverage",
#        scaled_feature_dir=f"{args.output_directory}/abnormal_ap_coverage/scaled_features",
#        outlier_objects_dir=f"{args.output_directory}/abnormal_ap_coverage/low_coverage/outlier_cells",
#        penetrance_dir=f"{args.output_directory}/abnormal_ap_coverage/low_coverage/penetrances",
#        cell_count_dir=f"{args.output_directory}/abnormal_ap_coverage/low_coverage/cell_counts",
#        strain_hits_dir=f"{args.output_directory}/abnormal_ap_coverage/low_coverage/strain_hits",
#        wt_pens_dir=f"{args.output_directory}/abnormal_ap_coverage/low_coverage/per_well_wt_pens",
#        plate=args.plate,
#        compartment_name="APs",
#        feature_table=ap_coverage_table,
#        cell_cycle_stages=["G1", "SG2", "MAT"],
#        outlier_pval_cutoff=0.05,
#        right_sided_outliers=False,
#        percentile_cutoff=0.95)


## ============================== AP DISPERSION (SPREAD OUT/CLUSTERED) ==============================
#
#    ap_dispersion_table = (
#        calculate_compartment_distances(
#            db_path=args.database_path,
#            compartment_name="APs",
#            plate=args.plate,
#            output_directory=f"{args.output_directory}/abnormal_ap_dispersion/dispersion_tables")
#        .filter(pl.col("Num_APs") > 2)  # no StDev for these
#    )
#
#    # high dispersion (spread out)
#    run_all_functions(
#        db_path=args.database_path,
#        all_cells=all_cells,
#        compartment_table_name="",
#        feature_name="Distance_StDev",
#        scaled_feature_dir=f"{args.output_directory}/abnormal_ap_dispersion/scaled_features",
#        outlier_objects_dir=f"{args.output_directory}/abnormal_ap_dispersion/high_dispersion/outlier_cells",
#        penetrance_dir=f"{args.output_directory}/abnormal_ap_dispersion/high_dispersion/penetrances",
#        cell_count_dir=f"{args.output_directory}/abnormal_ap_dispersion/high_dispersion/cell_counts",
#        strain_hits_dir=f"{args.output_directory}/abnormal_ap_dispersion/high_dispersion/strain_hits",
#        wt_pens_dir=f"{args.output_directory}/abnormal_ap_dispersion/high_dispersion/per_well_wt_pens",
#        plate=args.plate,
#        compartment_name="APs",
#        feature_table=ap_dispersion_table,
#        cell_cycle_stages=["G1", "SG2", "MAT"],
#        outlier_pval_cutoff=0.05,
#        right_sided_outliers=True,
#        percentile_cutoff=0.95)
#
#    # low dispersion (clustered/aggregated)
#    run_all_functions(
#        db_path=args.database_path,
#        all_cells=all_cells,
#        compartment_table_name="",
#        feature_name="Distance_StDev",
#        scaled_feature_dir=f"{args.output_directory}/abnormal_ap_dispersion/scaled_features",
#        outlier_objects_dir=f"{args.output_directory}/abnormal_ap_dispersion/low_dispersion/outlier_cells",
#        penetrance_dir=f"{args.output_directory}/abnormal_ap_dispersion/low_dispersion/penetrances",
#        cell_count_dir=f"{args.output_directory}/abnormal_ap_dispersion/low_dispersion/cell_counts",
#        strain_hits_dir=f"{args.output_directory}/abnormal_ap_dispersion/low_dispersion/strain_hits",
#        wt_pens_dir=f"{args.output_directory}/abnormal_ap_dispersion/low_dispersion/per_well_wt_pens",
#        plate=args.plate,
#        compartment_name="APs",
#        feature_table=ap_dispersion_table,
#        cell_cycle_stages=["G1", "SG2", "MAT"],
#        outlier_pval_cutoff=0.05,
#        right_sided_outliers=False,
#        percentile_cutoff=0.95)


## ============================== AP STD INTENSITY (TOO HIGH/TOO LOW) ==============================
#    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
#    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).
#
#    # low uniformity
#    run_all_functions(
#        db_path=args.database_path,
#        all_cells=all_cells,
#        compartment_table_name="Per_Cell",
#        feature_name="Cell_Intensity_StdIntensity_GFP",
#        scaled_feature_dir=f"{args.output_directory}/abnormal_ap_uniformity/scaled_features",
#        outlier_objects_dir=f"{args.output_directory}/abnormal_ap_uniformity/low_uniformity/outlier_cells",
#        penetrance_dir=f"{args.output_directory}/abnormal_ap_uniformity/low_uniformity/penetrances",
#        cell_count_dir=f"{args.output_directory}/abnormal_ap_uniformity/low_uniformity/cell_counts",
#        strain_hits_dir=f"{args.output_directory}/abnormal_ap_uniformity/low_uniformity/strain_hits",
#        wt_pens_dir=f"{args.output_directory}/abnormal_ap_uniformity/low_uniformity/per_well_wt_pens",
#        plate=args.plate,
#        compartment_name="APs",
#        feature_table="",
#        cell_cycle_stages=["G1", "SG2", "MAT"],
#        outlier_pval_cutoff=0.05,
#        right_sided_outliers=True,
#        percentile_cutoff=0.95)
#
#    # high uniformity
#    run_all_functions(
#        db_path=args.database_path,
#        all_cells=all_cells,
#        compartment_table_name="Per_Cell",
#        feature_name="Cell_Intensity_StdIntensity_GFP",
#        scaled_feature_dir=f"{args.output_directory}/abnormal_ap_uniformity/scaled_features",
#        outlier_objects_dir=f"{args.output_directory}/abnormal_ap_uniformity/high_uniformity/outlier_cells",
#        penetrance_dir=f"{args.output_directory}/abnormal_ap_uniformity/high_uniformity/penetrances",
#        cell_count_dir=f"{args.output_directory}/abnormal_ap_uniformity/high_uniformity/cell_counts",
#        strain_hits_dir=f"{args.output_directory}/abnormal_ap_uniformity/high_uniformity/strain_hits",
#        wt_pens_dir=f"{args.output_directory}/abnormal_ap_uniformity/high_uniformity/per_well_wt_pens",
#        plate=args.plate,
#        compartment_name="APs",
#        feature_table="",
#        cell_cycle_stages=["G1", "SG2", "MAT"],
#        outlier_pval_cutoff=0.05,
#        right_sided_outliers=False,
#        percentile_cutoff=0.95)


# ============================== NO AP LOCALIZATION TO BUD ==============================
    # In late G1/early SG2 when the bud forms, APs begin largely localizing there. Sometimes,
    # this may not happen. The feature MassDisplacement looks at where most of the GFP signal
    # is relative to the cell's center. The higher it is, the closer the signal is to the edges.

    # high mass displacement
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_MassDisplacement_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_bud_localization/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_bud_localization/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_bud_localization/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_bud_localization/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_bud_localization/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_bud_localization/per_well_wt_pens",
        plate=args.plate,
        compartment_name="APs",
        excluded_outlier_cc_stages=["G1", "MAT"], # G1s typically don't have a bud and MAT is past the bud formation stage
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== ABNORMAL RADIAL AP DISTRIBUTIONS ==============================
    # The features FracAtD_Nof4 describe the % of signal coming from each of the four radial rings that every
    # object is divided into, going from the center to the periphery. For every cell, all FracAtD features add
    # up to 1. To simplify this process, the inner two and outer 2 rings will be combined to give inner distribution
    # and outer distribution

    simplified_FracAtD_table = combine_FracAtD_rings(
        db_path=args.database_path,
        compartment_name="APs",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_radial_distribution/simplified_FracAtD_tables")


    # high distribution at center
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Inner_Distribution",
        scaled_feature_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/high_inner_distribution/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/high_inner_distribution/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/high_inner_distribution/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/high_inner_distribution/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/high_inner_distribution/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/high_inner_distribution/per_well_wt_pens",
        plate=args.plate,
        compartment_name="APs",
        excluded_outlier_cc_stages=[],
        feature_table=simplified_FracAtD_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # low distribution at center
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Inner_Distribution",
        scaled_feature_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/low_inner_distribution/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/low_inner_distribution/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/low_inner_distribution/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/low_inner_distribution/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/low_inner_distribution/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_inner_distribution/low_inner_distribution/per_well_wt_pens",
        plate=args.plate,
        compartment_name="APs",
        excluded_outlier_cc_stages=[],
        feature_table=simplified_FracAtD_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)

    # high distribution at periphery
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Outer_Distribution",
        scaled_feature_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/high_outer_distribution/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/high_outer_distribution/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/high_outer_distribution/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/high_outer_distribution/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/high_outer_distribution/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/high_outer_distribution/per_well_wt_pens",
        plate=args.plate,
        compartment_name="APs",
        excluded_outlier_cc_stages=[],
        feature_table=simplified_FracAtD_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # low distribution at center
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Outer_Distribution",
        scaled_feature_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/low_outer_distribution/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/low_outer_distribution/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/low_outer_distribution/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/low_outer_distribution/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/low_outer_distribution/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_radial_distribution/abnormal_outer_distribution/low_outer_distribution/per_well_wt_pens",
        plate=args.plate,
        compartment_name="APs",
        excluded_outlier_cc_stages=[],
        feature_table=simplified_FracAtD_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)

    print("Complete")
