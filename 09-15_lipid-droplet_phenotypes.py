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
                                             get_shape_outliers_for_multi_foci_comps)

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

# ============================== LD NUM. (TOO LOW/HIGH) ==============================

    tabulate_compartment_masks_per_strain(
        db_path=args.database_path,
        compartment_name="LDs",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_ld_count/ld_count_tables")

    # many LDs
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_LDs_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_count/many_ld/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_count/many_ld/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_count/many_ld/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_count/many_ld/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_count/many_ld/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # few LDs
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_LDs_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_count/few_ld/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_count/few_ld/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_count/few_ld/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_count/few_ld/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_count/few_ld/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== LD COVERAGE (TOO LOW/HIGH) ==============================

    ld_coverage_table = calculate_compartment_coverage(
        db_path=args.database_path,
        compartment_name="LDs",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_ld_coverage/coverage_tables")

    # high coverage
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Coverage",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_coverage/high_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_coverage/high_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_coverage/high_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_coverage/high_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_coverage/high_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table=ld_coverage_table,
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_coverage/low_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_coverage/low_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_coverage/low_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_coverage/low_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_coverage/low_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table=ld_coverage_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== LD STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).

    # low uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_uniformity/low_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_uniformity/low_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_uniformity/low_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_uniformity/low_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_uniformity/low_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== LD SIZE (TOO SMALL/TOO LARGE) ==============================
    # Identifying cells whose subcellular compartments have abnormal sizes isn't trivial in multi-foci
    # compartments where there are many foci of varying sizes (hence noisy). Therefore, first I identify
    # individual compartments notably large or small relative to WT sizes, and then I determine the % of
    # compartments in every cell that have an abnormal size. I use the % of abnormally sized cells in WT to
    # identify mutants with lots of small or large subcellular compartments. To prevent inflated % values,
    # I will only consider cells with at least 4 subcellular compartments.

    # large LDs
    get_shape_outliers_for_multi_foci_comps(
        db_path=args.database_path,
        compartment_name="LDs",
        feature_name="AreaShape_Area",
        plate=args.plate,
        proportions_dir=f"{args.output_directory}/abnormal_ld_size/outlier_compartment_proportions",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/per_well_wt_pens",
        all_cells=all_cells,
        pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # small LDs
    get_shape_outliers_for_multi_foci_comps(
        db_path=args.database_path,
        compartment_name="LDs",
        feature_name="AreaShape_Area",
        plate=args.plate,
        proportions_dir=f"{args.output_directory}/abnormal_ld_size/outlier_compartment_proportions",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/per_well_wt_pens",
        all_cells=all_cells,
        pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== LD DISPERSION (SPREAD OUT/CLUSTERED) ==============================

    ld_dispersion_table = (
        calculate_compartment_distances(
            db_path=args.database_path,
            compartment_name="LDs",
            plate=args.plate,
            output_directory=f"{args.output_directory}/abnormal_ld_dispersion/dispersion_tables")
        .filter(pl.col("Num_LDs") > 2)  # no StDev for these
    )

    # high dispersion (spread out)
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Distance_StDev",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_dispersion/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_dispersion/high_dispersion/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_dispersion/high_dispersion/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_dispersion/high_dispersion/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_dispersion/high_dispersion/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_dispersion/high_dispersion/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table=ld_dispersion_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # low dispersion (clustered/aggregated)
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Distance_StDev",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_dispersion/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_dispersion/low_dispersion/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_dispersion/low_dispersion/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_dispersion/low_dispersion/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_dispersion/low_dispersion/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_dispersion/low_dispersion/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table=ld_dispersion_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "ManyLDs": f"{args.output_directory}/abnormal_ld_count/many_ld/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "FewLDs": f"{args.output_directory}/abnormal_ld_count/few_ld/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "HighCoverage": f"{args.output_directory}/abnormal_ld_coverage/high_coverage/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "LowCoverage": f"{args.output_directory}/abnormal_ld_coverage/low_coverage/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "Dispersed": f"{args.output_directory}/abnormal_ld_dispersion/high_dispersion/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "Clustered": f"{args.output_directory}/abnormal_ld_dispersion/low_dispersion/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "HighUniform": f"{args.output_directory}/abnormal_ld_uniformity/high_uniformity/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "LowUniform": f"{args.output_directory}/abnormal_ld_uniformity/low_uniformity/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "LargeLDs": f"{args.output_directory}/abnormal_ld_size/large_ld/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "SmallLDs": f"{args.output_directory}/abnormal_ld_size/small_ld/outlier_cells/{args.plate}_LDs_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


    print("Complete")
