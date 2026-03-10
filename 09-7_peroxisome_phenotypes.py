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
                                             generate_comp_size_table)

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

# ============================== PEROXISOME NUM. (TOO LOW/HIGH) ==============================

    tabulate_compartment_masks_per_strain(
        db_path=args.database_path,
        compartment_name="Peroxisomes",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_peroxisome_count/peroxisome_count_tables")

    # many peroxisomes
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_Peroxisomes_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_count/many_peroxisomes/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_count/many_peroxisomes/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_count/many_peroxisomes/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_count/many_peroxisomes/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_count/many_peroxisomes/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # few peroxisomes
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_Peroxisomes_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_count/few_peroxisomes/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_count/few_peroxisomes/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_count/few_peroxisomes/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_count/few_peroxisomes/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_count/few_peroxisomes/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== PEROXISOME COVERAGE (TOO LOW/HIGH) ==============================

    peroxisome_coverage_table = calculate_compartment_coverage(
        db_path=args.database_path,
        compartment_name="Peroxisomes",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_peroxisome_coverage/coverage_tables")

    # high coverage
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Coverage",
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/high_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/high_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/high_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/high_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/high_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table=peroxisome_coverage_table,
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/low_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/low_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/low_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/low_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_coverage/low_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table=peroxisome_coverage_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== PEROXISOME STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).

    # low uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/low_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/low_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/low_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/low_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/low_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== PEROXISOME SIZE (TOO SMALL/TOO LARGE) ==============================

    peroxisomes_size_table = generate_comp_size_table(
        db_path=args.database_path,
        comp_name="Peroxisomes")

    # large peroxisomes
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Peroxisomes_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_size/large_peroxisomes/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_size/large_peroxisomes/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_size/large_peroxisomes/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_size/large_peroxisomes/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_size/large_peroxisomes/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table=peroxisomes_size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # small peroxisomes
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Peroxisomes_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_size/small_peroxisomes/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_size/small_peroxisomes/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_size/small_peroxisomes/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_size/small_peroxisomes/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_size/small_peroxisomes/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table=peroxisomes_size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== PEROXISOME DISPERSION (SPREAD OUT/CLUSTERED) ==============================

    peroxisome_dispersion_table = (
        calculate_compartment_distances(
            db_path=args.database_path,
            compartment_name="Peroxisomes",
            plate=args.plate,
            output_directory=f"{args.output_directory}/abnormal_peroxisome_dispersion/dispersion_tables")
        .filter(pl.col("Num_Peroxisomes") > 2)  # no StDev for these
    )

    # high dispersion (spread out)
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Distance_StDev",
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/high_dispersion/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/high_dispersion/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/high_dispersion/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/high_dispersion/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/high_dispersion/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table=peroxisome_dispersion_table,
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/low_dispersion/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/low_dispersion/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/low_dispersion/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/low_dispersion/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_peroxisome_dispersion/low_dispersion/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Peroxisomes",
        feature_table=peroxisome_dispersion_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "ManyPeroxisomes": f"{args.output_directory}/abnormal_peroxisome_count/many_peroxisomes/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "FewPeroxisomes": f"{args.output_directory}/abnormal_peroxisome_count/few_peroxisomes/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "HighCoverage": f"{args.output_directory}/abnormal_peroxisome_coverage/high_coverage/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "LowCoverage": f"{args.output_directory}/abnormal_peroxisome_coverage/low_coverage/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "Dispersed": f"{args.output_directory}/abnormal_peroxisome_dispersion/high_dispersion/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "Clustered": f"{args.output_directory}/abnormal_peroxisome_dispersion/low_dispersion/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "HighUniform": f"{args.output_directory}/abnormal_peroxisome_uniformity/high_uniformity/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "LowUniform": f"{args.output_directory}/abnormal_peroxisome_uniformity/low_uniformity/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "LargePeroxisomes": f"{args.output_directory}/abnormal_peroxisome_size/large_peroxisomes/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv",
            "SmallPeroxisomes": f"{args.output_directory}/abnormal_peroxisome_size/small_peroxisomes/outlier_cells/{args.plate}_Peroxisomes_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


    print("Complete")
