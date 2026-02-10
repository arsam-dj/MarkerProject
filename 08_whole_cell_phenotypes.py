import argparse
import os
import polars as pl
from scipy import stats
import sqlite3
from statistics import stdev, mean

from GEN_outlier_detection_functions import (scale_compartment_feature,
                                             identify_outlier_cells,
                                             calculate_strain_penetrances,
                                             tabulate_strain_cell_counts,
                                             get_strain_hits,
                                             run_all_functions,
                                             combine_output_phenotypes_from_plate
                                             )

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

# ============================== CELL SIZE DEFECTS (TOO LARGE/SMALL) ==============================
    # large cells
    run_all_functions(
            db_path=args.database_path,
            all_cells=all_cells,
            compartment_table_name="Per_Cell",
            feature_name="Cell_AreaShape_Area",
            scaled_feature_dir=f"{args.output_directory}/abnormal_cell_size/scaled_features",
            outlier_objects_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_large_cells/outlier_cells",
            penetrance_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_large_cells/penetrances",
            cell_count_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_large_cells/cell_counts",
            strain_hits_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_large_cells/strain_hits",
            wt_pens_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_large_cells/per_well_wt_pens",
            plate=args.plate,
            compartment_name="Cell",
            excluded_outlier_cc_stages=[],
            feature_table="",
            cell_cycle_stages=["G1", "SG2", "MAT"],
            outlier_pval_cutoff=0.05,
            right_sided_outliers=True,
            percentile_cutoff=0.95)

    # small cells
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_cell_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_small_cells/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_small_cells/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_small_cells/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_small_cells/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_cell_size/abnormally_small_cells/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Cell",
        excluded_outlier_cc_stages=[],
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== CELL ELONGATION/APOLARITY DEFECTS ==============================
    # elongated cells
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_AreaShape_Eccentricity",
        scaled_feature_dir=f"{args.output_directory}/abnormal_cell_eccentricity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_elongated/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_elongated/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_elongated/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_elongated/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_elongated/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Cell",
        excluded_outlier_cc_stages=["G1"],  # G1 cells aren't expected to be significantly elongated
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # apolar cells
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_AreaShape_Eccentricity",
        scaled_feature_dir=f"{args.output_directory}/abnormal_cell_eccentricity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_round/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_round/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_round/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_round/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_round/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Cell",
        excluded_outlier_cc_stages=["SG2", "MAT"],  # round SG2/MAT cells would imply some kind of CC classification error
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "Small": f"{args.output_directory}/abnormal_cell_size/abnormally_small_cells/outlier_cells/{args.plate}_Cell_outlier_cells.csv",
            "Large": f"{args.output_directory}/abnormal_cell_size/abnormally_large_cells/outlier_cells/{args.plate}_Cell_outlier_cells.csv",
            "Elongated": f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_elongated/outlier_cells/{args.plate}_Cell_outlier_cells.csv",
            "Nonpolar": f"{args.output_directory}/abnormal_cell_eccentricity/abnormally_round/outlier_cells/{args.plate}_Cell_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


print("Complete.")