import argparse
import polars as pl
import sqlite3

from GEN_outlier_detection_functions import (run_all_functions,
                                             combine_output_phenotypes_from_plate,
                                             tabulate_compartment_masks_per_strain,
                                             calculate_compartment_coverage
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

# ============================== EISOSOME NUM. (TOO LOW/HIGH) ==============================

    tabulate_compartment_masks_per_strain(
        db_path=args.database_path,
        compartment_name="Eisosomes",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_eisosome_count/eisosome_count_tables")

    # many Eisosomes
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_Eisosomes_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eisosome_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eisosome_count/many_eisosome/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eisosome_count/many_eisosome/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eisosome_count/many_eisosome/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eisosome_count/many_eisosome/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eisosome_count/many_eisosome/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Eisosomes",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # few Eisosomes
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_Eisosomes_Count",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eisosome_count/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eisosome_count/few_eisosome/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eisosome_count/few_eisosome/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eisosome_count/few_eisosome/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eisosome_count/few_eisosome/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eisosome_count/few_eisosome/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Eisosomes",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== EISOSOME COVERAGE (TOO LOW/HIGH) ==============================

    eisosome_coverage_table = calculate_compartment_coverage(
        db_path=args.database_path,
        compartment_name="Eisosomes",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_eisosome_coverage/coverage_tables")

    # high coverage
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="Coverage",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eisosome_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eisosome_coverage/high_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eisosome_coverage/high_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eisosome_coverage/high_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eisosome_coverage/high_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eisosome_coverage/high_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Eisosomes",
        feature_table=eisosome_coverage_table,
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_eisosome_coverage/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eisosome_coverage/low_coverage/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eisosome_coverage/low_coverage/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eisosome_coverage/low_coverage/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eisosome_coverage/low_coverage/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eisosome_coverage/low_coverage/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Eisosomes",
        feature_table=eisosome_coverage_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== EISOSOME STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).

    # low uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/low_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/low_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/low_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/low_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/low_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Eisosomes",
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_eisosome_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Eisosomes",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "ManyEisosomes": f"{args.output_directory}/abnormal_eisosome_count/many_eisosome/outlier_cells/{args.plate}_Eisosomes_outlier_cells.csv",
            "FewEisosomes": f"{args.output_directory}/abnormal_eisosome_count/few_eisosome/outlier_cells/{args.plate}_Eisosomes_outlier_cells.csv",
            "HighCoverage": f"{args.output_directory}/abnormal_eisosome_coverage/high_coverage/outlier_cells/{args.plate}_Eisosomes_outlier_cells.csv",
            "LowCoverage": f"{args.output_directory}/abnormal_eisosome_coverage/low_coverage/outlier_cells/{args.plate}_Eisosomes_outlier_cells.csv",
            "HighUniform": f"{args.output_directory}/abnormal_eisosome_uniformity/high_uniformity/outlier_cells/{args.plate}_Eisosomes_outlier_cells.csv",
            "LowUniform": f"{args.output_directory}/abnormal_eisosome_uniformity/low_uniformity/outlier_cells/{args.plate}_Eisosomes_outlier_cells.csv",
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


    print("Complete")
