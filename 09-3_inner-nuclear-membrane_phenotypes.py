import argparse
import polars as pl
import sqlite3

from GEN_outlier_detection_functions import (run_all_functions,
                                             combine_output_phenotypes_from_plate,
                                             generate_compartment_feature_table
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

# ============================== INM STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).

    # low uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_inm_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_inm_uniformity/low_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_inm_uniformity/low_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_inm_uniformity/low_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_inm_uniformity/low_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_inm_uniformity/low_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="INM",
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
        scaled_feature_dir=f"{args.output_directory}/abnormal_inm_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_inm_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_inm_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_inm_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_inm_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_inm_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="INM",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== ABNORMAL INM FORM FACTOR ==============================
    ff_feature_table = generate_compartment_feature_table(
        db_path=args.database_path,
        feature="INM_AreaShape_FormFactor",
        comp_name="INM")

    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_INM",
        feature_name="INM_AreaShape_FormFactor",
        scaled_feature_dir=f"{args.output_directory}/abnormal_inm_form_factor/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_inm_form_factor/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_inm_form_factor/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_inm_form_factor/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_inm_form_factor/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_inm_form_factor/per_well_wt_pens",
        plate=args.plate,
        compartment_name="INM",
        feature_table=ff_feature_table,
        cell_cycle_stages=["SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== ABNORMAL INM SOLIDITY ==============================
    solidity_feature_table = generate_compartment_feature_table(
        db_path=args.database_path,
        feature="INM_AreaShape_Solidity",
        comp_name="INM")

    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_INM",
        feature_name="INM_AreaShape_Solidity",
        scaled_feature_dir=f"{args.output_directory}/abnormal_inm_solidity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_inm_solidity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_inm_solidity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_inm_solidity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_inm_solidity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_inm_solidity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="INM",
        feature_table=solidity_feature_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "HighUniform": f"{args.output_directory}/abnormal_inm_uniformity/high_uniformity/outlier_cells/{args.plate}_INM_outlier_cells.csv",
            "LowUniform": f"{args.output_directory}/abnormal_inm_uniformity/low_uniformity/outlier_cells/{args.plate}_INM_outlier_cells.csv",
            "LowFF": f"{args.output_directory}/abnormal_inm_form_factor/outlier_cells/{args.plate}_INM_outlier_cells.csv",
            "LowSolidity": f"{args.output_directory}/abnormal_inm_solidity/outlier_cells/{args.plate}_INM_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


    print("Complete")
