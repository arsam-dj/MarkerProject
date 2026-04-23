import argparse
import math
import os
import polars as pl
import sqlite3

from GEN_outlier_detection_functions import (run_all_functions,
                                             combine_output_phenotypes_from_plate,
                                             generate_iint_norm_table,
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
    

# ============================== DIM/BRIGHT ER ==============================
    iint_norm_table = generate_iint_norm_table(db_path=args.database_path)

    # bright ER
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="IInt_Norm",
        scaled_feature_dir=f"{args.output_directory}/abnormal_er_intensity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_er_intensity/bright_er/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_er_intensity/bright_er/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_er_intensity/bright_er/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_er_intensity/bright_er/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_er_intensity/bright_er/per_well_wt_pens",
        plate=args.plate,
        compartment_name="ER",
        feature_table=iint_norm_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # dim ER
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="IInt_Norm",
        scaled_feature_dir=f"{args.output_directory}/abnormal_er_intensity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_er_intensity/dim_er/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_er_intensity/dim_er/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_er_intensity/dim_er/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_er_intensity/dim_er/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_er_intensity/dim_er/per_well_wt_pens",
        plate=args.plate,
        compartment_name="ER",
        feature_table=iint_norm_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== ER STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the GFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).

    # high uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_er_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_er_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_er_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_er_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_er_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_er_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="ER",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)
    
    # low uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_er_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_er_uniformity/low_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_er_uniformity/low_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_er_uniformity/low_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_er_uniformity/low_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_er_uniformity/low_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="ER",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)


# ============================== ER MASS DISPLACEMENT ==============================

    # high MD
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_MassDisplacement_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_er_mass/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_er_mass/high_mass_displacement/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_er_mass/high_mass_displacement/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_er_mass/high_mass_displacement/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_er_mass/high_mass_displacement/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_er_mass/high_mass_displacement/per_well_wt_pens",
        plate=args.plate,
        compartment_name="ER",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)
    

# ============================== ER RADIAL INTENSITY DISTRIBUTION ==============================

    fracD_table = combine_FracAtD_rings(
        db_path=args.database_path,
        compartment_name="ER",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_er_fracD/combined_fracD_tables"
    )

    # high inner distribution
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Inner_Distribution",
        scaled_feature_dir=f"{args.output_directory}/abnormal_er_fracD/higher_inner_distribution/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_er_fracD/higher_inner_distribution/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_er_fracD/higher_inner_distribution/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_er_fracD/higher_inner_distribution/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_er_fracD/higher_inner_distribution/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_er_fracD/higher_inner_distribution/per_well_wt_pens",
        plate=args.plate,
        compartment_name="ER",
        feature_table=fracD_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)
    
    # high outer distribution
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Outer_Distribution",
        scaled_feature_dir=f"{args.output_directory}/abnormal_er_fracD/higher_outer_distribution/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_er_fracD/higher_outer_distribution/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_er_fracD/higher_outer_distribution/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_er_fracD/higher_outer_distribution/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_er_fracD/higher_outer_distribution/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_er_fracD/higher_outer_distribution/per_well_wt_pens",
        plate=args.plate,
        compartment_name="ER",
        feature_table=fracD_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)
    

# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "DimER": f"{args.output_directory}/abnormal_er_intensity/dim_er/outlier_cells/{args.plate}_ER_outlier_cells.csv",
            "BrightER": f"{args.output_directory}/abnormal_er_intensity/bright_er/outlier_cells/{args.plate}_ER_outlier_cells.csv",
            "LowUniform": f"{args.output_directory}/abnormal_er_uniformity/low_uniformity/outlier_cells/{args.plate}_ER_outlier_cells.csv",
            "HighUniform": f"{args.output_directory}/abnormal_er_uniformity/high_uniformity/outlier_cells/{args.plate}_ER_outlier_cells.csv",
            "HighMD": f"{args.output_directory}/abnormal_er_mass/high_mass_displacement/outlier_cells/{args.plate}_ER_outlier_cells.csv",
            "HighInnerFracD": f"{args.output_directory}/abnormal_er_fracD/higher_inner_distribution/outlier_cells/{args.plate}_ER_outlier_cells.csv",
            "HighOuterFracD": f"{args.output_directory}/abnormal_er_fracD/higher_outer_distribution/outlier_cells/{args.plate}_ER_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)

    print("Complete")
