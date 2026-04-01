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
                                             tabulate_compartment_masks_per_strain)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save phenotype information.')
parser.add_argument('-p', '--plate', default='', help='Plate identifier for saving files.')

args = parser.parse_args()


def calculate_compartment_coverage(db_path, compartment_name, plate, output_directory):
    """
    Compartment coverage is the % of cell area that is covered by child compartments. It is the ratio of
    <compartment_name>_AreaShape_Area to Cell_AreaShape_Area.

    Args:
        db_path (str): path to database with compartment and cell information
        compartment_name (str): name of compartment being analyzed, should match what's in columns
        plate (str): plate identifier for writing output file
        output_directory (str): where to write output table

    Returns:
        pl.DataFrame with coverage values for every cell
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    conn = sqlite3.connect(db_path)
    coverage = (
        pl
        .read_database(
            query=f"""
                    WITH total_comp_area AS (
	                    SELECT
	                    	Cell_ID,
	                    	SUM({compartment_name}_AreaShape_Area) AS {compartment_name}_AreaShape_Area
	                    FROM Per_{compartment_name}
	                    WHERE Is_Nuclear = 'T'
	                    GROUP BY Cell_ID
	                    )

                    SELECT
                    	Replicate,
                    	Condition,
                    	Row,
                    	Column,
                    	ORF,
                    	Name,
                    	Strain_ID,
                    	Predicted_Label,
                    	Per_Cell.Cell_ID,
                    	(CAST({compartment_name}_AreaShape_Area AS NUMERIC) / Cell_AreaShape_Area) AS Coverage
                    FROM Per_Cell
                    JOIN total_comp_area
                    ON total_comp_area.Cell_ID = Per_Cell.Cell_ID;
                   """,
            connection=conn
        )
    )
    conn.close()

    coverage.write_csv(f"{output_directory}/{plate}_{compartment_name}_coverage.csv")

    return coverage


def generate_comp_size_table(db_path, comp_name):
    """
    Creates a table with cell/comp info and <comp>_AreaShape_Area for all comps so they can be scaled later.

    Args:
        db_path (str): path to database with compartment and cell information
        comp_name (str): name of compartment for selecting compartment table

    Returns:
        pl.DataFrame with <comp>_AreaShape_Area feature for all comps
    """
    conn = sqlite3.connect(db_path)
    all_comp_areas = (
        pl
        .read_database(
            query=f"""SELECT 
                        Replicate, 
                        Condition, 
                        Row, 
                        Column, 
                        Per_{comp_name}.Cell_ID,
                        {comp_name}_Number_Object_Number, 
                        ORF, 
                        Name, 
                        Strain_ID, 
                        Predicted_Label, 
                        {comp_name}_AreaShape_Area 
                      FROM Per_{comp_name}
                      JOIN (SELECT Cell_ID, Predicted_Label FROM Per_Cell) pc
                        ON Per_{comp_name}.Cell_ID = pc.Cell_ID
                      WHERE Is_Nuclear = 'T';
                   """,
            connection=conn
        )
    )
    conn.close()

    return all_comp_areas


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
        compartment_name="NuclearLDs",
        plate=args.plate,
        output_directory=f"{args.output_directory}/abnormal_ld_count/ld_count_tables")

    # many LDs
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Children_NuclearLDs_Count",
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
        feature_name="Cell_Children_NuclearLDs_Count",
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


# ============================== LD SIZE (TOO SMALL/TOO LARGE) ==============================
    ld_size_table = generate_comp_size_table(
        db_path=args.database_path,
        comp_name="LDs")

    # large LD
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="LDs_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_size/large_ld/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table=ld_size_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # small LD
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="",
        feature_name="LDs_AreaShape_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_ld_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_ld_size/small_ld/per_well_wt_pens",
        plate=args.plate,
        compartment_name="LDs",
        feature_table=ld_size_table,
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
            "LargeLDs": f"{args.output_directory}/abnormal_ld_size/large_ld/outlier_cells/{args.plate}_LDs_outlier_cells.csv",
            "SmallLDs": f"{args.output_directory}/abnormal_ld_size/small_ld/outlier_cells/{args.plate}_LDs_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)

    print("Complete")
