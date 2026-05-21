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


def nuclear_cell_ratio_table(db_path):
    """
    Get table with combined nuclear over cell area ratios for all cells.
    
    Args:
        db_path (str): path to database
    """

    conn = sqlite3.connect(db_path)

    ratios = pl.read_database(
        query=f"""
                SELECT 
                    Replicate, 
                    Condition, 
                    Row, 
                    Column, 
                    Per_Nuclei.Cell_ID,
                    ORF, 
                    Name, 
                    Strain_ID, 
                    Predicted_Label, 
                    Nuclei_Number_Object_Number,
                    Nuclei_AreaShape_Area / Cell_AreaShape_Area AS Nuclear_Area_Over_Cell_Area
                FROM Per_Nuclei
                JOIN (SELECT Cell_ID, Cell_AreaShape_Area, Predicted_Label FROM Per_Cell) pc
                    ON Per_Nuclei.Cell_ID = pc.Cell_ID;
                """,
        connection=conn)

    conn.close()

    return ratios


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

# ============================== NUCLEAR SIZE RELATIVE TO CELL SIZE (TOO HIGH/TOO LOW) ==============================

    nuclear_cell_ratios = nuclear_cell_ratio_table(db_path=args.database_path)

    # large size
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Nuclei",
        feature_name="Nuclear_Area_Over_Cell_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_nuclear_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_nuclear_size/large_size/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_nuclear_size/large_size/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_nuclear_size/large_size/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_nuclear_size/large_size/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_nuclear_size/large_size/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Nuclei",
        feature_table=nuclear_cell_ratios,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)

    # large size
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Nuclei",
        feature_name="Nuclear_Area_Over_Cell_Area",
        scaled_feature_dir=f"{args.output_directory}/abnormal_nuclear_size/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_nuclear_size/small_size/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_nuclear_size/small_size/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_nuclear_size/small_size/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_nuclear_size/small_size/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_nuclear_size/small_size/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Nuclei",
        feature_table=nuclear_cell_ratios,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== NUCLEUS STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # The feature StdIntensity looks at the overall uniformity of the RFP channel within each cell.
    # Higher StdIntensity suggests lower uniformity (e.g., there is greater pixel-to-pixel intensity deviation).

    # high uniformity
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_RFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_nuclear_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_nuclear_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_nuclear_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_nuclear_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_nuclear_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_nuclear_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Nuclei",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== ABNORMAL NUCLEAR SOLIDITY ==============================
    solidity_feature_table = generate_compartment_feature_table(
        db_path=args.database_path,
        feature="Nuclei_AreaShape_Solidity",
        comp_name="Nuclei")

    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Nuclei",
        feature_name="Nuclei_AreaShape_Solidity",
        scaled_feature_dir=f"{args.output_directory}/abnormal_nuclear_solidity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_nuclear_solidity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_nuclear_solidity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_nuclear_solidity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_nuclear_solidity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_nuclear_solidity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Nuclei",
        feature_table=solidity_feature_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.01,
        right_sided_outliers=False,
        percentile_cutoff=0.95)


# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "LargeNuclei": f"{args.output_directory}/abnormal_nuclear_size/large_size/outlier_cells/{args.plate}_Nuclei_outlier_cells.csv",
            "SmallNuclei": f"{args.output_directory}/abnormal_nuclear_size/small_size/outlier_cells/{args.plate}_Nuclei_outlier_cells.csv",
            "HighUniform": f"{args.output_directory}/abnormal_nuclear_uniformity/high_uniformity/outlier_cells/{args.plate}_Nuclei_outlier_cells.csv",
            "LowSolidity": f"{args.output_directory}/abnormal_nuclear_solidity/outlier_cells/{args.plate}_Nuclei_outlier_cells.csv"
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


    print("Complete")
