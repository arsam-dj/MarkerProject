import argparse
import os
import polars as pl
import sqlite3

from GEN_outlier_detection_functions import (calculate_strain_penetrances,
                                             tabulate_strain_cell_counts,
                                             get_strain_hits,
                                             run_all_functions,
                                             combine_output_phenotypes_from_plate)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save phenotype information.')
parser.add_argument('-p', '--plate', default='', help='Plate identifier for saving files.')

args = parser.parse_args()

def create_nuclear_size_relative_to_nucleus_table(database_path):
    """
    Creates a table with per-cell information and the ratio of nucleolar size to nucleus size for every cell.
    
    Args:
        database_path: path to database with cell and nucleolus information.
    
    Returns:
        pd.DataFrame with nucleolus:nucleus size ratios for every cell.
    """
    conn = sqlite3.connect(database_path)
    query = """
            WITH nucleolar_sizes AS (
            SELECT 
            	Cell_ID, 
            	SUM(Nucleolus_AreaShape_Area) AS Total_Nucleolar_Area 
            FROM Per_Nucleolus
            GROUP BY Cell_ID),

            nuclear_sizes AS (
            SELECT 
            	Cell_ID, 
            	SUM(Nuclei_AreaShape_Area) AS Total_Nuclear_Area 
            FROM Per_Nuclei
            GROUP BY Cell_ID),

            size_ratios AS (
            SELECT
            	nucleolar_sizes.Cell_ID,
            	(CAST(Total_Nucleolar_Area AS NUMERIC) / Total_Nuclear_Area) AS Ratio
            FROM nucleolar_sizes
            JOIN nuclear_sizes ON nuclear_sizes.Cell_ID = nucleolar_sizes.Cell_ID)

            SELECT
            	Replicate,
            	Condition,
            	Row,
            	Column,
            	Per_Cell.Cell_ID,
            	ORF,
            	Name,
            	Strain_ID,
            	Predicted_Label,
            	Ratio
            FROM Per_Cell
            JOIN size_ratios ON Per_Cell.Cell_ID = size_ratios.Cell_ID;"""
            
    size_ratio_df = pl.read_database(query=query, connection=conn)
    conn.close()
    
    return size_ratio_df
    


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
    
# ============================== ABNORMAL NUCLEOLUS SIZE RELATIVE TO NUCLEAR SIZE ==============================
    size_ratio_table = create_nuclear_size_relative_to_nucleus_table(database_path=args.database_path)

    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Nucleolus",
        feature_name="Ratio",
        scaled_feature_dir=f"{args.output_directory}/abnormal_nucleolus_size/large_nucleolus/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_nucleolus_size/large_nucleolus/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_nucleolus_size/large_nucleolus/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_nucleolus_size/large_nucleolus/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_nucleolus_size/large_nucleolus/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_nucleolus_size/large_nucleolus/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Nucleolus",
        feature_table=size_ratio_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95)
    
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Nucleolus",
        feature_name="Ratio",
        scaled_feature_dir=f"{args.output_directory}/abnormal_nucleolus_size/small_nucleolus/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_nucleolus_size/small_nucleolus/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_nucleolus_size/small_nucleolus/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_nucleolus_size/small_nucleolus/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_nucleolus_size/small_nucleolus/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_nucleolus_size/small_nucleolus/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Nucleolus",
        feature_table=size_ratio_table,
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)

# ============================== NUCLEOLUS STD INTENSITY (TOO HIGH/TOO LOW) ==============================
    # high uniformity (per cell -- find nucleoli with very low signal)
    run_all_functions(
        db_path=args.database_path,
        all_cells=all_cells,
        compartment_table_name="Per_Cell",
        feature_name="Cell_Intensity_StdIntensity_GFP",
        scaled_feature_dir=f"{args.output_directory}/abnormal_nucleolus_uniformity/scaled_features",
        outlier_objects_dir=f"{args.output_directory}/abnormal_nucleolus_uniformity/high_uniformity/outlier_cells",
        penetrance_dir=f"{args.output_directory}/abnormal_nucleolus_uniformity/high_uniformity/penetrances",
        cell_count_dir=f"{args.output_directory}/abnormal_nucleolus_uniformity/high_uniformity/cell_counts",
        strain_hits_dir=f"{args.output_directory}/abnormal_nucleolus_uniformity/high_uniformity/strain_hits",
        wt_pens_dir=f"{args.output_directory}/abnormal_nucleolus_uniformity/high_uniformity/per_well_wt_pens",
        plate=args.plate,
        compartment_name="Nucleolus",
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=False,
        percentile_cutoff=0.95)

# ============================== ABNORMAL NUCLEOLAR COUNT ==============================
    conn = sqlite3.connect(args.database_path)
    outlier_objects = (
        pl
        .read_database(
            query="""
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
	                Cell_Children_Nucleolus_Count
                FROM Per_Cell
                WHERE ((Cell_Children_Nucleolus_Count > 1) AND Predicted_Label IN ('G1', 'SG2')) OR
                      ((Cell_Children_Nucleolus_Count > 2) AND Predicted_Label IN ('MAT'));""",
            connection=conn
        )
    )
    conn.close()
    
    if not os.path.exists(f"{args.output_directory}/abnormal_nucleolus_number/outlier_cells"):
        os.makedirs(f"{args.output_directory}/abnormal_nucleolus_number/outlier_cells")
    outlier_objects.write_csv(f"{args.output_directory}/abnormal_nucleolus_number/outlier_cells/{args.plate}_Nucleolus_outlier_cells.csv")
    
    penetrance_table = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=outlier_objects,
        output_dir=f"{args.output_directory}/abnormal_nucleolus_number/penetrances",
        plate=args.plate,
        compartment_name="Nucleolus",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=outlier_objects,
        output_dir=f"{args.output_directory}/abnormal_nucleolus_number/cell_counts",
        plate=args.plate,
        compartment_name="Nucleolus",
        cell_cycle_stages=["G1", "SG2", "MAT"])

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=outlier_objects,
        penetrance_table=penetrance_table,
        wt_pens_dir=f"{args.output_directory}/abnormal_nucleolus_number/per_well_wt_pens",
        output_dir=f"{args.output_directory}/abnormal_nucleolus_number/strain_hits",
        plate=args.plate,
        cc_stages=["G1", "SG2", "MAT"],
        percentile_cutoff=0.95)

# ============================== COMBINE PHENOTYPES ==============================
    combine_output_phenotypes_from_plate(
        phenotype_outliers={
            "LargeNucleolus": f"{args.output_directory}/abnormal_nucleolus_size/large_nucleolus/outlier_cells/{args.plate}_Nucleolus_outlier_cells.csv",
            "SmallNucleolus": f"{args.output_directory}/abnormal_nucleolus_size/small_nucleolus/outlier_cells/{args.plate}_Nucleolus_outlier_cells.csv",
            "HighUniform": f"{args.output_directory}/abnormal_nucleolus_uniformity/high_uniformity/outlier_cells/{args.plate}_Nucleolus_outlier_cells.csv",
            "Fragmented": f"{args.output_directory}/abnormal_nucleolus_number/outlier_cells/{args.plate}_Nucleolus_outlier_cells.csv",
                        
        },
        db_path=args.database_path,
        output_dir=args.output_directory,
        plate=args.plate)


    print("Complete")
