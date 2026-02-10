import argparse
import os
import polars as pl
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-q', '--qc_directory', default='', help='Path to directory to write quality check files to.')
parser.add_argument('-c', '--qc_cell_features', default='', help='Path to file with Cell_IDs and scaled QC features.')
parser.add_argument('-n', '--qc_nuclear_features', default='', help='Path to file with Cell_IDs and QC features for nuclei.')
parser.add_argument('-a', '--additional_features', default='', help='Path to file with Cell_IDs and QC features for any additional features.')
parser.add_argument('-p', '--plate', default='', help='Number for identifying plate being processed.')

args = parser.parse_args()


def save_filtered_objects(db_path, filtered_cells, output_path, plate):
    """
    First identifies and saves cells to be filtered out along with their gene/strain information. Then, saves the
    percentage of cells that got filtered out for each strain.

    Args:
        db_path (str): path to database being modified
        filtered_cells (list): list of Cell_IDs to filter out (not including those filtered out due to num. nuclei abnormalities)
        output_path (str): where to save list of filtered cells
        plate (str): plate identifier for saving output files
    """
    with sqlite3.connect(db_path) as conn:

        cells = ",".join(f"'{item}'" for item in filtered_cells)

        # Getting filtered cells
        query1 = f"""SELECT DISTINCT
                        Replicate,
                        Cell_ID,
                        ORF,
                        Name,
                        Strain_ID
                    FROM Per_Cell 
                    WHERE 
                        Cell_ID IN ({cells}) 
                        OR Cell_Children_Nuclei_Count > 2 
                        OR Cell_Children_Nuclei_Count = 0
                        OR ORF = 'BLANK'
                 """
        (
            pl
            .read_database(query=query1, connection=conn)
            .write_csv(f"{output_path}/{plate}_filtered_cells.csv")
        )

        # Getting percentage of cells filtered out for each strain
        query2 = f"""
            WITH strain_total_cell_counts AS (
	            SELECT
	                Replicate,
	                ORF,
	                Name,
	            	Strain_ID,
	            	COUNT(*) AS Total_Num_Cells
	            FROM Per_Cell
	            GROUP BY Replicate, ORF, Name, Strain_ID
	            ),
	    
            strain_filtered_cell_counts AS (
	            SELECT
	            	Replicate,
	            	Strain_ID,
	            	COUNT(*) AS Filtered_Num_Cells
	            FROM Per_Cell
	            WHERE 
	                Cell_ID IN ({cells}) 
	                OR Cell_Children_Nuclei_Count > 4  
	                OR Cell_Children_Nuclei_Count = 0
	                OR ORF = 'BLANK'
	            GROUP BY Replicate, Strain_ID
	            )
    
            SELECT DISTINCT
            	strain_total_cell_counts.Replicate,
            	ORF,
            	Name,
            	strain_total_cell_counts.Strain_ID,
            	Total_Num_Cells,
            	Filtered_Num_Cells,
            	Total_Num_Cells - Filtered_Num_Cells AS New_Total_Num_Cells,
            	CASE 
            	    WHEN Total_Num_Cells > 0 
            	    THEN (CAST(Filtered_Num_Cells AS REAL) / Total_Num_Cells) * 100 
            	    ELSE 0 
            	END AS Percent_Filtered
            FROM strain_total_cell_counts
            JOIN strain_filtered_cell_counts 
                ON strain_total_cell_counts.Replicate = strain_filtered_cell_counts.Replicate 
                AND strain_total_cell_counts.Strain_ID = strain_filtered_cell_counts.Strain_ID
        """
        perc_filtered_df = (
            pl
            .read_database(query=query2, connection=conn)
        )
        if "TS1" in perc_filtered_df["Replicate"] or "TS2" in perc_filtered_df["Replicate"] or "TS3" in perc_filtered_df["Replicate"]:
            perc_filtered_df = (
                perc_filtered_df
                .with_columns(
                    (pl.col("Replicate").replace_strict(["TS1", "TS2", "TS3"], ["R1", "R2", "R3"])).alias("Replicate")
                )
            )

        (
            perc_filtered_df
            .pivot(on="Replicate",
                   index=["ORF", "Name", "Strain_ID"],
                   values=["Total_Num_Cells", "Filtered_Num_Cells", "New_Total_Num_Cells", "Percent_Filtered"]
                   )
            .select([
                "ORF",
                "Name",
                "Strain_ID",
                "Total_Num_Cells_R1",
                "Filtered_Num_Cells_R1",
                "New_Total_Num_Cells_R1",
                "Percent_Filtered_R1",
                "Total_Num_Cells_R2",
                "Filtered_Num_Cells_R2",
                "New_Total_Num_Cells_R2",
                "Percent_Filtered_R2",
                "Total_Num_Cells_R3",
                "Filtered_Num_Cells_R3",
                "New_Total_Num_Cells_R3",
                "Percent_Filtered_R3",
            ])
            .fill_null(0)
            .with_columns(
                (
                        pl.col("Total_Num_Cells_R1") +
                        pl.col("Total_Num_Cells_R2") +
                        pl.col("Total_Num_Cells_R3")
                ).alias("Total_Num_Cells"),
                (
                        pl.col("Filtered_Num_Cells_R1") +
                        pl.col("Filtered_Num_Cells_R2") +
                        pl.col("Filtered_Num_Cells_R3")
                ).alias("Total_Filtered_Num_Cells")
            )
            .with_columns(
                (
                        pl.col("Total_Num_Cells") -
                        pl.col("Total_Filtered_Num_Cells")
                ).alias("New_Total_Num_Cells"),
                (
                        (pl.col("Total_Filtered_Num_Cells") / pl.col("Total_Num_Cells")) * 100
                ).alias("Percent_Filtered")
            )
            .sort("Percent_Filtered", descending=True)
            .write_csv(f"{output_path}/{plate}_strain_stats_after_filtering.csv")
        )

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    # Combine QC dataframes
    all_cells = pl.read_csv(args.qc_cell_features)
    nuclei = pl.read_csv(args.qc_nuclear_features)

    all_data = (
        all_cells
        .join(nuclei, on=["Cell_ID", "Image_Path", "Center_X", "Center_Y"], how="left")
    )

    if args.additional_features:
        additional_features = pl.read_csv(args.additional_features, columns=["Cell_ID", "Image_Path", "Center_X", "Center_Y", "Trilobed", "Asymmetric"])
        all_data = (
            all_data
            .join(additional_features, on=["Cell_ID", "Image_Path", "Center_X", "Center_Y"], how="left")
        )

    # Get filtered cells/objects
    filtered_cells = (
        all_data
        .filter(
            (pl.col('Cell_AreaShape_Area') <= -1.25) | (pl.col('Cell_AreaShape_Area') >= 3) |
            (pl.col('Cell_AreaShape_Perimeter') <= -1.5) | (pl.col('Cell_AreaShape_Perimeter') >= 3.25) |
            (pl.col('Cell_AreaShape_MajorAxisLength') <= -1.5) | (pl.col('Cell_AreaShape_MajorAxisLength') >= 5) |
            (pl.col('Cell_AreaShape_MinorAxisLength') <= -2) | (pl.col('Cell_AreaShape_MinorAxisLength') >= 3) |
            (pl.col('Cell_AreaShape_FormFactor') >= 0.8) |
            (pl.col('Cell_AreaShape_Extent') <= -4) | (pl.col('Cell_AreaShape_Extent') >= 3) |
            (pl.col('Cell_AreaShape_Compactness') <= -1) | (pl.col('Cell_AreaShape_Compactness') >= 3) |
            (pl.col('Cell_AreaShape_MaxFeretDiameter') <= -1.5) | (pl.col('Cell_AreaShape_MaxFeretDiameter') >= 5) |
            (pl.col('Cell_AreaShape_MinFeretDiameter') <= -2) | (pl.col('Cell_AreaShape_MinFeretDiameter') >= 3) |
            (pl.col('Area_Over_Perimeter') <= -1.8) | (pl.col('Area_Over_Perimeter') >= 3.75) |
            (pl.col('MajorAxisLength_Over_MinorAxisLength') >= 5) |
            (pl.col('MaxFeretDiameter_Over_MinFeretDiameter') >= 5) |
            (pl.col('Trilobed') >= 0.008) |
            (pl.col('Asymmetric') >= 0.12) |
            (pl.col('Nuclei_Distance_Minimum_Cell') <= 4) | (pl.col('Nuclei_Distance_Minimum_Cell') >= 18) |
            (pl.col('Nuclear_Area_Over_Cell_Area') >= 0.4)
        )
        .select("Cell_ID")
        .to_series()
        .to_list()
    )

    # Save:
    #   1) filtered objects for each plate
    #   2) percentage of cells that got filtered out for each strain
    save_filtered_objects(
        db_path=args.database_path,
        filtered_cells=filtered_cells,
        output_path=args.qc_directory,
        plate=args.plate
    )

    print("Complete.")
