import matplotlib.pyplot as plt
import os
import pandas as pd
import polars as pl
import seaborn as sns
import sqlite3


# Function for creating matrix of feature distribution histograms
def feature_distributions_matrix(qc_features, qc_directory, output_figure_name):
    """
    Generate a matrix of histogram showing feature distributions (log10 freq vs. feature Z scores).

    Args:
        qc_features (DataFrame): dataframe of scaled feature values
        qc_directory (str): path to directory to write qc files to
        output_figure_name (str): what to save output figure as
    """

    # Since I'm aiming to filter out small numbers cells with extreme values, I'm going to generate plots with y-axis on
    # log10 scale.

    # Convert to long format for plotting
    qc_features_long = qc_features.melt(variable_name='Feature', value_name='Z_Score')

    # First, plot everything as a grid to get a general idea of all distributions
    g = sns.displot(data=qc_features_long,
                    bins=100,
                    x='Z_Score',
                    col='Feature',
                    col_wrap=4,
                    height=5,
                    aspect=2,
                    common_bins=False,
                    facet_kws={'sharex': False, 'sharey': False}
                    )

    # Set the y-axis scale to log10 for each facet
    for ax in g.axes.flat:
        # Enable minor ticks
        ax.minorticks_on()

        # Change y-axis to log
        ax.set_yscale('log')  # Set y-axis to log scale

        # Draw grid for better readability
        ax.grid(True)  # Add grid

        # Make the title bold and larger
        ax.set_title(ax.get_title(), fontsize=14, fontweight='bold')

    # Adjust spacing between rows of plots (increase gap)
    plt.subplots_adjust(hspace=0.5)

    plt.tight_layout()
    plt.savefig(f"{qc_directory}/{output_figure_name}.png", bbox_inches="tight", dpi=700)
    plt.close()


def delete_problematic_compartment_masks(db_path, filtered_comps, comp_name, output_dir, plate, delete_all_comp_masks="False", replace_comp_num_with=-1, save_csv="True"):
    """
    Deletes problematic compartment masks specifid in filtered_comps. If indicated, deletes all other masks
    for that cell as well. Updates number of children for that cell with the new value or <replace_comp_num_with>
    if all compartments deleted.

    Also calculates and saves the number of compartments deleted for each strain.

    Args:
        db_path (str): path to database being modified
        filtered_comps (pl.DataFrame): dataframe with compartments that have been filtered out (two columns: Cell_ID of parent cell and <comp>_Number_Object_Number)
        comp_name (str): name of compartment used in table names (ex. SPBs)
        output_dir (str): where to save output files to
        plate (str): plate identifier for saving output files
        delete_all_comp_masks (str): indicate if all compartment masks in given parent cell should be deleted ('True') or only the problematic mask ('False')
        replace_comp_num_with (int): what to replace Cell_Children_<comp>_Count as if deleting all masks; defaults to -1
        save_csv (str): if 'True', creates files showing percentage of compartments filtered and every cell modified for every strain
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Turn filtered_comps to a temp_table in the database
    filtered_comps.to_pandas().to_sql("temp_table", conn, index=False, if_exists="replace")
    conn.commit()

    # Save filtered compartment masks with strain information
    if save_csv == "True":
        (
            pl
            .read_database(query="SELECT ORF, Name, Strain_ID, Cell_ID FROM Per_Cell", connection=conn)
            .join(filtered_comps, on=["Cell_ID"])
            .write_csv(f"{output_dir}/{plate}_filtered_{comp_name}.csv")
        )



        # Determine mean and median number of compartments removed from cells in each strain
        query1 = f"""
                 WITH num_cells_affected AS (
                    SELECT
                        ORF,
                        Name,
                        Strain_ID,
                        COUNT(DISTINCT temp_table.Cell_ID) AS Num_Cells_Affected
                    FROM temp_table
                    JOIN (SELECT Cell_ID, ORF, Name, Strain_ID FROM Per_Cell) pc
                    ON temp_table.Cell_ID = pc.Cell_ID
                    GROUP BY Strain_ID
                 ),

                 total_cells AS (
                    SELECT
                        Strain_ID,
                        COUNT(DISTINCT Cell_ID) AS Total_Cells
                    FROM Per_Cell
                    GROUP BY Strain_ID
                )

                 SELECT
                    ORF,
                    Name,
                    total_cells.Strain_ID,
                    Total_Cells,
                    COALESCE(Num_Cells_Affected, 0) AS Num_Cells_Affected
                FROM total_cells
                LEFT JOIN num_cells_affected
                    ON total_cells.Strain_ID = num_cells_affected.Strain_ID;
                  """

        query2 = f"""
                 WITH original_comp_counts AS (
                      SELECT
                        Strain_ID,
                        Cell_ID,
                        Cell_Children_{comp_name}_Count
                      FROM Per_Cell
                 ),

                 num_comps_removed AS (
                      SELECT
                        Cell_ID,
                        COUNT(*) AS Comp_Masks_Removed
                      FROM temp_table
                      GROUP BY Cell_ID
                 )

                 SELECT
                     Strain_ID,
                     Cell_Children_{comp_name}_Count,
                     COALESCE(Comp_Masks_Removed, 0) AS Comp_Masks_Removed
                 FROM original_comp_counts
                 LEFT JOIN num_comps_removed
                     ON original_comp_counts.Cell_ID = num_comps_removed.Cell_ID;
                  """

        df1 = (
            pl
            .read_database(query=query1, connection=conn)
            .with_columns(
                    (pl.col("Num_Cells_Affected") / pl.col("Total_Cells") * 100).alias("Percent_Cells_Affected")
            )
        )

        df2 = (
            pl
            .read_database(query=query2, connection=conn)
            .with_columns(
                ((pl.col("Comp_Masks_Removed") / pl.col(f"Cell_Children_{comp_name}_Count")) * 100).alias("Percent_Comps_Removed")
            )
            .fill_nan(0)
            .group_by(["Strain_ID"])
            .agg(
                pl.col("Percent_Comps_Removed").mean().alias("Mean_Percent_Comps_Removed"),
                pl.col("Percent_Comps_Removed").median().alias("Median_Percent_Comps_Removed")
            )
        )

        (
            df1
            .join(df2, on="Strain_ID")
            .sort("Percent_Cells_Affected", descending=True)
            .write_csv(f"{output_dir}/{plate}_strain_stats.csv")
        )

    # Delete specific problematic compartment masks
    if delete_all_comp_masks == "False":
        # Create indexes for efficiency
        cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_temp_table
        ON temp_table (Cell_ID, {comp_name}_Number_Object_Number);
        """)

        cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_per_comps
        ON Per_{comp_name} (Cell_ID, {comp_name}_Number_Object_Number);
        """)

        delete_query = f"""
                        DELETE FROM Per_{comp_name}
                        WHERE EXISTS (
                            SELECT 1
                            FROM temp_table
                            WHERE temp_table.Cell_ID = Per_{comp_name}.Cell_ID
                              AND temp_table.{comp_name}_Number_Object_Number = Per_{comp_name}.{comp_name}_Number_Object_Number
                        );
                        """
        # Update children count for affected parent cells
        update_query = f"""
                       WITH counts AS (
                            SELECT Cell_ID, COUNT({comp_name}_Number_Object_Number) AS count
                            FROM Per_{comp_name}
                            GROUP BY Cell_ID
                       )
                       UPDATE Per_Cell
                       SET Cell_Children_{comp_name}_Count = COALESCE((
                           SELECT count
                           FROM counts
                           WHERE counts.Cell_ID = Per_Cell.Cell_ID
                       ), 0);
                        """
        cursor.execute(delete_query)
        cursor.execute(update_query)

    # Delete all masks belonging to a given cell, regardless of whether it's problematic
    else:
        delete_query = f"""
                       DELETE FROM Per_{comp_name}
                       WHERE Cell_ID IN (SELECT Cell_ID FROM temp_table);
                        """
        # Update children count for affected parent cells
        update_query = f"""
                       UPDATE Per_Cell
                       SET Cell_Children_{comp_name}_Count = {replace_comp_num_with}
                       WHERE Per_Cell.Cell_ID IN (SELECT Cell_ID FROM temp_table);
                        """
        cursor.execute(delete_query)
        cursor.execute(update_query)

    conn.commit()
    cursor.execute("DROP INDEX idx_temp_table;")
    cursor.execute("DROP INDEX idx_per_comps;")
    cursor.execute("DROP TABLE IF EXISTS temp_table;")
    conn.close()
