import argparse
import pandas as pd
import polars as pl
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-i', '--comp_identifier', default='', help='Number to identify a specific screen when generating Cell IDs.')
parser.add_argument('-t', '--comp_table', default='', help='Name of table with compartment features.')
parser.add_argument('-c', '--comp_column', default='', help='Name of column in compartment table that has parent identifier.')
parser.add_argument('-x', '--coordinates', default='', help='Path to file with coordinates for each ORF/strain.')
parser.add_argument('-r', '--temp_sensitive', default='False', help='Indicate if this is a TSA plate (True) or DMA plate (False).')
parser.add_argument('-z', '--leading_zeroes', default=3, help='When merging with coordinates file, need to specify how many characters there are for row/column values (differs between Opera and Phenix). For Opera, put down 3. For Phenix, put down 2.')


args = parser.parse_args()


### Add Replicate/Plate/Row/Column/Field/ORF/Marker information to Per_Cell
def add_cell_info(path_to_db, comp_identifier, coordinates_path, is_tsa='False', leading_zeroes=3):
    """
    Add cell information to Per_Cell table of database.

    Args:
        path_to_db (str): path to database to be modified
        comp_identifier (int): a number to identifier a specific screen when generating Cell IDs
        coordinates_path (str): path to csv file with array mapping for ORFs/strains on plate
        is_tsa (str): indicate whether plate is TSA ('True') or DMA ('False'). If True, adds -26C and -37C suffixes to Strain IDs
        leading_zeroes (int): how many characters there are in row/column values. Phenix screens have 2, Opera screens have 3.
                              The number of leading zeroes is actually leading_zeroes - 1 but for zfill, put leading_zeroes.
    """

    # Load the well plate coordinates, remove rows with NA values, and turn coordinates into strings with leading 0s
    coordinates = (
        pl
        .read_csv(coordinates_path)
        .with_columns(
            pl.col("Plate").cast(pl.String).str.zfill(2).alias("Plate"),
            pl.col("Row").cast(pl.String).str.zfill(leading_zeroes).alias("Row"),
            pl.col("Column").cast(pl.String).str.zfill(leading_zeroes).alias("Column")
        )
    )

    # Connect to database
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Read necessary columns from Per_Cell and Per_Image data tables
    cells_df = pl.read_database(query="""SELECT 
                                            ImageNumber, 
                                            Cell_Number_Object_Number 
                                         FROM Per_Cell ORDER BY ImageNumber ASC, Cell_Number_Object_Number ASC;
                                        """,
                                connection=conn)
    images_df = pl.read_database(query="""SELECT 
                                            ImageNumber, 
                                            Image_Metadata_Replicate AS Replicate, 
                                            Image_Metadata_Condition AS Condition,
                                            Image_Metadata_Plate AS Plate,
                                            Image_Metadata_Row AS Row, 
                                            Image_Metadata_Column AS Column, 
                                            Image_Metadata_Field AS Field
                                          FROM Per_Image;
                                        """,
                                 connection=conn)

    # Do a left-join on ImageNumber, re-order resulting dataframe and create a Cell_ID column
    column_order = ['Replicate', 'Condition', 'Plate', 'Row', 'Column', 'Field', 'ImageNumber', 'Cell_Number_Object_Number']
    columns_in_cellid = ['Replicate', 'Condition', 'Plate', 'Row', 'Column', 'Field', 'Cell_Number_Object_Number']

    combined_df = (
        cells_df
        .join(images_df, on="ImageNumber", how="left")  # add image metadata
        .select(column_order)  # re-order columns
        .with_columns((pl.lit(str(comp_identifier)) + (pl.concat_str(columns_in_cellid, separator=""))).alias("Cell_ID"))  # create a Cell_ID column
        .join(coordinates, on=["Plate", "Row", "Column"], how="left"))  # add ORF/marker information

    # Add Replicate, Plate, Row, Column, Field, Cell_ID, ORF, Name, Allele, and Strain_ID,
    # to database. ImageNumber and Cell_Number_Object_Number are common between
    # combined_df and Per_Cell, so it's possible to join them together.

    # Step 1: turn combined_df into a temp_table in database
    combined_df.to_pandas().to_sql("temp_table", conn, index=False, if_exists="replace")

    # Step 2: create new columns in Per_Cell that exist in combined_df
    missing_columns = [x for x in combined_df.columns if x not in {"ImageNumber", "Cell_Number_Object_Number"}]
    for col in missing_columns:
        cursor.execute(f'ALTER TABLE Per_Cell ADD COLUMN "{col}" TEXT')

    # Step 3: Create indexes on join keys (if they don't already exist)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_Per_Cell ON Per_Cell (ImageNumber, Cell_Number_Object_Number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_temp_table ON temp_table (ImageNumber, Cell_Number_Object_Number)")
    conn.commit()

    # Step 4: Use a CTE (Common Table Expression) for faster updates
    update_clause = ", ".join(
        f"{col} = (SELECT temp_table.{col} FROM temp_table WHERE temp_table.ImageNumber = Per_Cell.ImageNumber AND temp_table.Cell_Number_Object_Number = Per_Cell.Cell_Number_Object_Number)"
        for col in missing_columns
    )
    update_query = f"""
        UPDATE Per_Cell
        SET {update_clause}
        WHERE EXISTS (
            SELECT 1 FROM temp_table 
            WHERE temp_table.ImageNumber = Per_Cell.ImageNumber 
            AND temp_table.Cell_Number_Object_Number = Per_Cell.Cell_Number_Object_Number
        )
    """
    cursor.execute(update_query)

    # Step 5: if it's a TSA plate, add the -26C and -37C suffixes to Strain IDs
    if is_tsa == 'True':
        cursor.execute("""
                        UPDATE Per_Cell 
                        SET Strain_ID = Strain_ID || '-' || Condition || 'C';
                        """)

    # Step 6: drop temp table
    cursor.execute("DROP TABLE temp_table;")
    cursor.execute("DROP INDEX idx_Per_Cell;")

    # Commit and close connection
    conn.commit()
    conn.close()


### Add Cell_ID to Per_<Compartment> tables
def add_cell_ids_to_comp_table(path_to_db, table_src, table_dst, col_src, col_dst):
    """
    Add Cell_ID and other cell information to the Per_<Compartment> table in a given database using information from
    the Per_Cell table.

    Args:
        path_to_db (str): path to database to be modified
        table_src (str): table name to get cell information from
        table_dst (str): table name to add information to
        col_src (str): name of column in source table to use in joining, should match col_dst
        col_dst (str): name of column in destination table to use in joining, should match col_src
    """

    # Connect to the SQLite database
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Step 1: Add missing columns to <table_dst>
    new_columns = {
        "Replicate": "TEXT",
        "Condition": "TEXT",
        "Plate": "TEXT",
        "Row": "TEXT",
        "Column": "TEXT",
        "Field": "TEXT",
        "Cell_ID": "TEXT",
        "ORF": "TEXT",
        "Name": "TEXT",
        "Allele": "TEXT",
        "Strain_ID": "TEXT",
    }

    for column, col_type in new_columns.items():
        try:
            cursor.execute(f'ALTER TABLE {table_dst} ADD COLUMN "{column}" {col_type}')
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

    # Step 2: Create indexes to speed up joins
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_src ON {table_src}(ImageNumber, {col_src})")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_dst ON {table_dst}(ImageNumber, {col_dst})")

    # Step 3: Create a temporary table with the join result
    cursor.execute("DROP TABLE IF EXISTS temp_joined_data")
    cursor.execute(f"""
        CREATE TEMP TABLE temp_joined_data AS
        SELECT
            n.RowID AS comp_rowid,
            c.Replicate,
            c.Condition,
            c.Plate,
            c.Row,
            c.Column,
            c.Field,
            c.Cell_ID,
            c.ORF,
            c.Name,
            c.Allele,
            c.Strain_ID
        FROM {table_dst} n
        JOIN {table_src} c
            ON n.ImageNumber = c.ImageNumber
            AND n.{col_dst} = c.{col_src}
    """)

    # Step 4: Create an index on the temp table for faster lookup
    cursor.execute("CREATE INDEX idx_temp_rowid ON temp_joined_data(comp_rowid)")

    # Step 5: Update <table_dst> using data from temp table
    cursor.execute(f"""
        UPDATE {table_dst}
        SET
            Replicate = (
                SELECT Replicate FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Condition = (
                SELECT Condition FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Plate = (
                SELECT Plate FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Row = (
                SELECT Row FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Column = (
                SELECT Column FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Field = (
                SELECT Field FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Cell_ID = (
                SELECT Cell_ID FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            ORF = (
                SELECT ORF FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Name = (
                SELECT Name FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Allele = (
                SELECT Allele FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            ),
            Strain_ID = (
                SELECT Strain_ID FROM temp_joined_data
                WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
            )
        WHERE EXISTS (
            SELECT 1 FROM temp_joined_data
            WHERE {table_dst}.RowID = temp_joined_data.comp_rowid
        )
    """)

    # Step 6: Clean-up
    cursor.execute("DROP TABLE temp_joined_data;")
    cursor.execute("DROP INDEX idx_src;")
    cursor.execute("DROP INDEX idx_dst;")

    # Commit and close
    conn.commit()
    conn.close()
    
if __name__ == '__main__':
    add_cell_info(
        path_to_db=args.database_path,
        comp_identifier=args.comp_identifier,
        coordinates_path=args.coordinates,
        is_tsa=args.temp_sensitive,
        leading_zeroes=int(args.leading_zeroes)
    )

    add_cell_ids_to_comp_table(path_to_db=args.database_path, table_src="Per_Cell", table_dst="Per_Nuclei", col_src="Cell_Number_Object_Number", col_dst="Nuclei_Parent_Cell")
    add_cell_ids_to_comp_table(path_to_db=args.database_path, table_src="Per_Cell", table_dst=args.comp_table, col_src="Cell_Number_Object_Number", col_dst=args.comp_column)

    print("Complete.")
