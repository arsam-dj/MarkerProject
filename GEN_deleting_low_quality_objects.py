import argparse
import polars as pl
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-t', '--compartment_table', default='', help='Name of specific compartment table to remove parentless objects from.')
parser.add_argument('-c', '--compartment_image_column', default='', help='Name of column in Per_Image table pertaining to number of compartment masks in that image.')
parser.add_argument('-f', '--filtered_cells', default='', help='csv file with Cell_IDs to filter out.')

args = parser.parse_args()


def delete_low_quality_objects(db_path, tables, cell_ids):
    """
    Removes low quality objects by their Cell_ID from database.

    Args:
        db_path (str): path to database being modified
        tables (dict(str: str)): dictionary of table names and corresponding column name in Per_Image table that describes number of compartment masks in image
        cell_ids (list(str)): list of Cell_IDs to remove from tables
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for table, per_image_col in tables.items():
        # Delete low quality objects
        cells = ",".join(f"'{item}'" for item in cell_ids)
        cursor.execute(f"DELETE FROM {table} WHERE Cell_ID IN ({cells});")

        # Update all images, setting missing counts to 0
        cursor.execute(f"""
            WITH counts AS (
                SELECT ImageNumber, COUNT(Cell_ID) AS count
                FROM {table}
                GROUP BY ImageNumber
            )
            UPDATE Per_Image
            SET {per_image_col} = COALESCE((
                SELECT count
                FROM counts
                WHERE counts.ImageNumber = Per_Image.ImageNumber
            ), 0);
        """)

        conn.commit()

    conn.close()


if __name__ == '__main__':

    # Get Cell_IDs to filter out
    cell_ids = (
        pl
        .read_csv(args.filtered_cells)
        .select(["Cell_ID"])
        .to_series()
        .to_list()
    )
    # Delete low quality objects from all tables (compartments of low quality objects are deleted too)
    delete_low_quality_objects(
        db_path=args.database_path,
        tables={"Per_Cell": "Image_Count_Cell", "Per_Nuclei": "Image_Count_Nuclei", args.compartment_table: args.compartment_image_column},
        cell_ids=cell_ids
    )

    # Vacuum database
    conn = sqlite3.connect(args.database_path)
    cursor = conn.cursor()
    cursor.execute("VACUUM;")
    conn.close()

    print("Complete.")

