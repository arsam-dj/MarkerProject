import argparse
import polars as pl
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-t', '--compartment_table', default='', help='Name of specific compartment table to remove parentless objects from.')
parser.add_argument('-c', '--compartment_image_column', default='', help='Name of column in Per_Image table pertaining to number of compartment masks in that image.')

args = parser.parse_args()


def delete_parentless_objects(compartment_db, tables):
    """
    Removes objects with no assigned parent cell from specified tables in compartment database.

    Args:
        compartment_db (str): path to database being modified
        tables (dict(str: str)): dictionary of table names and corresponding column name in Per_Image table that describes number of compartment masks in iamge
    """

    conn = sqlite3.connect(compartment_db)
    cursor = conn.cursor()

    for table, per_image_col in tables.items():
        # Delete parentless objects
        cursor.execute(f"DELETE FROM {table} WHERE Cell_ID IS NULL;")

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
    delete_parentless_objects(
        compartment_db=args.database_path,
        tables={"Per_Nuclei": "Image_Count_Nuclei", args.compartment_table: args.compartment_image_column}
    )

    print("Complete.")

