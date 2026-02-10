import argparse
import os
import re
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-o', '--overlay_dir', default='', help='Path to directory with overlay images whose paths need to be fixed.')

args = parser.parse_args()


# Fix formatting issues in Per_Image table of database
def fix_metadata_format(path_to_db):
    """
    The metadata columns are formatted such that the actual data is wrapped in b''. To make future analysis smoother, the
    b'' wrapping needs to be removed.

    Args:
        path_to_db (str): path to database with erroneous formatting in Per_Image table
    """
    # Connect to your SQLite database
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Update the table using SUBSTR to remove the b'' format
    cursor.execute("""
        UPDATE Per_Image
        SET 
            Image_Metadata_Screen = SUBSTR(Image_Metadata_Screen, 3, LENGTH(Image_Metadata_Screen) - 3),
            Image_Metadata_Replicate = SUBSTR(Image_Metadata_Replicate, 3, LENGTH(Image_Metadata_Replicate) - 3),
            Image_Metadata_Condition = SUBSTR(Image_Metadata_Condition, 3, LENGTH(Image_Metadata_Condition) - 3),
            Image_Metadata_Row = SUBSTR(Image_Metadata_Row, 3, LENGTH(Image_Metadata_Row) - 3),
            Image_Metadata_Column = SUBSTR(Image_Metadata_Column, 3, LENGTH(Image_Metadata_Column) - 3),
            Image_Metadata_Plate = SUBSTR(Image_Metadata_Plate, 3, LENGTH(Image_Metadata_Plate) - 3),
            Image_Metadata_Field = SUBSTR(Image_Metadata_Field, 3, LENGTH(Image_Metadata_Field) - 3)
        WHERE Image_Metadata_Row LIKE "b%'"
    """)

    # Commit the changes to the database
    conn.commit()

    # Close the connection
    conn.close()

### Fix overlay paths
def path_fixer(root_dir):
    """
    Given a root_dir, loop through all images in root_dir and rename them using correct row/col/field
    convention.

    Args:
        root_dir (str): path to directory with all images to be renamed
    """
    image_files = os.listdir(root_dir)
    for image_file in image_files:
        os.rename(src=f"{root_dir}/{image_file}", dst=f"{root_dir}/{re.sub(r"(?:^|(\w))b'([A-Za-z0-9]+)'", lambda m: (m.group(1) or "") + m.group(2), image_file)}")

if __name__ == '__main__':
    if args.database_path:
        fix_metadata_format(args.database_path)

    if args.overlay_dir:
        path_fixer(args.overlay_dir)

    print("Complete.")