import argparse
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')

args = parser.parse_args()


def add_is_nuclear_feature(path_to_db):
    """
    Adds a new feature to the Per_LDs table of TGL3 screen called Is_Nuclear. The values are T or F where T indicates
    a perfect overlap with the nucleus while F indicates otherwise. In addition, the Per_NuclearLDs table is dropped
    from the database.

    Args:
        path_to_db (str): path to database to be modified
    """

    # Connect to database
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()

    # Step 1: create indexes
    cursor.execute("CREATE INDEX idx_perlds_img_ld ON Per_LDs(ImageNumber, LDs_Number_Object_Number);")
    cursor.execute("CREATE INDEX idx_nuclear_img_parent ON Per_NuclearLDs(ImageNumber, NuclearLDs_Parent_LDs);")

    # Step 2: add a new feature column to Per_LDs
    cursor.execute("ALTER TABLE Per_LDs ADD COLUMN Is_Nuclear TEXT DEFAULT 'F';")

    # Step 3: update the feature column based on whether that LD is present in the Per_NuclearLDs table
    cursor.execute("""
        UPDATE Per_LDs
        SET Is_Nuclear = 'T'
        WHERE EXISTS (
        	SELECT 1
        	FROM Per_NuclearLDs
        	WHERE Per_NuclearLDs.ImageNumber = Per_LDs.ImageNumber
        		AND Per_NuclearLDs.NuclearLDs_Parent_LDs = Per_LDs.LDs_Number_Object_Number
        );
        """)

    # Step 4: drop Per_NuclearLDs and indices
    cursor.execute("DROP TABLE Per_NuclearLDs;")
    cursor.execute("DROP INDEX idx_perlds_img_ld;")

    # Step 5: commit changes and close
    conn.commit()
    conn.close()


if __name__ == '__main__':

    # In the TGL3 screen, some LDs are further labelled as 'nuclear LDs' if they happen to perfectly overlap the cell
    # nucleus. The Per_NuclearLDs table contains duplicate rows from the Per_LDs table, so the point of this script is
    # to:
        # 1. Add a new feature to Per_LDs ("Is_Nuclear") where value of F indicates cell LD and T indicates nuclear LD
        # 2. Drop the Per_NuclearLDs table

    add_is_nuclear_feature(path_to_db=args.database_path)

    print("Complete.")
