import argparse
import os
import polars as pl
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--screen', default='', help='Name of screen or marker.')
parser.add_argument('-d', '--database_path', default='', help='Path to .db file with CellProfiler features.')
parser.add_argument('-v', '--overlay_directory', default='', help='Path to folder with overlay images.')
parser.add_argument('-r', '--raw_directory', default='', help='Path to folder with raw images.')
parser.add_argument('-t', '--is_tsa', default='False', help='Indicate whether plate is TSA or not. Defaults to False.')
parser.add_argument('-c', '--condition', default='both', help='If TSA, indicate condition. Defaults to both (26C, 37C). Can also specify just 26C or just 37C.')
parser.add_argument('-p', '--plate_number', default='', help='Plate number to use when saving output file.')
parser.add_argument('-o', '--output_directory', default='', help='Where to save coordinates.')

args = parser.parse_args()


def create_coordinate_file_overlay(db_path, overlay_dir, plate, is_tsa, output_folder):
    """
    Create a coordinate file with every cell's Cell_ID, x-coordinate, y-coordinate, and image path. Images show cell,
    nucleus, and compartment overlays. To be used as input for singlecelltool.

    Args:
        db_path (str): path to database containing cells and their information
        overlay_dir (str): path to directory containing images
        plate (str): plate number to use when saving output file
        is_tsa (str): state whether plate is DMA or TSA
        output_folder (str): where to save output
    """
    conn = sqlite3.connect(db_path)
    plate_type = 'DMA'
    if is_tsa == 'True':
        plate_type = 'TSA'
    (
        pl
        .read_database(
            query="""
            SELECT 
                Cell_ID, 
                Cell_AreaShape_Center_X AS Center_X,
                Cell_AreaShape_Center_Y AS Center_Y,
                Replicate,
                Condition,
                Row,
                Column,
                Field,
                Plate
            FROM Per_Cell;
            """,
            connection=conn
        )
        .with_columns(
        (
            pl
            .lit(f"{overlay_dir}/")
            + pl.col('Replicate')
            + pl.lit('_c')
            + pl.col('Condition')
            + pl.lit('r')
            + pl.col('Row')
            + pl.lit('c')
            + pl.col('Column')
            + pl.lit('f')
            + pl.col('Field')
            + pl.lit('p')
            + pl.col('Plate')
            + pl.lit("_overlays.png").cast(pl.Utf8)).alias("Image_Path")
        )
        .select(["Cell_ID", "Image_Path", "Center_X", "Center_Y"])
        .write_csv(f"{output_folder}/{plate_type}_Plate{plate}_overlay_image_paths.csv")
    )

    conn.close()


def create_coordinate_file_fp(db_path, image_dirs, plate, is_tsa, plate_condition, output_folder):
    """
    Create a coordinate file with every cell's Cell_ID, x-coordinate, y-coordinate, and raw image path. To be used as
    input for singlecelltool.

    Args:
        db_path (str): path to database containing cells and their information
        image_dirs (list(str)): paths to directories containing images
        plate (str): number of plate
        is_tsa (str): indicate whether plate is TSA (True) or DMA (False)
        plate_condition (str): for TSA, indicate condition. Can either be 'both' (26C, 37C), just 26C, or just 37C.
        output_folder (str): where to save output file
    """
    conn = sqlite3.connect(db_path)
    coordinate_dfs = []

    if is_tsa == 'True':
        if plate_condition == "26C":
            tsa_folders = {"26": "26oC_PermissiveTemperature"}
        elif plate_condition == "37C":
            tsa_folders = {"37": "37oC_NonPermissiveTemperature"}
        else:
            tsa_folders = {"26": "26oC_PermissiveTemperature", "37": "37oC_NonPermissiveTemperature"}

        for condition, folder_name in tsa_folders.items():

            cell_info = pl.read_database(
                query=f"""
                    SELECT 
                        Cell_ID, 
                        Cell_AreaShape_Center_X AS Center_X,
                        Cell_AreaShape_Center_Y AS Center_Y,
                        Replicate,
                        Condition,
                        Row,
                        Column,
                        Field,
                        Plate
                    FROM Per_Cell
                    WHERE Condition = {condition};
                    """,
                connection=conn
            )

            for rep, image_dir in enumerate(image_dirs):
                rep_coords = (
                    cell_info
                    .filter(pl.col("Replicate") == f"TS{rep + 1}")
                    .with_columns(
                        (
                                pl
                                .lit(f"{image_dir}/{folder_name}/Plate{plate}/")
                                + pl.col('Row')
                                + pl.col('Column')
                                + pl.col('Field')
                                + pl.lit(".flex")
                                .cast(pl.Utf8)).alias("Image_Path")
                    )
                    .select(["Cell_ID", "Image_Path", "Center_X", "Center_Y"])
                )
                coordinate_dfs.append(rep_coords)

        (
            pl
            .concat(items=coordinate_dfs, how="vertical")
            .write_csv(f"{output_folder}/TSA_Plate{plate}_raw_image_paths.csv")
        )

    else:
        cell_info = pl.read_database(
            query="""
                SELECT 
                    Cell_ID, 
                    Cell_AreaShape_Center_X AS Center_X,
                    Cell_AreaShape_Center_Y AS Center_Y,
                    Replicate,
                    Row,
                    Column,
                    Field,
                    Plate
                FROM Per_Cell;
                """,
            connection=conn
        )

        for rep, image_dir in enumerate(image_dirs):
            rep_coords = (
                cell_info
                .filter(pl.col("Replicate") == f"R{rep + 1}")
                .with_columns(
                    (
                            pl
                            .lit(f"{image_dir}/Plate{plate}/")
                            + pl.col('Row')
                            + pl.col('Column')
                            + pl.col('Field')
                            + pl.lit(".flex")
                            .cast(pl.Utf8)).alias("Image_Path")
                )
                .select(["Cell_ID", "Image_Path", "Center_X", "Center_Y"])
            )
            coordinate_dfs.append(rep_coords)

        (
            pl
            .concat(items=coordinate_dfs, how="vertical")
            .write_csv(f"{output_folder}/DMA_Plate{plate}_raw_image_paths.csv")
        )
    conn.close()


if __name__ == '__main__':
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    create_coordinate_file_overlay(
        db_path=args.database_path,
        overlay_dir=args.overlay_directory,
        plate=str(args.plate_number),
        is_tsa=args.is_tsa,
        output_folder=args.output_directory)

    if args.is_tsa == 'True':
        create_coordinate_file_fp(
            db_path=args.database_path,
            image_dirs=[
                f"{args.raw_directory}/TS1_{args.screen}",
                f"{args.raw_directory}/TS2_{args.screen}",
                f"{args.raw_directory}/TS3_{args.screen}"
            ],
            plate=str(args.plate_number),
            is_tsa='True',
            plate_condition=args.condition,
            output_folder=args.output_directory)

    else:
        create_coordinate_file_fp(
            db_path=args.database_path,
            image_dirs=[
                f"{args.raw_directory}/R1_{args.screen}",
                f"{args.raw_directory}/R2_{args.screen}",
                f"{args.raw_directory}/R3_{args.screen}"
            ],
            plate=str(args.plate_number),
            is_tsa='False',
            plate_condition=args.condition,
            output_folder=args.output_directory)

    print("Complete.")