import argparse
import os
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sqlite3

from GEN_quality_check_functions import feature_distributions_matrix

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--qc_directory', default='', help='Path to directory to write quality check files to.')
parser.add_argument('-d', '--database_directory', default='', help='Path to directory with all databases.')
parser.add_argument('-x', '--cell_coordinates', default='', help='Path to file with all cell overlay coordinates.')

args = parser.parse_args()


def create_cell_qc_raw_scaled_df(database_directory, coordinates_path, qc_directory):
    """
    Create a dataframe with cells from all three reps, as well as QC features of interest. Create
    a second dataframe with scaled feature values (z-scores). Include the segmentation mask paths so they can be viewed with Single Cell Tool.

    Args:
        database_directory (str): path to directory with all plate databases
        coordinates_path (str): path to file with cell overlay coordinates
        qc_directory (str): path to directory to write qc files to
    """
    # Define cell info and QC features
    cell_info_features = [
        'Cell_ID', 'ORF', 'Name', 'Strain_ID', 'Cell_AreaShape_Center_X', 'Cell_AreaShape_Center_Y'
    ]

    qc_features = [
        'Cell_AreaShape_Area', 'Cell_AreaShape_Perimeter', 'Cell_AreaShape_MajorAxisLength',
        'Cell_AreaShape_MinorAxisLength', 'Cell_AreaShape_Eccentricity', 'Cell_AreaShape_FormFactor',
        'Cell_AreaShape_Extent', 'Cell_AreaShape_Compactness', 'Cell_AreaShape_MaxFeretDiameter',
        'Cell_AreaShape_MinFeretDiameter'
    ]

    # Read all plate databases and get cell info + qc feature columns, combine
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]
    qc_dfs = []

    for db_path in databases:
        conn = sqlite3.connect(db_path)

        # Read simpler features
        plate_qc_df = pl.read_database(
            query=f"SELECT {', '.join(cell_info_features + qc_features)} FROM Per_Cell", connection=conn)

        # Read Zernike features for getting more complex shapes
        query = """SELECT 
                        Cell_ID, 
                        Cell_AreaShape_Zernike_0_0 AS Z00, 
                        Cell_AreaShape_Zernike_1_1 AS Z11, 
                        Cell_AreaShape_Zernike_2_0 AS Z20,
                        Cell_AreaShape_Zernike_2_2 AS Z22, 
                        Cell_AreaShape_Zernike_3_1 AS Z31,
                        Cell_AreaShape_Zernike_3_3 AS Z33, 
                        Cell_AreaShape_Zernike_4_0 AS Z40,
                        Cell_AreaShape_Zernike_4_2 AS Z42,
                        Cell_AreaShape_Zernike_4_4 AS Z44,
                        Cell_AreaShape_Zernike_5_1 AS Z51,
                        Cell_AreaShape_Zernike_5_3 AS Z53, 
                        Cell_AreaShape_Zernike_5_5 AS Z55, 
                        Cell_AreaShape_Zernike_6_0 AS Z60,
                        Cell_AreaShape_Zernike_6_2 AS Z62,
                        Cell_AreaShape_Zernike_6_4 AS Z64,
                        Cell_AreaShape_Zernike_6_6 AS Z66,
                        Cell_AreaShape_Zernike_7_1 AS Z71,
                        Cell_AreaShape_Zernike_7_3 AS Z73, 
                        Cell_AreaShape_Zernike_7_5 AS Z75,
                        Cell_AreaShape_Zernike_7_7 AS Z77,
                        Cell_AreaShape_Zernike_8_0 AS Z80,
                        Cell_AreaShape_Zernike_8_2 AS Z82,
                        Cell_AreaShape_Zernike_8_4 AS Z84,
                        Cell_AreaShape_Zernike_8_6 AS Z86,
                        Cell_AreaShape_Zernike_8_8 AS Z88,
                        Cell_AreaShape_Zernike_9_1 AS Z91,
                        Cell_AreaShape_Zernike_9_3 AS Z93,
                        Cell_AreaShape_Zernike_9_5 AS Z95,
                        Cell_AreaShape_Zernike_9_7 AS Z97,
                        Cell_AreaShape_Zernike_9_9 AS Z99
                    FROM Per_Cell;
                """
        additional_features = (
            pl
            .read_database(query=query, connection=conn)
            .with_columns(
                (
                        (pl.col("Z33") ** 2) +
                        (pl.col("Z53") ** 2) +
                        (pl.col("Z73") ** 2) +
                        (pl.col("Z93") ** 2)
                ).alias("E_Trilobed"),
                (
                        (pl.col("Z11") ** 2) +
                        (pl.col("Z22") ** 2) +
                        (pl.col("Z31") ** 2) +
                        (pl.col("Z33") ** 2) +
                        (pl.col("Z42") ** 2) +
                        (pl.col("Z44") ** 2) +
                        (pl.col("Z51") ** 2) +
                        (pl.col("Z53") ** 2) +
                        (pl.col("Z55") ** 2) +
                        (pl.col("Z62") ** 2) +
                        (pl.col("Z64") ** 2) +
                        (pl.col("Z66") ** 2) +
                        (pl.col("Z71") ** 2) +
                        (pl.col("Z73") ** 2) +
                        (pl.col("Z75") ** 2) +
                        (pl.col("Z77") ** 2) +
                        (pl.col("Z82") ** 2) +
                        (pl.col("Z84") ** 2) +
                        (pl.col("Z86") ** 2) +
                        (pl.col("Z88") ** 2) +
                        (pl.col("Z91") ** 2) +
                        (pl.col("Z93") ** 2) +
                        (pl.col("Z95") ** 2) +
                        (pl.col("Z97") ** 2) +
                        (pl.col("Z99") ** 2)
                ).alias("E_Asymmetric"),
                (
                        (pl.col("Z00") ** 2) +
                        (pl.col("Z11") ** 2) +
                        (pl.col("Z20") ** 2) +
                        (pl.col("Z22") ** 2) +
                        (pl.col("Z31") ** 2) +
                        (pl.col("Z33") ** 2) +
                        (pl.col("Z40") ** 2) +
                        (pl.col("Z42") ** 2) +
                        (pl.col("Z44") ** 2) +
                        (pl.col("Z51") ** 2) +
                        (pl.col("Z53") ** 2) +
                        (pl.col("Z55") ** 2) +
                        (pl.col("Z60") ** 2) +
                        (pl.col("Z62") ** 2) +
                        (pl.col("Z64") ** 2) +
                        (pl.col("Z66") ** 2) +
                        (pl.col("Z71") ** 2) +
                        (pl.col("Z73") ** 2) +
                        (pl.col("Z75") ** 2) +
                        (pl.col("Z77") ** 2) +
                        (pl.col("Z80") ** 2) +
                        (pl.col("Z82") ** 2) +
                        (pl.col("Z84") ** 2) +
                        (pl.col("Z86") ** 2) +
                        (pl.col("Z88") ** 2) +
                        (pl.col("Z91") ** 2) +
                        (pl.col("Z93") ** 2) +
                        (pl.col("Z95") ** 2) +
                        (pl.col("Z97") ** 2) +
                        (pl.col("Z99") ** 2)
                ).alias("E_Total")
            )
            .with_columns(
                (pl.col("E_Trilobed") / pl.col("E_Total")).alias("Trilobed"),
                (pl.col("E_Asymmetric") / pl.col("E_Total")).alias("Asymmetric")
            )
            .select(["Cell_ID", "Trilobed", "Asymmetric"])
        )

        plate_qc_df = plate_qc_df.join(additional_features, on="Cell_ID")
        qc_dfs.append(plate_qc_df)
        conn.close()

    qc_df = (
        pl
        .concat(items=qc_dfs, how='vertical')
        .rename({"Cell_AreaShape_Center_X": "Center_X", "Cell_AreaShape_Center_Y": "Center_Y"})
    )

    # Add segmentation mask paths
    qc_df = (
        pl
        .read_csv(coordinates_path)
        .join(qc_df, on=["Cell_ID", "Center_X", "Center_Y"])
    )

    # Add additional features for QC based on existing features (if applicable)
    new_features = []
    necessary_features = [
        "Cell_AreaShape_Area", "Cell_AreaShape_Perimeter",
        "Cell_AreaShape_MajorAxisLength", "Cell_AreaShape_MinorAxisLength",
        "Cell_AreaShape_MaxFeretDiameter", "Cell_AreaShape_MinFeretDiameter"
    ]

    if all(necessary_feature in qc_features for necessary_feature in necessary_features):
        new_features = [
            "Area_Over_Perimeter",
            "MajorAxisLength_Over_MinorAxisLength",
            "MaxFeretDiameter_Over_MinFeretDiameter"
        ]

        qc_df = (
            qc_df
            .with_columns(
                (
                    pl.col("Cell_AreaShape_Area") / pl.col("Cell_AreaShape_Perimeter")
                 ).alias("Area_Over_Perimeter"),
                (
                    pl.col("Cell_AreaShape_MajorAxisLength") / pl.col("Cell_AreaShape_MinorAxisLength")
                ).alias("MajorAxisLength_Over_MinorAxisLength"),
                (
                    pl.col("Cell_AreaShape_MaxFeretDiameter") / pl.col("Cell_AreaShape_MinFeretDiameter")
                 ).alias("MaxFeretDiameter_Over_MinFeretDiameter")
            )
        )

        # In the new features, some values can be inf due to division by zero. Turn these to NaNs.
        qc_df = (
            qc_df
            .with_columns([
                pl
                .when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
                for col in new_features
            ]))

    # Save raw feature values
    qc_df.write_csv(file=f"{qc_directory}/raw_cell_qc_features.csv")

    # Standardize features so all features are on the same scale. Here, imputation is done
    # by replacing NaNs with the column mean (otherwise, StandardScaler will not work).

    cell_info_df = qc_df.select(
        ["Image_Path", "Center_X", "Center_Y"] + cell_info_features[:-2])  # exclude x/y coord from cell_info_features
    qc_features_df = qc_df.select(qc_features + new_features).to_numpy()

    # Impute missing values (replace NaNs with column mean)
    imputer = SimpleImputer(strategy="mean")
    qc_features_df = pl.DataFrame(imputer.fit_transform(qc_features_df))
    qc_features_df.columns = qc_features + new_features

    # Standardize features
    scaler = StandardScaler()
    scaled_data = pl.DataFrame(scaler.fit_transform(qc_features_df))
    scaled_data.columns = qc_features_df.columns

    # Save standardized features
    qc_df_scaled = pl.concat([cell_info_df, scaled_data], how='horizontal')
    qc_df_scaled.write_csv(file=f"{qc_directory}/scaled_cell_qc_features.csv")

    return qc_df, qc_df_scaled


def create_nuclear_qc_raw_scaled_df(database_directory, coordinates_path, qc_directory):
    """
    Create a dataframe with cells from all three reps, as well as QC features of interest. Create
    a second dataframe with scaled feature values (z-scores). Include the segmentation mask paths so they can be viewed with Single Cell Tool.

    Args:
        database_directory (str): path to directory with all plate databases
        coordinates_path (str): path to file with cell overlay coordinates
        qc_directory (str): path to directory to write qc files to
    """

    qc_features = [
        'Nuclei_Distance_Minimum_Cell'
    ]

    # Read all plate databases and get cell info + qc feature columns, combine
    databases = [f"{database_directory}/{db_name}" for db_name in os.listdir(database_directory)]
    qc_dfs = []

    for db_path in databases:
        conn = sqlite3.connect(db_path)

        plate_qc_df = pl.read_database(
            query=f"""
                    SELECT 
                        Per_Nuclei.Cell_ID, 
                        Nuclei_Number_Object_Number, 
                        {', '.join(qc_features)},
                        Nuclei_AreaShape_Area / Cell_AreaShape_Area AS Nuclear_Area_Over_Cell_Area
                    FROM Per_Nuclei
                    JOIN (SELECT Cell_ID, Cell_AreaShape_Area FROM Per_Cell) pc
                        ON Per_Nuclei.Cell_ID = pc.Cell_ID;
                    """,
            connection=conn)
        qc_dfs.append(plate_qc_df)

        conn.close()

    qc_df = (
        pl
        .concat(items=qc_dfs, how='vertical')
    )

    # Add segmentation mask paths
    qc_df = (
        pl
        .read_csv(coordinates_path)
        .join(qc_df, on=["Cell_ID"])
    )
    qc_df.write_csv(file=f"{qc_directory}/raw_nucleus_qc_features.csv")

    # Standardize features so all features are on the same scale. Here, imputation is done
    # by replacing NaNs with the column mean (otherwise, StandardScaler will not work).
    qc_features = qc_features + ["Nuclear_Area_Over_Cell_Area"]

    cell_info_df = qc_df.select(
        ["Image_Path", "Center_X", "Center_Y"] + ["Cell_ID", "Nuclei_Number_Object_Number"])
    qc_features_df = qc_df.select(qc_features).to_numpy()

    # Impute missing values (replace NaNs with column mean)
    imputer = SimpleImputer(strategy="mean")
    qc_features_df = pl.DataFrame(imputer.fit_transform(qc_features_df))
    qc_features_df.columns = qc_features

    # Standardize features
    scaler = StandardScaler()
    scaled_data = pl.DataFrame(scaler.fit_transform(qc_features_df))
    scaled_data.columns = qc_features_df.columns

    # Save standardized features
    qc_df_scaled = pl.concat([cell_info_df, scaled_data], how='horizontal')
    qc_df_scaled.write_csv(file=f"{qc_directory}/scaled_nucleus_qc_features.csv")

    return qc_df, qc_df_scaled


if __name__ == '__main__':
    if not os.path.exists(args.qc_directory):
        os.makedirs(args.qc_directory)

    # Cell features and histograms
    cell_qc_df_raw, cell_qc_df_scaled = create_cell_qc_raw_scaled_df(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)

    cell_qc_df_raw = cell_qc_df_raw.select(["Trilobed", "Asymmetric"])
    feature_distributions_matrix(
        qc_features=cell_qc_df_raw,
        qc_directory=args.qc_directory,
        output_figure_name="Cells_qc_feature_distributions_raw")

    cell_qc_df_scaled = cell_qc_df_scaled.drop(
        ["Image_Path", "Center_X", "Center_Y", "Cell_ID", "ORF", "Name", "Strain_ID", "Trilobed", "Asymmetric"])
    feature_distributions_matrix(
        qc_features=cell_qc_df_scaled,
        qc_directory=args.qc_directory,
        output_figure_name="Cells_qc_feature_distributions_scaled")

    # Nuclear features and histograms
    nuclear_qc_df_raw, _ = create_nuclear_qc_raw_scaled_df(
        database_directory=args.database_directory,
        coordinates_path=args.cell_coordinates,
        qc_directory=args.qc_directory)

    nuclear_qc_df_raw = (
        nuclear_qc_df_raw
        .select(
            ["Nuclei_Distance_Minimum_Cell", "Nuclear_Area_Over_Cell_Area"]
        )
        .with_columns(
            (
                pl
                .when(pl.col("Nuclear_Area_Over_Cell_Area") > 1)
                .then(1)
                .otherwise(pl.col("Nuclear_Area_Over_Cell_Area"))
            ).alias("Nuclear_Area_Over_Cell_Area")
        )
        .rename(
            {
                "Nuclei_Distance_Minimum_Cell": "Nuclei_Distance_Minimum_Cell (Raw)",
                "Nuclear_Area_Over_Cell_Area": "Nuclear_Area_Over_Cell_Area (Raw)"
            }
        )
    )

    feature_distributions_matrix(
        qc_features=nuclear_qc_df_raw,
        qc_directory=args.qc_directory,
        output_figure_name="Nuclei_qc_feature_distributions")

    print("Complete.")
