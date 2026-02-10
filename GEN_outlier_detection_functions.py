import os
import polars as pl
from scipy import stats
import sqlite3
from functools import reduce
from statistics import stdev, mean


def scale_compartment_feature(db_path, table_name, feature_name, output_dir, plate, compartment_name, cell_cycle_stages=["G1", "SG2", "MAT"], feature_table=""):
    """
    First identifies cells that have an abnormal number of compartment masks, then scales the compartment sizes according to
    wildtype for each cell cycle stage while ignoring these cells from the calculation. Exports the results to be used in a
    separate function for identifying outliers.

    Args:
        db_path (str): path to database with cell/compartment information
        table_name (str): name of table with compartment information
        feature_name (str): name of feature column to extract data from
        output_dir (str): where to save output files
        plate (str): plate identifier for saving files
        compartment_name (str): name of compartment used in saving files
        cell_cycle_stages (list, optional): cell cycle stages to filter by; defaults to ["G1", "SG2", "MAT"]
        feature_table (pl.DataFrame, optional): if no db is provided, information is read from this table instead. MUST have columns Replicate, Condition, Row, Column, Cell_ID, ORF, Name, Strain_ID, Predicted_Label, and feature of interest.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read data
    if isinstance(feature_table, pl.DataFrame):
        all_cells = (
            feature_table
            .filter(pl.col("Predicted_Label").is_in(cell_cycle_stages))
            # Copy the <feature_name> column, then update the new column in each cc_stage loop
            .with_columns(
                (pl.col(feature_name)).alias(f"{feature_name}_Scaled")
            )
        )

    else:
        conn = sqlite3.connect(db_path)

        # Scale <feature_name> according to wildtype values and convert to p-value
        all_cells = (
            pl
            .read_database(
                query=f"SELECT Replicate, Condition, Row, Column, Cell_ID, ORF, Name, Strain_ID, Predicted_Label, {feature_name} FROM {table_name};",
                connection=conn)
            .filter(pl.col("Predicted_Label").is_in(cell_cycle_stages))
            # Copy the <feature_name> column, then update the new column in each cc_stage loop
            .with_columns(
                (pl.col(feature_name)).alias(f"{feature_name}_Scaled")
            )
        )
        conn.close()


    # Get replicates and conditions (features will be scaled per-plate# x per-rep x per-condition x per-cc)
    replicates = (
        all_cells
        .get_column("Replicate")
        .unique()
    )

    conditions = (
        all_cells
        .get_column("Condition")
        .unique()
    )

    for condition in conditions:
        for replicate in replicates:
            for cc_stage in cell_cycle_stages:
                wt_cc_comps = (
                    all_cells
                    .filter(
                        (pl.col("Condition") == condition) &
                        (pl.col("Replicate") == replicate) &
                        (pl.col("Predicted_Label") == cc_stage) &
                        (pl.col("ORF").is_in(['YOR202W', 'YMR271C']))
                    )
                )

                if wt_cc_comps.shape[0] < 2: # Check if there are any wildtypes on plates (some plates don't have enough controls)
                    print(f"Skipping {plate} - {condition}C - {replicate} - {cc_stage} due to insufficient number of wildtype controls.")
                    continue
                wt_mean_distance = mean(wt_cc_comps[feature_name])
                wt_stdev_distance = stdev(wt_cc_comps[feature_name])

                all_cells = (
                    all_cells
                    # For cells in the correct cc_stage, update the *_Scaled column by scaling
                    # the corresponding values in the unscaled column according to wildtype cells in that
                    # cc_stage
                    .with_columns(
                        (
                            pl
                            .when(
                                (pl.col("Condition") == condition) &
                                (pl.col("Replicate") == replicate) &
                                (pl.col("Predicted_Label") == cc_stage)
                            )
                            .then(((pl.col(f"{feature_name}")) - wt_mean_distance) / wt_stdev_distance)
                            .otherwise(pl.col(f"{feature_name}_Scaled"))
                        ).alias(f"{feature_name}_Scaled")
                    )
                )

    # Convert scaled values to one-sided p-values (depends on orientation of scaled value -- negative values get
    # left-sided p-values while positive values get right-sided p-values)
    all_cells = (
        all_cells
        .with_columns(
            pl
            .when(pl.col(f"{feature_name}_Scaled") < 0)
            .then(pl.col(f"{feature_name}_Scaled").map_elements(lambda z: stats.norm.cdf(z), return_dtype=pl.Float64))
            .otherwise(
                pl.col(f"{feature_name}_Scaled").map_elements(lambda z: stats.norm.sf(z), return_dtype=pl.Float64))
            .alias("pval")
        ))

    # Export
    (
        all_cells
        .select([
            "Replicate", "Condition", "Row", "Column", "Cell_ID", "ORF", "Name", "Strain_ID", "Predicted_Label",
            feature_name, f"{feature_name}_Scaled",
            "pval"])
        .write_csv(f"{output_dir}/{plate}_{compartment_name}_scaled_pval.csv")
    )

    return all_cells


def identify_outlier_cells(feature_pvals, scaled_col_name, output_dir, plate, compartment_name, pval_cutoff=0.05, right_sided_outliers=True, excluded_cc_stages=[]):
    """
    Obtain and save outlier cells from a given cell population based on their calculated p-values.

    Args:
        feature_pvals (pl.DataFrame): dataframe that has cell cycle stage, Cell ID, and cell size pvalue
        scaled_col_name (str): name of column with scaled values for each cell
        output_dir (str): where to save output files
        plate (str): plate identifier for saving files
        compartment_name (str): name of compartment used in saving files
        pval_cutoff (float, optional): p-value cutoff for identifying outliers; cells with p-value below this are tagged outlier
        right_sided_outliers (bool): when set to true, only looks at cells with positive Z-Scores (indicating that they're above the wt distribution) when looking for outliers
        excluded_cc_stages (list(str, optional)): if there are any cell cycle stages from which no cells are considered to be outliers, exclude them
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Separate outlier cells from rest of population based on cutoff and detection strategy (above or below cutoff)
    if right_sided_outliers:
        outlier_cells = (
            feature_pvals
            .filter(
                (pl.col(scaled_col_name) > 0) &
                (pl.col("pval") < pval_cutoff) &
                (~pl.col("Predicted_Label").is_in(excluded_cc_stages))
            )
        )
    else:
        outlier_cells = (
            feature_pvals
            .filter(
                (pl.col(scaled_col_name) < 0) &
                (pl.col("pval") < pval_cutoff) &
                (~pl.col("Predicted_Label").is_in(excluded_cc_stages))
            )
        )

    # Export all outlier cells
    outlier_cells.write_csv(f"{output_dir}/{plate}_{compartment_name}_outlier_cells.csv")

    return outlier_cells


def calculate_strain_penetrances(all_cells, all_outlier_cells, output_dir, plate, compartment_name, cell_cycle_stages=["G1", "SG2", "MAT"]):
    """
    Calculates penetrance of abnormal compartment size phenotype for a given compartment for wildtype controls and mutants.
    For foci compartments, this is typically characterized by having at least one large/small foci in a cell. This is done a
    per-replicate x per-cell cycle stage basis.

    Args:
        all_cells (pl.DataFrame): dataframe that has replicate, cell cycle stage, and Cell ID for all cells
        all_outlier_cells (pl.DataFrame): dataframe that has replicate, cell cycle stage, and Cell ID for all outlier cells
        output_dir (str): where to save output files
        plate (str): plate identifier for saving files
        compartment_name (str): name of compartment used in saving files
        cell_cycle_stages (list, optional): cell cycle stages to filter by; defaults to ["G1", "SG2", "MAT"]
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get unique replicates in dataframe
    replicates = (
        all_cells
        .get_column("Replicate")
        .unique()
        .to_list()
    )
    replicates.sort()

    # Get unique strains in dataframe
    strains = set(all_cells["Strain_ID"].to_list())

    # Initialize the per-strain penetrance dataframe
    penetrance_dict = {"ORF": [], "Name": [], "Strain_ID": [], "Overall_Penetrance": []}
    for replicate in replicates:
        penetrance_dict[f"Overall_{replicate}_Penetrance"] = []
    for cc_stage in cell_cycle_stages:
        penetrance_dict[f"Overall_{cc_stage}_Penetrance"] = []
        for replicate in replicates:
            penetrance_dict[f"{replicate}_{cc_stage}_Penetrance"] = []

    # Get overall, per-cc, and per-cc-rep penetrances for each strain
    for strain in strains:
        # Add basic strain information
        penetrance_dict["ORF"].append(all_cells.filter(pl.col("Strain_ID") == strain)["ORF"].item(0))
        penetrance_dict["Name"].append(all_cells.filter(pl.col("Strain_ID") == strain)["Name"].item(0))
        penetrance_dict["Strain_ID"].append(strain)

        # Get cell counts for calculating overall penetrance
        all_strain_outlier_cells = (
            all_outlier_cells
            .filter(
                (pl.col("Strain_ID") == strain)
            )
            .height
        )

        all_strain_cells = (
            all_cells
            .filter(
                (pl.col("Strain_ID") == strain)
            )
            .height
        )

        # Calculate overall penetrance
        try:
            penetrance_dict[f"Overall_Penetrance"].append(all_strain_outlier_cells / all_strain_cells)
        except ZeroDivisionError:
            penetrance_dict[f"Overall_Penetrance"].append(-1.0)

        for replicate in replicates:
            all_strain_rep_outlier_cells = (
                all_outlier_cells
                .filter(
                    (pl.col("Strain_ID") == strain) &
                    (pl.col("Replicate") == replicate)
                )
                .height
            )

            all_strain_rep_cells = (
                all_cells
                .filter(
                    (pl.col("Strain_ID") == strain) &
                    (pl.col("Replicate") == replicate)
                )
                .height
            )

            # Calculate overall penetrance
            try:
                penetrance_dict[f"Overall_{replicate}_Penetrance"].append(all_strain_rep_outlier_cells / all_strain_rep_cells)
            except ZeroDivisionError:
                penetrance_dict[f"Overall_{replicate}_Penetrance"].append(-1.0)

        for cc_stage in cell_cycle_stages:

            all_strain_outlier_cc_cells = (
                all_outlier_cells
                .filter(
                    (pl.col("Strain_ID") == strain) &
                    (pl.col("Predicted_Label") == cc_stage)
                )
                .height
            )

            all_strain_cc_cells = (
                all_cells
                .filter(
                    (pl.col("Strain_ID") == strain) &
                    (pl.col("Predicted_Label") == cc_stage)
                )
                .height
            )

            # Calculate overall cell cycle stage penetrance
            try:
                penetrance_dict[f"Overall_{cc_stage}_Penetrance"].append(all_strain_outlier_cc_cells / all_strain_cc_cells)
            except ZeroDivisionError:
                penetrance_dict[f"Overall_{cc_stage}_Penetrance"].append(-1.0)

            for replicate in replicates:

                all_strain_outlier_cc_rep_cells = (
                    all_outlier_cells
                    .filter(
                        (pl.col("Strain_ID") == strain) &
                        (pl.col("Predicted_Label") == cc_stage) &
                        (pl.col("Replicate") == replicate)
                    )
                    .height
                )

                all_strain_cc_rep_cells = (
                    all_cells
                    .filter(
                        (pl.col("Strain_ID") == strain) &
                        (pl.col("Predicted_Label") == cc_stage) &
                        (pl.col("Replicate") == replicate)
                    )
                    .height
                )
                # Calculate cell cycle stage penetrance for replicate
                try:
                    penetrance_dict[f"{replicate}_{cc_stage}_Penetrance"].append(all_strain_outlier_cc_rep_cells / all_strain_cc_rep_cells)
                except ZeroDivisionError:
                    penetrance_dict[f"{replicate}_{cc_stage}_Penetrance"].append(-1.0)

    # Export penetrance dataframe
    penetrance_table = pl.DataFrame(penetrance_dict)
    (
        penetrance_table
        .sort(["Overall_Penetrance"], descending=True)
        .write_csv(f"{output_dir}/{plate}_{compartment_name}_strain_penetrances.csv")
    )

    return penetrance_table


def tabulate_strain_cell_counts(all_cells, all_outlier_cells, output_dir, plate, compartment_name, cell_cycle_stages=["G1", "SG2", "MAT"]):
    """
    Calculates penetrance of abnormal compartment size phenotype for a given compartment for wildtype controls and mutants.
    For foci compartments, this is typically characterized by having at least one large/small foci in a cell. This is done a
    per-replicate x per-cell cycle stage basis.

    Args:
        all_cells (pl.DataFrame): dataframe that has replicate, cell cycle stage, and Cell ID for all cells
        all_outlier_cells (pl.DataFrame): dataframe that has replicate, cell cycle stage, and Cell ID for all outlier cells
        output_dir (str): where to save output files
        plate (str): plate identifier for saving files
        compartment_name (str): name of compartment used in saving files
        cell_cycle_stages (list, optional): cell cycle stages to filter by; defaults to ["G1", "SG2", "MAT"]
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get unique replicates in dataframe
    replicates = (
        all_cells
        .get_column("Replicate")
        .unique()
        .to_list()
    )
    replicates.sort()

    # Get unique strains in dataframe
    strains = set(all_cells["Strain_ID"].to_list())

    # Initialize the per-strain penetrance dataframe
    cell_count_dict = {"ORF": [], "Name": [], "Strain_ID": [], "Total_Cells": [], "Total_Outlier_Cells": []}
    for replicate in replicates:
        cell_count_dict[f"Total_{replicate}_Cells"] = []
        cell_count_dict[f"Total_{replicate}_Outlier_Cells"] = []
    for cc_stage in cell_cycle_stages:
        cell_count_dict[f"Total_{cc_stage}_Cells"] = []
        cell_count_dict[f"Total_{cc_stage}_Outlier_Cells"] = []
        for replicate in replicates:
            cell_count_dict[f"{replicate}_{cc_stage}_Cells"] = []
            cell_count_dict[f"{replicate}_{cc_stage}_Outlier_Cells"] = []

    # Get overall, per-cc, and per-cc-rep penetrances for each strain
    for strain in strains:
        # Add basic strain information
        cell_count_dict["ORF"].append(all_cells.filter(pl.col("Strain_ID") == strain)["ORF"].item(0))
        cell_count_dict["Name"].append(all_cells.filter(pl.col("Strain_ID") == strain)["Name"].item(0))
        cell_count_dict["Strain_ID"].append(strain)

        # Get cell counts for calculating overall penetrance
        all_strain_outlier_cells = (
            all_outlier_cells
            .filter(
                (pl.col("Strain_ID") == strain)
            )
            .height
        )

        all_strain_cells = (
            all_cells
            .filter(
                (pl.col("Strain_ID") == strain)
            )
            .height
        )

        # Calculate overall penetrance
        cell_count_dict[f"Total_Cells"].append(all_strain_cells)
        cell_count_dict[f"Total_Outlier_Cells"].append(all_strain_outlier_cells)

        for replicate in replicates:
            all_strain_rep_outlier_cells = (
                all_outlier_cells
                .filter(
                    (pl.col("Strain_ID") == strain) &
                    (pl.col("Replicate") == replicate)
                )
                .height
            )

            all_strain_rep_cells = (
                all_cells
                .filter(
                    (pl.col("Strain_ID") == strain) &
                    (pl.col("Replicate") == replicate)
                )
                .height
            )

            # Calculate overall penetrance
            cell_count_dict[f"Total_{replicate}_Cells"].append(all_strain_rep_cells)
            cell_count_dict[f"Total_{replicate}_Outlier_Cells"].append(all_strain_rep_outlier_cells)

        for cc_stage in cell_cycle_stages:

            all_strain_outlier_cc_cells = (
                all_outlier_cells
                .filter(
                    (pl.col("Strain_ID") == strain) &
                    (pl.col("Predicted_Label") == cc_stage)
                )
                .height
            )

            all_strain_cc_cells = (
                all_cells
                .filter(
                    (pl.col("Strain_ID") == strain) &
                    (pl.col("Predicted_Label") == cc_stage)
                )
                .height
            )

            # Calculate overall cell cycle stage penetrance
            cell_count_dict[f"Total_{cc_stage}_Cells"].append(all_strain_cc_cells)
            cell_count_dict[f"Total_{cc_stage}_Outlier_Cells"].append(all_strain_outlier_cc_cells)

            for replicate in replicates:

                all_strain_outlier_cc_rep_cells = (
                    all_outlier_cells
                    .filter(
                        (pl.col("Strain_ID") == strain) &
                        (pl.col("Predicted_Label") == cc_stage) &
                        (pl.col("Replicate") == replicate)
                    )
                    .height
                )

                all_strain_cc_rep_cells = (
                    all_cells
                    .filter(
                        (pl.col("Strain_ID") == strain) &
                        (pl.col("Predicted_Label") == cc_stage) &
                        (pl.col("Replicate") == replicate)
                    )
                    .height
                )
                # Calculate cell cycle stage penetrance for replicate
                cell_count_dict[f"{replicate}_{cc_stage}_Cells"].append(all_strain_cc_rep_cells)
                cell_count_dict[f"{replicate}_{cc_stage}_Outlier_Cells"].append(all_strain_outlier_cc_rep_cells)

    # Export penetrance dataframe
    (
        pl
        .DataFrame(cell_count_dict)
        .sort(["Total_Cells"], descending=True)
        .write_csv(f"{output_dir}/{plate}_{compartment_name}_strain_cell_counts.csv")
    )


def get_strain_hits(all_cells, outlier_cells, penetrance_table, wt_pens_dir, output_dir, plate, cc_stages=["G1", "SG2", "MAT"], percentile_cutoff=0.95):
    """
    For a given compartment phenotype, identifies mutants that display phenotype at a higher/lower incidence relative
    to wildtype controls. Hit mutants are those that have CC penetrance above 95th wt percentile or below 5th percentile
    by default, but this can be changed.

    Args:
        all_cells (pl.DataFrame): polars dataframe with information on all cells on plate
        outlier_cells (str): path to csv file with Cell_IDs of all outlier cells
        penetrance_table (pl.DataFrame): polars dataframe with per-CC penetrances for all strains on plate
        wt_pens_dir (str): where to save per-well wildtype penetrances
        output_dir (str): where to save output file to
        plate (str): plate identifier for saving files
        cc_stages (list(str), optional): list of cell cycle stages to filter by; defaults to ["G1", "SG2", "MAT"]
        percentile_cutoff (float): threshold to use for obtaining significant penetrance cutoff from wildtype distribution
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(wt_pens_dir):
        os.makedirs(wt_pens_dir)

    # To filter hits, I need to create a distribution of wildtype penetrances. In the penetrances file, I only have the
    # aggregate wildtype penetrance across all wells. However, I need to get the wildtype penetrance in each well to
    # create a distribution. Unique wells can be identified using Replicate/Row/Column.

    all_wildtype_cells = (
        all_cells
        .filter(
            (pl.col("ORF").is_in(['YOR202W', 'YMR271C']))
        )
    )

    all_wildtype_outliers = (
        outlier_cells
        .filter(
            (pl.col("ORF").is_in(['YOR202W', 'YMR271C']))
        )
    )

    # Create a log file for saving thresholds used for each cell cycle stage
    log_dict = {"Plate": plate}

    strain_hits_list = []
    for cc_stage in cc_stages:
        all_wildtype_cells_cc = all_wildtype_cells.filter(pl.col("Predicted_Label") == cc_stage)
        all_wildtype_outliers_cc = all_wildtype_outliers.filter(pl.col("Predicted_Label") == cc_stage)

        # Add an Is_Outlier column to all_wildtype_cells using information from all_wildtype_outliers
        all_wildtype_cells_cc = (
            all_wildtype_cells_cc
            .with_columns(
                (
                    pl
                    .when(pl.col("Cell_ID").is_in(all_wildtype_outliers_cc["Cell_ID"]))
                    .then(True)
                    .otherwise(False))
                .alias("Is_Outlier")
            )
        )

        wildtype_well_penetrances = (
            all_wildtype_cells_cc
            .group_by(["Replicate", "Row", "Column"])  # group by wells
            .agg(
                (pl.col("Is_Outlier") == True).sum().alias("Outliers"),  # get number of outliers and total cells
                (pl.len().alias("Total"))
            )
            .with_columns((pl.col("Outliers") / pl.col("Total")).alias("Penetrance"))  # calculate well penetrance
        )
        wildtype_well_penetrances.write_csv(f"{wt_pens_dir}/{plate}_per_well_wt_penetrances.csv")

        # Get cutoffs
        pen_cutoff = wildtype_well_penetrances["Penetrance"].quantile(quantile=percentile_cutoff, interpolation="nearest")

        log_dict[f"{cc_stage}_Cutoff"] = pen_cutoff

        # Use cutoffs to identify hit genes for each given CC stage
        strains_past_wt_cutoff = (
            penetrance_table
            .filter(pl.col(f"Overall_{cc_stage}_Penetrance") > pen_cutoff)
            .with_columns(
                (pl.lit(cc_stage).alias("CC_Stage")),
                (pl.col(f"Overall_{cc_stage}_Penetrance").alias("Penetrance"))
            )
            .select(["ORF", "Name", "Strain_ID", "CC_Stage", "Penetrance"])
        )
        strain_hits_list.append(strains_past_wt_cutoff)

    (
        pl
        .concat(items=strain_hits_list, how="vertical")
        .write_csv(f"{output_dir}/{plate}_hit_strains.csv")
    )

    (
        pl
        .DataFrame(log_dict)
        .write_csv(f"{output_dir}/{plate}_thresholds_used_for_getting_strain_hits.csv")
    )


def run_all_functions(
        db_path,
        all_cells,
        compartment_table_name,
        feature_name,
        scaled_feature_dir,
        outlier_objects_dir,
        penetrance_dir,
        cell_count_dir,
        strain_hits_dir,
        wt_pens_dir,
        plate,
        compartment_name,
        excluded_outlier_cc_stages=[],
        feature_table="",
        cell_cycle_stages=["G1", "SG2", "MAT"],
        outlier_pval_cutoff=0.05,
        right_sided_outliers=True,
        percentile_cutoff=0.95):
    """
    Runs all functions (scale_compartment_feature(), identify_outlier_cells(), calculate_strain_penetrances(),
    tabulate_strain_cell_counts(), tabulate_strain_cell_counts(), get_strain_hits()) at once.

    Args:
        db_path (str): path to database being processed
        all_cells (pl.DataFrame): dataframe of all cells and their information
        compartment_table_name (str): name of table in db_path that has features for compartment of interest
        feature_name (str): name of feature used for scaling and outlier detection. Can be a column in db_path or a dataframe input
        scaled_feature_dir (str): where to save spreadsheet with scaled feature for each object
        outlier_objects_dir (str): where to save spreadsheet with list of cells or outlier objects
        penetrance_dir (str): where to save spreadsheet with strain penetrances for phenotype in question
        cell_count_dir (str): where to save spreadsheet with cell counts for each strain
        strain_hits_dir (str): where to save spreadsheet with hit strains
        wt_pens_dir (str): where to save per-well wildtype penetrances
        plate (str): plate identifier for saving files
        compartment_name (str): compartment identifier for saving files
        excluded_outlier_cc_stages (list(str, optional)): if there are any cell cycle stages from which no cells are considered to be outliers, exclude them
        feature_table (pl.DataFrame, optional): alternative to database; can just provide a suitable table instead. Defaults to ""
        cell_cycle_stages (list(str)): what cell cycle stages to consider when getting phenotype info. Defaults to all cell cycle stages
        outlier_pval_cutoff (float): what p-val cutoff to use from scaled feature p-values when getting outlier objects. Defaults to 0.05
        right_sided_outliers (bool): p-vals are left- and right-sided; choose True to look at right-sided pvals when getting outlier objects and False to look at left-sided pvals. Defaults to True
        percentile_cutoff (float): what cutoff to use when getting the penetrance cutoff from a wildtype distribution prior to identifying significant strains. Defaults to 0.95
    """

    all_objects_scaled = scale_compartment_feature(
        db_path=db_path,
        table_name=compartment_table_name,
        feature_name=feature_name,
        output_dir=scaled_feature_dir,
        plate=plate,
        compartment_name=compartment_name,
        cell_cycle_stages=cell_cycle_stages,
        feature_table=feature_table)

    outlier_objects = identify_outlier_cells(
        feature_pvals=all_objects_scaled,
        scaled_col_name=f"{feature_name}_Scaled",
        output_dir=outlier_objects_dir,
        plate=plate,
        compartment_name=compartment_name,
        pval_cutoff=outlier_pval_cutoff,
        right_sided_outliers=right_sided_outliers,
        excluded_cc_stages=excluded_outlier_cc_stages)

    penetrance_table = calculate_strain_penetrances(
        all_cells=all_cells,
        all_outlier_cells=outlier_objects,
        output_dir=penetrance_dir,
        plate=plate,
        compartment_name=compartment_name,
        cell_cycle_stages=cell_cycle_stages)

    tabulate_strain_cell_counts(
        all_cells=all_cells,
        all_outlier_cells=outlier_objects,
        output_dir=cell_count_dir,
        plate=plate,
        compartment_name=compartment_name,
        cell_cycle_stages=cell_cycle_stages)

    get_strain_hits(
        all_cells=all_cells,
        outlier_cells=outlier_objects,
        penetrance_table=penetrance_table,
        wt_pens_dir=wt_pens_dir,
        output_dir=strain_hits_dir,
        plate=plate,
        cc_stages=cell_cycle_stages,
        percentile_cutoff=percentile_cutoff)


def combine_output_phenotypes_from_plate(phenotype_outliers, db_path, output_dir, plate):
    """
    Once outlier detection is done, this function (1) aggregates Cell_IDs with their assigned outlier phenotype(s) and number
    of assigned phenotypes, then combines overall penetrance, per-CC penetrance, per-phenotype penetrance, and per-cc-phenotype
    penetrance. This is done for each specific plate.

    Args:
        phenotype_outliers (dict(str: str)): dictionary where keys are custom phenotype labels and values are paths to *_outlier_cells.csv files
        db_path (str): path to database with cell information
        output_dir (str): where to save output files to
        plate (str): plate identifier for saving files
    """
    if not os.path.exists(f"{output_dir}/aggregated_cell_outlier_data"):
        os.makedirs(f"{output_dir}/aggregated_cell_outlier_data")

    if not os.path.exists(f"{output_dir}/aggregated_penetrance_data"):
        os.makedirs(f"{output_dir}/aggregated_penetrance_data")

    cc_stages = ["G1", "SG2", "MAT"]
    phenotypes = list(phenotype_outliers.keys())
    plate_num = int(plate.split("Plate")[1])

    # Combine all *_outlier_cells.csv files, attach custom phenotype labels, and aggregate assigned labels for each Cell_ID
    all_data = (
        pl
        .concat(
            items=[
                (
                    pl
                    .read_csv(phenotype_outliers[phenotype])
                    .select(["Replicate", "Row", "Column", "Cell_ID", "ORF", "Name", "Strain_ID", "Predicted_Label"])
                    .with_columns(pl.lit(phenotype).alias("Cell_Phenotype"))
                )
                for phenotype in phenotype_outliers
            ],
            how="vertical"
        )
        .with_columns( # if Name is NULL, this causes a lot of issues with joins; so I changed NULLs to ""
            (
                pl
                .when(pl.col("Name").is_null())
                .then(pl.lit(""))
                .otherwise(pl.col("Name"))
            ).alias("Name")
        )
    )

    all_data_agg = (
        all_data
        .group_by(["Replicate", "Row", "Column", "Cell_ID", "ORF", "Name", "Strain_ID", "Predicted_Label"])
        .agg(
            pl.col("Cell_Phenotype"),
            pl.len().alias("Num_Cell_Phenotypes")
        )
        .with_columns(
            pl.format("{}",
                      pl.col("Cell_Phenotype").cast(pl.List(pl.String)).list.join(" | "))
        )
        .select(
            pl.lit(plate_num).alias("Plate"),
            pl.all()
        )
    )
    all_data_agg.write_csv(f"{output_dir}/aggregated_cell_outlier_data/{plate}_aggregated_cell_outlier_data.csv")


    # Get overall cell counts for each strain in each of its replicates for which data is available. Do this without CC and
    # with CC.
    conn = sqlite3.connect(db_path)
    cell_counts_no_cc = (
        pl
        .read_database(
            query="""
                    WITH iq1 AS (
    	                SELECT
    	                	Row,
    	                	Column,
    						ORF,
    						COALESCE(Name, "") AS Name,
    	                	Strain_ID,
    	                	COUNT(Cell_ID) AS Num_Cells
    	                FROM Per_Cell
    	                GROUP BY Row, Column, ORF, Strain_ID),

                    iq2 AS (
                    	SELECT DISTINCT
                    		Replicate,
                    		Strain_ID
                    	FROM Per_Cell)

                    SELECT 
                    	Replicate,
                    	Row,
                    	Column,
                    	ORF,
    					Name,
                    	iq1.Strain_ID,
                    	Num_Cells
                    FROM iq1
                    JOIN iq2 ON iq1.Strain_ID = iq2.Strain_ID;
                """,
            connection=conn
        )
        .with_columns(
            pl.col("Row").cast(pl.Int32).alias("Row"),
            pl.col("Column").cast(pl.Int32).alias("Column")
        )
    )

    cell_counts_with_cc = (
        pl
        .read_database(
            query="""
                    WITH iq1 AS (
    	                SELECT
    	                	Row,
    	                	Column,
    	                	ORF, 
    						COALESCE(Name, "") AS Name,
    	                	Strain_ID,
                            Predicted_Label,
    	                	COUNT(Cell_ID) AS Num_Cells
    	                FROM Per_Cell
    	                GROUP BY Row, Column, ORF, Strain_ID, Predicted_Label),

                    iq2 AS (
                    	SELECT DISTINCT
                    		Replicate,
                    		Strain_ID
                    	FROM Per_Cell)

                    SELECT 
                    	Replicate,
                    	Row,
                    	Column,
                    	ORF,
    					Name,
                    	iq1.Strain_ID,
                        Predicted_Label,
                    	Num_Cells
                    FROM iq1
                    JOIN iq2 ON iq1.Strain_ID = iq2.Strain_ID;
                """,
            connection=conn
        )
        .with_columns(
            pl.col("Row").cast(pl.Int32).alias("Row"),
            pl.col("Column").cast(pl.Int32).alias("Column")
        )
    )
    conn.close()


    # Overall penetrance
    overall_penetrance = (
        all_data
        .drop(["Predicted_Label", "Cell_Phenotype"])
        .unique()
        .group_by(["Replicate", "ORF", "Name", "Strain_ID", "Row", "Column"])
        .agg(pl.len().alias("Num_Outliers"))
        .join(cell_counts_no_cc, on=["Replicate", "Row", "Column", "ORF", "Name", "Strain_ID"], how="right")
        .group_by(["ORF", "Name", "Strain_ID", "Row", "Column"])
        .agg(
            (pl.col("Num_Outliers").sum() / (pl.col("Num_Cells").sum() / pl.len())).alias("Overall_Penetrance")
        )
    )

    # Overall CC penetrance
    overall_cc_penetrance = (
        all_data
        .drop(["Cell_Phenotype"])
        .unique()
        .group_by(["Replicate", "ORF", "Name", "Strain_ID", "Predicted_Label", "Row", "Column"])
        .agg(pl.len().alias("Num_Outliers"))
        .join(cell_counts_with_cc, on=["Replicate", "Row", "Column", "ORF", "Name", "Strain_ID", "Predicted_Label"],
              how="right")
        .with_columns(pl.col("Num_Outliers").fill_null(0))
        .group_by(["ORF", "Name", "Strain_ID", "Predicted_Label", "Row", "Column"])
        .agg(
            (pl.col("Num_Outliers").sum() / (pl.col("Num_Cells").sum() / pl.len())).alias("Penetrance")
        )
        .pivot(
            on="Predicted_Label",
            index=["ORF", "Name", "Strain_ID", "Row", "Column"],
            values="Penetrance")
        .rename({
            "G1": "Overall_G1_Penetrance",
            "SG2": "Overall_SG2_Penetrance",
            "MAT": "Overall_MAT_Penetrance"
        }
        )
        .select(
            ["ORF", "Name", "Strain_ID", "Row", "Column", "Overall_G1_Penetrance", "Overall_SG2_Penetrance", "Overall_MAT_Penetrance"])
    )


    # Per Phenotype penetrance
    rename_dict1 = {phenotype: f"Overall_{phenotype}_Penetrance" for phenotype in phenotype_outliers.keys()}
    select_list1 = ["ORF", "Name", "Strain_ID", "Row", "Column"] + [f"Overall_{phenotype}_Penetrance" for phenotype in phenotype_outliers.keys()]

    per_phenotype_penetrance = (
        all_data
        .group_by(["Replicate", "ORF", "Name", "Strain_ID", "Row", "Column", "Cell_Phenotype"])
        .agg(pl.len().alias("Num_Outliers"))
        .join(cell_counts_no_cc, on=["Replicate", "Row", "Column", "ORF", "Name", "Strain_ID"], how="right")
        .with_columns(pl.col("Num_Outliers").fill_null(0))
        .group_by(["ORF", "Name", "Strain_ID", "Row", "Column", "Cell_Phenotype"])
        .agg(
            (pl.col("Num_Outliers").sum() / (pl.col("Num_Cells").sum() / pl.len())).alias("Penetrance")
        )
        .pivot(
            on="Cell_Phenotype",
            index=["ORF", "Name", "Strain_ID", "Row", "Column"],
            values="Penetrance")
        .rename(rename_dict1)
        .select(select_list1)
    )

    # Per Phenotype by CC penetrance
    rename_dict2 = {
        f'{{"{cc_stage}","{phenotype}"}}': f"{phenotype}_{cc_stage}_Penetrance"
        for phenotype in phenotypes
        for cc_stage in cc_stages
    }
    select_list2 = ["ORF", "Name", "Strain_ID", "Row", "Column"] + [f"{phenotype}_{cc_stage}_Penetrance" for phenotype in phenotypes for cc_stage in cc_stages]

    per_phenotype_cc_penetrance = (
        all_data
        .group_by(["Replicate", "ORF", "Name", "Strain_ID", "Row", "Column", "Predicted_Label", "Cell_Phenotype"])
        .agg(pl.len().alias("Num_Outliers"))
        .join(cell_counts_with_cc, on=["Replicate", "Row", "Column", "ORF", "Name", "Strain_ID", "Predicted_Label"],
              how="right")
        .with_columns(pl.col("Num_Outliers").fill_null(0))
        .group_by(["ORF", "Name", "Strain_ID", "Row", "Column", "Predicted_Label", "Cell_Phenotype"])
        .agg(
            (pl.col("Num_Outliers").sum() / (pl.col("Num_Cells").sum() / pl.len())).alias("Penetrance")
        )
        .with_columns(pl.col("Cell_Phenotype").fill_null("Null"))
        .pivot(
            on=["Predicted_Label", "Cell_Phenotype"],
            index=["ORF", "Name", "Strain_ID", "Row", "Column"],
            values="Penetrance")
        .drop(['{"G1","Null"}', '{"SG2","Null"}', '{"MAT","Null"}'], strict=False)
        .rename(rename_dict2,strict=False)
    )

    for phenotype in phenotype_outliers.keys():
        for cc_stage in ["G1", "SG2", "MAT"]:
            if f"{phenotype}_{cc_stage}_Penetrance" not in per_phenotype_cc_penetrance.columns:
                per_phenotype_cc_penetrance = (
                    per_phenotype_cc_penetrance
                    .with_columns(pl.lit(0).alias(f"{phenotype}_{cc_stage}_Penetrance"))
                )

    per_phenotype_cc_penetrance = (
        per_phenotype_cc_penetrance
        .select(select_list2)
    )


    # Merge all dataframes together
    combined_df = reduce(
        lambda left, right: left.join(right, on=["ORF", "Name", "Strain_ID", "Row", "Column"], how="left"),
        [overall_penetrance, overall_cc_penetrance, per_phenotype_penetrance, per_phenotype_cc_penetrance]
    )
    # Turn all NULL penetrances to 0 for better interpretability
    for col in set(combined_df.columns) - set(["ORF", "Name", "Strain_ID", "Row", "Column"]):
        combined_df = (
            combined_df
            .with_columns(pl.col(col).fill_null(0))
        )

    # Add Plate column
    combined_df = (
        combined_df
        .select(
            pl.lit(plate_num).alias("Plate"),
            pl.all()
        )
    )

    combined_df.write_csv(f"{output_dir}/aggregated_penetrance_data/{plate}_aggregated_penetrance_data.csv")


def tabulate_compartment_masks_per_strain(db_path, compartment_name, plate, output_directory):
    """
    This function tabulates the number of compartment masks within a cell mask and writes to a CSV. This is done on a
    cell cycle stage basis between wildtype and mutant cells.

    Args:
        db_path (str): path to database with compartment and cell information
        compartment_name (str): name of compartment being analyzed, should match what's in columns
        plate (str): plate identifier for writing output file
        output_directory (str): where to write output table
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    conn = sqlite3.connect(db_path)

    data = (
        pl
        .read_database(
            query=f"SELECT Replicate, Condition, ORF, Name, Strain_ID, Predicted_Label, Cell_Children_{compartment_name}_Count FROM Per_Cell;",
            connection=conn)
        .filter(pl.col(f"Cell_Children_{compartment_name}_Count") != -1) # Remove cells whose compartments were removed for being problematic
    )

    CI = 0.95
    z = stats.norm.ppf((1 + CI) / 2)

    # All wildtypes combined
    agg_data1 = (
        data
        .filter(pl.col("ORF").is_in(["YOR202W", "YMR271C"]))
        .group_by(["Replicate", "Condition", "Predicted_Label"])
        .agg(
            pl.col(f"Cell_Children_{compartment_name}_Count").mean().alias(f"Mean_Num_{compartment_name}"),
            pl.col(f"Cell_Children_{compartment_name}_Count").median().alias(f"Median_Num_{compartment_name}"),
            pl.col(f"Cell_Children_{compartment_name}_Count").std().alias(f"StDev_Num_{compartment_name}"),
            # Compute CI lower bound
            (
                    pl.col(f"Cell_Children_{compartment_name}_Count").mean()
                    - z * (
                            pl.col(f"Cell_Children_{compartment_name}_Count").std()
                            / pl.col(f"Cell_Children_{compartment_name}_Count").count().sqrt()
                    )
            ).alias("Lower_95CI"),
            # Compute CI upper bound
            (
                    pl.col(f"Cell_Children_{compartment_name}_Count").mean()
                    + z * (
                            pl.col(f"Cell_Children_{compartment_name}_Count").std()
                            / pl.col(f"Cell_Children_{compartment_name}_Count").count().sqrt()
                    )
            ).alias("Upper_95CI"),
        )
        .pivot(
            on="Predicted_Label",
            index=["Replicate", "Condition"],
            values=[f"Mean_Num_{compartment_name}", f"Median_Num_{compartment_name}", f"StDev_Num_{compartment_name}", "Lower_95CI", "Upper_95CI"]
        )
        .with_columns(pl.lit("").alias("ORF"), pl.lit("WT").alias("Name"), pl.lit("ALL_WILDTYPES").alias("Strain_ID"))
        .sort(["Replicate", "Condition"])
    )

    # All mutants combined
    agg_data2 = (
        data
        .filter(~pl.col("ORF").is_in(["YOR202W", "YMR271C"]))
        .group_by(["Replicate", "Condition", "Predicted_Label"])
        .agg(
            pl.col(f"Cell_Children_{compartment_name}_Count").mean().alias(f"Mean_Num_{compartment_name}"),
            pl.col(f"Cell_Children_{compartment_name}_Count").median().alias(f"Median_Num_{compartment_name}"),
            pl.col(f"Cell_Children_{compartment_name}_Count").std().alias(f"StDev_Num_{compartment_name}"),
            # Compute CI lower bound
            (
                    pl.col(f"Cell_Children_{compartment_name}_Count").mean()
                    - z * (
                            pl.col(f"Cell_Children_{compartment_name}_Count").std()
                            / pl.col(f"Cell_Children_{compartment_name}_Count").count().sqrt()
                    )
            ).alias("Lower_95CI"),
            # Compute CI upper bound
            (
                    pl.col(f"Cell_Children_{compartment_name}_Count").mean()
                    + z * (
                            pl.col(f"Cell_Children_{compartment_name}_Count").std()
                            / pl.col(f"Cell_Children_{compartment_name}_Count").count().sqrt()
                    )
            ).alias("Upper_95CI"),
        )
        .pivot(
            on="Predicted_Label",
            index=["Replicate", "Condition"],
            values=[f"Mean_Num_{compartment_name}", f"Median_Num_{compartment_name}", f"StDev_Num_{compartment_name}", "Lower_95CI", "Upper_95CI"]
        )
        .with_columns(pl.lit("").alias("ORF"), pl.lit("MUT").alias("Name"), pl.lit("ALL_MUTANTS").alias("Strain_ID"))
        .sort(["Replicate", "Condition"])
    )

    # Individual strains
    agg_data3 = (
        data
        .group_by(["Replicate", "Condition", "ORF", "Name", "Strain_ID", "Predicted_Label"])
        .agg(
            pl.col(f"Cell_Children_{compartment_name}_Count").mean().alias(f"Mean_Num_{compartment_name}"),
            pl.col(f"Cell_Children_{compartment_name}_Count").median().alias(f"Median_Num_{compartment_name}"),
            pl.col(f"Cell_Children_{compartment_name}_Count").std().alias(f"StDev_Num_{compartment_name}"),
            # Compute CI lower bound
            (
                    pl.col(f"Cell_Children_{compartment_name}_Count").mean()
                    - z * (
                            pl.col(f"Cell_Children_{compartment_name}_Count").std()
                            / pl.col(f"Cell_Children_{compartment_name}_Count").count().sqrt()
                    )
            ).alias("Lower_95CI"),
            # Compute CI upper bound
            (
                    pl.col(f"Cell_Children_{compartment_name}_Count").mean()
                    + z * (
                            pl.col(f"Cell_Children_{compartment_name}_Count").std()
                            / pl.col(f"Cell_Children_{compartment_name}_Count").count().sqrt()
                    )
            ).alias("Upper_95CI"),
        )
        .pivot(
            on="Predicted_Label",
            index=["Replicate", "Condition", "ORF", "Name", "Strain_ID"],
            values=[f"Mean_Num_{compartment_name}", f"Median_Num_{compartment_name}", f"StDev_Num_{compartment_name}", "Lower_95CI", "Upper_95CI"]
        )
        .sort(["Replicate", "Condition", "Name"])
    )

    # Combine all three dataframes and save
    (
        pl
        .concat([agg_data1.select(agg_data3.columns), agg_data2.select(agg_data3.columns), agg_data3], how="vertical")
        .select(["Replicate", "Condition", "ORF", "Name", "Strain_ID",
                 f"Mean_Num_{compartment_name}_G1", f"Mean_Num_{compartment_name}_SG2", f"Mean_Num_{compartment_name}_MAT",
                 f"Median_Num_{compartment_name}_G1", f"Median_Num_{compartment_name}_SG2", f"Median_Num_{compartment_name}_MAT",
                 f"StDev_Num_{compartment_name}_G1", f"StDev_Num_{compartment_name}_SG2", f"StDev_Num_{compartment_name}_MAT",
                 "Lower_95CI_G1", "Lower_95CI_SG2", "Lower_95CI_MAT",
                 "Upper_95CI_G1", "Upper_95CI_SG2", "Upper_95CI_MAT"])
        .write_csv(f"{output_directory}/{plate}_{compartment_name.lower()}_number_in_cells_by_cc_and_rep.csv")
    )


def calculate_compartment_coverage(db_path, compartment_name, plate, output_directory):
    """
    Compartment coverage is the % of cell area that is covered by child compartments. It is the ratio of 
    <compartment_name>_AreaShape_Area to Cell_AreaShape_Area.

    Args:
        db_path (str): path to database with compartment and cell information
        compartment_name (str): name of compartment being analyzed, should match what's in columns
        plate (str): plate identifier for writing output file
        output_directory (str): where to write output table
    
    Returns:
        pl.DataFrame with coverage values for every cell
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    conn = sqlite3.connect(db_path)
    coverage = (
        pl
        .read_database(
            query=f"""
                    WITH total_comp_area AS (
	                    SELECT
	                    	Cell_ID,
	                    	SUM({compartment_name}_AreaShape_Area) AS {compartment_name}_AreaShape_Area
	                    FROM Per_{compartment_name}
	                    GROUP BY Cell_ID
	                    )

                    SELECT
                    	Replicate,
                    	Condition,
                    	Row,
                    	Column,
                    	ORF,
                    	Name,
                    	Strain_ID,
                    	Predicted_Label,
                    	Per_Cell.Cell_ID,
                    	(CAST({compartment_name}_AreaShape_Area AS NUMERIC) / Cell_AreaShape_Area) AS Coverage
                    FROM Per_Cell
                    JOIN total_comp_area
                    ON total_comp_area.Cell_ID = Per_Cell.Cell_ID;
                   """,
            connection=conn
        )
    )
    conn.close()

    coverage.write_csv(f"{output_directory}/{plate}_{compartment_name}_coverage.csv")

    return coverage


def calculate_compartment_distances(db_path, compartment_name, plate, output_directory):
    """
    Calculates the distance between the centers of all subcellular compartment masks within a cell, and then determines
    the mean, median, standard deviation, and 95th percentile confidence intervals of distances for every cell.

    Args:
        db_path (str): path to database with compartment and cell information
        compartment_name (str): name of compartment being analyzed, should match what's in columns
        plate (str): plate identifier for writing output file
        output_directory (str): where to write output table

    Returns:
        pl.DataFrame with subcellular compartment distance metrics
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    conn = sqlite3.connect(db_path)
    CI = 0.95
    z = stats.norm.ppf((1 + CI) / 2)

    # Part 1: get table with all comps
    all_comp_distances = (
        pl
        # First, read the dataframe. Compartments are cross-joined to one another and filtered to only
        # include those within the same cell. Self-pairs are also removed.
        .read_database(query=f"""
                        WITH 
                            table1 AS (
                                SELECT 
    	                            Replicate,
    	                            Condition,
    	                            Row,
    	                            Column,
    	                            ORF,
    	                            Name,
    	                            Strain_ID,
    	                            Num_{compartment_name},
    	                            Predicted_Label,
    	                            Per_{compartment_name}.Cell_ID AS Cell_ID1,  
    	                            ROW_NUMBER() OVER (PARTITION BY Per_{compartment_name}.Cell_ID) AS {compartment_name}_Num1,
    	                            {compartment_name}_AreaShape_Center_X AS X1, 
    	                            {compartment_name}_AreaShape_Center_Y AS Y1 
                                FROM Per_{compartment_name}
                                JOIN (SELECT Cell_ID, Cell_Children_{compartment_name}_Count AS Num_{compartment_name}, Predicted_Label FROM Per_Cell) pc
    	                            ON pc.Cell_ID = Per_{compartment_name}.Cell_ID),
    
                            table2 AS (
                                SELECT 
                            	    Cell_ID AS Cell_ID2, 
                            	    ROW_NUMBER() OVER (PARTITION BY Per_{compartment_name}.Cell_ID) AS {compartment_name}_Num2,
                            	    {compartment_name}_AreaShape_Center_X AS X2, 
                            	    {compartment_name}_AreaShape_Center_Y AS Y2 
                                FROM Per_{compartment_name})
    
                        SELECT * 
                        FROM table1
                        CROSS JOIN table2
                        WHERE Cell_ID1 = Cell_ID2 AND {compartment_name}_Num1 < {compartment_name}_Num2;""",
                       connection=conn)
        # Calculate distance between subcellular compartments within the same cell.
        .with_columns(
                (
                    (
                        (pl.col("X2") - pl.col("X1")) ** 2 +
                        (pl.col("Y2") - pl.col("Y1")) ** 2
                    ) ** 0.5
                )
                .alias(f"{compartment_name}_Distance")
            )
        ## Drop extra Cell_ID column
        .rename({"Cell_ID1": "Cell_ID"})
        .drop(["Cell_ID2"])
        # Get aggregate metrics
        .group_by(["Replicate", "Condition", "Row", "Column", "ORF", "Name", "Strain_ID", f"Num_{compartment_name}", "Predicted_Label", "Cell_ID"])
        .agg(
            pl.count().alias(f"Num_Distances"),
            pl.col(f"{compartment_name}_Distance").mean().alias("Distance_Mean"),
            pl.col(f"{compartment_name}_Distance").median().alias("Distance_Median"),
            pl.col(f"{compartment_name}_Distance").std().alias("Distance_StDev"),
            # Compute CI lower bound
            (
                    pl.col(f"{compartment_name}_Distance").mean()
                    - z * (
                            pl.col(f"{compartment_name}_Distance").std()
                            / pl.col(f"{compartment_name}_Distance").count().sqrt()
                    )
            ).alias("Lower_95CI"),
            # Compute CI upper bound
            (
                    pl.col(f"{compartment_name}_Distance").mean()
                    + z * (
                            pl.col(f"{compartment_name}_Distance").std()
                            / pl.col(f"{compartment_name}_Distance").count().sqrt()
                    )
            ).alias("Upper_95CI"),
        )
    )

    all_comp_distances.write_csv(f"{output_directory}/{plate}_{compartment_name}_distance_metrics.csv")

    return all_comp_distances


def combine_FracAtD_rings(db_path, compartment_name, plate, output_directory):
    """
    For FracAtD features, every cell has been divided into 4 rings going from its center to periphery.
    To simplify analysis, the two inner rings and two outer rings are combined.

    Args:
        db_path (str): path to database with compartment and cell information
        compartment_name (str): name of compartment being analyzed, should match what's in columns
        plate (str): plate identifier for writing output file
        output_directory (str): where to write output table

    Returns:
        pl.DataFrame with simplified FracAtD values.
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    conn = sqlite3.connect(db_path)
    simplified_FracAtD = (
        pl
        .read_database(
            query="""
                    SELECT 
    	                Replicate,
    	                Condition,
    	                Row,
    	                Column,
    	                ORF,
    	                Name,
    	                Strain_ID,
    	                Predicted_Label,
    	                Cell_ID,
    	                Cell_RadialDistribution_FracAtD_GFP_1of4 AS FracD_1of4,
    	                Cell_RadialDistribution_FracAtD_GFP_2of4 AS FracD_2of4,
    	                Cell_RadialDistribution_FracAtD_GFP_3of4 AS FracD_3of4,
    	                Cell_RadialDistribution_FracAtD_GFP_4of4 AS FracD_4of4
    	            FROM Per_Cell;
    	          """,
            connection=conn
        )
        .with_columns(
            (pl.col("FracD_1of4") + pl.col("FracD_2of4")).alias("Inner_Distribution"),
            (pl.col("FracD_3of4") + pl.col("FracD_4of4")).alias("Outer_Distribution"),
        )
        .drop(["FracD_1of4", "FracD_2of4", "FracD_3of4", "FracD_4of4"])
    )

    conn.close()

    simplified_FracAtD.write_csv(f"{output_directory}/{plate}_{compartment_name}_simplified_FracAtD_features.csv")

    return simplified_FracAtD
