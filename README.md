# Marker Project Analysis

### Project Setup
Clone the repository
```
$ git clone https://github.com/arsam-dj/MarkerProject.git
$ cd MarkerProject
```

Create conda environment
```
$ conda env create -f environment.yml
```

---

### Overview
This repository is a collection of scripts and notebooks used to analyze each Marker Project screen in a step-wise manner. Scripts that don't have a screen name (e.g., ```01_add_cell-IDs_and_strain_information.py```) are general scripts that apply to all screens. Those with a screen name (e.g., ```01-1_additional_tgl3_processing.py```) are screen-specific and address particular exceptions specfic to that screen.

Raw images were processed and undergone feature extraction using CellPose and CellProfiler. Extracted features are stored in a series of databases for every screen. Every screen has 20-22 plates × 3 replicates and these scripts assume that every database contains objects for each plate across its three replicates (e.g., ```Nop10_DMA_Plate01.db``` contains all Nop10 objects from R1_Plate01, R2_Plate01, and R3_Plate01).

Screens should be run in order as indicated by each script name. Some generalist scripts, such as those deleting a list of specified objects, are run multiple times in the pipeline. For example, the order of scripts run for the Nucleolus screen would be:

**Initial Processing**

1. ```00_fix_overlay_paths_and_metadata_columns.py```
2. ```01_add_cell-IDs_and_strain_information_nop10.py```
3. ```02_deleting_parentless_compartments.py```


**Getting Coordinates for Every Segmented Object**

4. ```03-1_generate_file_with_cell_coords_and_image_paths_for_opera.py```
5. ```GEN_combine_files_from_all_plates.py```


**Doing Quality Check on Every Cell and Nucleus**

6. ```04_generate_cell_and_nucleus_quality_check_histograms.py```
7. ```04-0_generate_singlecelltool_inputs_for_checking_cells.ipynb```
8. ```04-1_getting_low_quality_objects.py```
9. ```GEN_combine_files_from_all_plates.py```
10. ```GEN_deleting_low_quality_objects.py```

**Doing Quality Check on Every Subcellular Compartment of Interest**

11. ```05-4_generate_nucleolus_quality_check_histograms.py```
12. ```05-4-1_removing_low_quality_nucleolus_objects.py```
13. ```GEN_combine_files_from_all_plates.py```


**Cell Cycle Classification**

14. ```06-1_get_features_for_cell_cycle_classification.py```
15. ```06-2_make_cell_cycle_classification_model_on_training_data.py```
16. ```06-3_do_cell_cycle_classification_on_all_cells.py```
17. ```GEN_combine_files_from_all_plates.py```
18. ```06-4_get_random_cells_for_viewing.py```


**Doing Quality Check on Cell Cycle**

19. ```07-0_get_sct_inputs_for_low_confidence_cell_cycle_classification.ipynb```
20. ```07_getting_low_confidence_cell_cycle_classifications.py```
21. ```GEN_combine_files_from_all_plates.py```
22. ```GEN_deleting_low_quality_objects.py```


**Whole-Cell Phenotyping**

23. ```08_whole_cell_phenotypes.py```
24. ```GEN_phenotype_outputs_nop10_additional_processing.py```
25. ```GEN_combine_files_from_all_plates_for_phenotypes_directory.py```


**Subcellular Compartment Phenotyping**

26. ```09-4_nucleolus_phenotypes.py```
27. ```GEN_phenotype_outputs_nop10_additional_processing.py```
28. ```GEN_combine_files_from_all_plates_for_phenotypes_directory.py```


**Combining Whole-Cell and Subcellular Penetrances**

29. ```10_combine_cell_phenotype_overall_penetrances.py```


**Obtaining Final Strain Hits (Whole-Cell)**

30. ```11-1_get_replicate_distances_nop10.py```
31. ```11-2_final_strain_filtering_nop10.py```


**Obtaining Final Strain Hits (Subcellular Compartment)**

32. ```11-1_get_replicate_distances_nop10.py```
33. ```11-2_final_strain_filtering_nop10.py```


Some scripts are run on each plate individually (e.g., ```01_add_cell-IDs_and_strain_information_nop10.py```) while others are run on a whole directory of outputs (e.g., ```GEN_combine_files_from_all_plates.py```). Text files in the directory ```submitted_jobs``` list all the jobs that were run for every screen and the exact parameters.

More information about each script is provided below.


### Part 0. Activate Conda Environment

```
$ conda activate cellprofiler
```


### Part 1. Initial Processing

**```00_fix_overlay_paths_and_metadata_columns.py```**

When files and location features are exported from CellProfiler, they are formatted like b'TS3'_cb'37'rb'03'cb'06'fb'01'p01_overlays.png or b'03' respectively. Given the output file directory and/or database, this script fixes the formatting by removing the b'...' strings. For example, the file name becomes TS3_c37r03c06f01p01_overlays.png.

```
$ python 00_fix_overlay_paths_and_metadata_columns.py -d <database_path> -o <exported_files_path>

$ python /home/alex/alex_files/markerproject_redux/scripts/00_fix_overlay_paths_and_metadata_columns.py -d /home/alex/alex_files/markerproject_redux/screens/Nop10/Nop10_DMA_Plate01.db -o /home/morphology/mpg53/alex/MP_Redux_Features/Nop10/DMA/Plate01/overlays
```

**```01_add_cell-IDs_and_strain_information_nop10.py```**

This script adds unique Cell IDs to every segmented Cell object, maps the Cell IDs to the Cell's children, and adds ORF, Name, and Strain ID from the provided array mapping files. Unique Cell IDs (e.g., 4R130010010010013) are descriptive and can be broken down as: ```4 | R1 | 30 | 01 | 001 | 001| 001 | 3``` which translates to ```<screen_num> | <replicate> | <screening_temp> | <plate_num> | <row> | <column> | <field> | <object_num_on_image>```.

```
$ python 01_add_cell-IDs_and_strain_information.py -d <database_path> -i <screen_num> -t <subcellular_comp_table_name> -c <subcellular_comp_parent_cell_column> -x <array_mapping_sheet> -r <is_tsa_plate> -z <num_digits_in_row_col>

$ python /home/alex/alex_files/markerproject_redux/scripts/01_add_cell-IDs_and_strain_information.py -d /home/alex/alex_files/markerproject_redux/screens/Nop10/Nop10_DMA_Plate01.db -i 4 -t Per_Nucleolus -c Nucleolus_Parent_Cell -x /home/alex/alex_files/markerproject_redux/array_mapping_files/SGA-Array-ver2-1536.csv -r False -z 3
```

**```02_deleting_parentless_compartments.py```**

Lots of segmented subcellular objects have no assigned parent (e.g., they only represent noise) and are thus removed from the database by this script. Cells with no nucleus are also removed. Finally, the columns describing the number of Cells, Nuclei, and Subcellular Compartments in each image are updated to reflect the removed objects.

```
$ python 02_deleting_parentless_compartments.py -d <database_path> -t <subcellular_comp_table_name> -c <num_subcellular_compartment_in_image>

$ python /home/alex/alex_files/markerproject_redux/scripts/02_deleting_parentless_compartments.py -d /home/alex/alex_files/markerproject_redux/screens/Nop10/Nop10_DMA_Plate01.db -t Per_Nucleolus -c Image_Count_Nucleolus
```


### Part 2. Getting Coordinates for Every Segmented Object

**```03-1_generate_file_with_cell_coords_and_image_paths_for_opera.py```**

Cell coordinates are necessary for viewing cells as done farther down this pipeline. This script extracts each Cell object's Cell ID, Image Path, X-coordinate, and Y-coordinate and exports them to two csv files; one for overlay images, one for raw images.

```

$ python 03-1_generate_file_with_cell_coords_and_image_paths_for_opera.py -s <screen_marker> -d <database_path> -v <overlay_images_directory> -r <raw_images_directory> -t <is_tsa> -p <plate_num> -o <output_directory>

$ python /home/alex/alex_files/markerproject_redux/scripts/03-1_generate_file_with_cell_coords_and_image_paths_for_opera.py -s Nop10 -d /home/alex/alex_files/markerproject_redux/screens/Nop10/Nop10_DMA_Plate01.db -v //192.168.0.48/mpg53/alex/MP_Redux_Features/Nop10/DMA/Plate01/overlays -r //192.168.0.48/Morphology/Marker_Project_Screens/Screens_Nop10 -t False -p 01 -o /home/alex/alex_files/markerproject_redux/coordinates/Nop10
```

**```GEN_combine_files_from_all_plates.py```**

Once the above script has been run on all plates, the resulting csv files are combined to make filtering cells from multiple plates easier.

```
$ python GEN_combine_files_from_all_plates.py -d <directory_path> -s <file_substring> -o <output_file_name>

# overlay images
$ python /home/alex/alex_files/markerproject_redux/scripts/GEN_combine_files_from_all_plates.py -d /home/alex/alex_files/markerproject_redux/coordinates/Nop10 -s overlay_image_paths -o all_overlay_paths

# raw images
$ python /home/alex/alex_files/markerproject_redux/scripts/GEN_combine_files_from_all_plates.py -d /home/alex/alex_files/markerproject_redux/coordinates/Nop10 -s raw_image_paths -o all_raw_paths
```


### Part 3. Doing Quality Check on Every Cell and Nucleus

**```04_generate_cell_and_nucleus_quality_check_histograms.py```**

During quality check, many abnormal looking objects are removed (e.g., abnormally large/large, abnormally elongated, abormal nucleus:cell ratio, etc...) This is done by generating histograms for a list of QC features, viewing a sample of cells at a series of different thresholds, and removing cells above/below a cutoff threshold where there are more odd segmentation masks than normal ones. Cells from all plates are combined and features are converted to Z-Scores (in most cases), and then plotted on histograms with y-axes on log scale. It also produces several csv files with all cells, their Cell IDs, and raw/scaled QC features.

**NOTE:** depending on screen size, this may be extremely memory-intensive. Running on a cluster for very large screens is recommended.

```
$ python 04_generate_cell_and_nucleus_quality_check_histograms.py -q <quality_check_directory> -d <database_directory> -x <overlay_path_csv>

$ python /home/alex/alex_files/markerproject_redux/scripts/04_generate_cell_and_nucleus_quality_check_histograms.py -q /home/alex/alex_files/markerproject_redux/quality_check/Nop10/cell_and_nuclei -d /home/alex/alex_files/markerproject_redux/screens/Nop10 -x /home/alex/alex_files/markerproject_redux/coordinates/Nop10/all_overlay_paths.csv
```

**```04-0_generate_singlecelltool_inputs_for_checking_cells.ipynb```**

This Jupyter notebook provides an easy way of checking segmented objects. Provide the sct_input_maker function with the csv feature table, the feature of interest, output file name, lower threshold, and upper threshold to produce a file valid for the SingleCellTool (https://github.com/BooneAndrewsLab/singlecelltool/tree/master).

**```04-1_getting_low_quality_objects.py```**

Once cut-off thresholds for each feature have been determined using manual assessments, this script is edited with these cutoff thresholds. Cell IDs for objects above or below these thresholds are exported. This script also exports per-strain data (i.e., how many cells are being removed from each strain). **This script MUST be edited for each screen, as cut-off thresholds differ from screen to screen.**

For example, ```(pl.col('Cell_AreaShape_Area') <= -1.75) | (pl.col('Cell_AreaShape_Area') >= 4.75)``` identifies objects whose Area Z-Score is abnormally small (<= -1.75) or abnormally large (>= 4.75).

QC thresholds for every screen are found in additional_files/quality_check_filters.xlsx
```
$ python 04-1_getting_low_quality_objects.py -d <database_path> -q <output_directory> -c <scaled_cell_qc_features> -n <raw_nucleus_qc_features> -a <raw_cell_qc_features> -p <plate_label>

$ python /home/alex/alex_files/markerproject_redux/scripts/04-1_getting_low_quality_objects.py -d /home/alex/alex_files/markerproject_redux/screens/Nop10/Nop10_DMA_Plate01.db -q /home/alex/alex_files/markerproject_redux/quality_check/Nop10/cell_and_nuclei/filtered_cells -c /home/alex/alex_files/markerproject_redux/quality_check/Nop10/cell_and_nuclei/scaled_cell_qc_features.csv -n /home/alex/alex_files/markerproject_redux/quality_check/Nop10/cell_and_nuclei/raw_nucleus_qc_features.csv -a /home/alex/alex_files/markerproject_redux/quality_check/Nop10/cell_and_nuclei/raw_cell_qc_features.csv -p DMA_Plate01
```

**```GEN_combine_files_from_all_plates.py```**

Files with Cell IDs to be removed and per-strain information are combined here.

```
$ python GEN_combine_files_from_all_plates.py -d <directory_path> -s <file_substring> -o <output_file_name>

# cell ids
$ python /home/alex/alex_files/markerproject_redux/scripts/GEN_combine_files_from_all_plates.py -d /home/alex/alex_files/markerproject_redux/quality_check/Nop10/cell_and_nuclei/filtered_cells -s filtered_cells -o all_filtered_cells

# strain information
$ python /home/alex/alex_files/markerproject_redux/scripts/GEN_combine_files_from_all_plates.py -d /home/alex/alex_files/markerproject_redux/quality_check/Nop10/cell_and_nuclei/filtered_cells -s strain_stats_after_filtering -o all_strain_stats_after_filtering
```

**```GEN_deleting_low_quality_objects.py```**

This script modifies the databases and removes objects with matching Cell IDs. It also removes all of their children and updates the number of Cell/Nuclei/Subcellular Compartment masks in each image accordingly.

```
$ python GEN_deleting_low_quality_objects.py -d <database_path> -t <subcellular_comp_table_name> -c <num_subcellular_compartment_in_image> -f <cell_ids_to_remove>

$ python /home/alex/alex_files/markerproject_redux/scripts/GEN_deleting_low_quality_objects.py -d /home/alex/alex_files/markerproject_redux/screens/Nop10/Nop10_DMA_Plate01.db -t Per_Nucleolus -c Image_Count_Nucleolus -f /home/alex/alex_files/markerproject_redux/quality_check/Nop10/cell_and_nuclei/filtered_cells/DMA_Plate01_filtered_cells.csv
```


### Part 4. Doing Quality Check on Every Subcellular Compartment of Interest

**```05-4_generate_nucleolus_quality_check_histograms.py```**

Segmentation masks for some subcellular compartments can poor or misleading. This script creates histograms or exports csv files of object Cell IDs whose children should be removed (either some children can be removed or all can be removed).

**NOTE:** depending on screen size, this may be extremely memory-intensive. Running on a cluster for very large screens is recommended.

```
$ python 05-4_generate_nucleolus_quality_check_histograms.py -q <quality_check_directory> -d <database_directory> -x <overlay_path_csv>

$ python /home/alex/alex_files/markerproject_redux/scripts/05-4_generate_nucleolus_quality_check_histograms.py -q /home/alex/alex_files/markerproject_redux/quality_check/Nop10/nucleolus -d /home/alex/alex_files/markerproject_redux/screens/Nop10 -x /home/alex/alex_files/markerproject_redux/coordinates/Nop10/all_overlay_paths.csv
```

```05-4-1_removing_low_quality_nucleolus_objects.py```

This script deletes some or all children of specified Cell objects. It updates the number of children for every Cell accordingly and exports per-strain information (i.e., how many children were removed on average for every strain).

```
$ python 05-4-1_removing_low_quality_nucleolus_objects.py -d <database_path> -q <output_directory> -c <feature_table> -p <plate_label> -x <delete_all_children>

python /home/alex/alex_files/markerproject_redux/scripts/05-4-1_removing_low_quality_nucleolus_objects.py -d /home/alex/alex_files/markerproject_redux/screens/Nop10/Nop10_DMA_Plate01.db -q /home/alex/alex_files/markerproject_redux/quality_check/Nop10/nucleolus/filtered_nucleoli -c /home/alex/alex_files/markerproject_redux/quality_check/Nop10/nucleolus/raw_nucleolus_qc_features.csv -p DMA_Plate01 -x True
```

```GEN_combine_files_from_all_plates.py```

Files with removed children and per-strain information are combined.

```
$ python GEN_combine_files_from_all_plates.py -d <directory_path> -s <file_substring> -o <output_file_name>

# removed children
$ python /home/alex/alex_files/markerproject_redux/scripts/GEN_combine_files_from_all_plates.py -d /home/alex/alex_files/markerproject_redux/quality_check/Nop10/nucleolus/filtered_nucleoli -s filtered_Nucleolus -o all_filtered_nucleoli

# per-strain information
$ python /home/alex/alex_files/markerproject_redux/scripts/GEN_combine_files_from_all_plates.py -d /home/alex/alex_files/markerproject_redux/quality_check/Nop10/nucleolus/filtered_nucleoli -s strain_stats -o all_strain_stats
```


### License
This software is licensed under the [BSD 3-Clause License][BSD3]. Please see the 
``LICENSE`` file for more details.

[intracellular_organization_dynamics_test_images]: https://thecellvision.org/intracellular_organization_dynamics/Intracellular_Organization_Dynamics_Demo_Images.tar.gz
[BSD3]: https://opensource.org/license/bsd-3-clause