# FUNCTIONAL ENCRICHMENT SCRIPT
# APRIL 28, 2026
# ALEX DAIEJAVAD

# What does this script do?
## Do functional enrichment for cc and bp using GO Slims and gProfiler2 for
## ordered gene lists per marker-cluster

# -----------------------------------------------------------------------------
library(openxlsx)
library(dplyr)
library(gprofiler2)
library(ggplot2)
library(stringr)
library(naturalsort)
#------------------------------------------------------------------------------

# Define directories and data splits
# Windows
INPUT_DIRECTORY = "//wsl.localhost/Ubuntu/home/alex/alex_files/markerproject_redux/strain_filtering/Compartments/filtered_strain_workbooks"
OUTPUT_DIRECTORY = "//wsl.localhost/Ubuntu/home/alex/alex_files/markerproject_redux/functional_enrichment/Compartments"
dir.create(OUTPUT_DIRECTORY, recursive = TRUE)

# Function for doing functional enrichment 
gost_enrich = function(go_standard, marker) {
  bg_genes = openxlsx::read.xlsx(sprintf("%s/%s_filtered_strains.xlsx", INPUT_DIRECTORY, marker), sheet = "SheetB")
  bg_genes = unique(bg_genes[ , 4]) # turn it into a vector
  
  marker_data = openxlsx::read.xlsx(sprintf("%s/%s_filtered_strains.xlsx", INPUT_DIRECTORY, marker), sheet = "SheetD")
  marker_data = marker_data[order(marker_data$Overall_Penetrance, decreasing = TRUE), ]
  
  enrichment = gprofiler2::gost(query = marker_data$ORF,
                                organism = go_standard,
                                ordered_query = TRUE,
                                significant = TRUE,
                                user_threshold = 0.01,
                                correction_method = "fdr",
                                sources = c(),
                                custom_bg = bg_genes)$result
  
  write.xlsx(enrichment, file = sprintf("%s/%s_bioprocess_enrichments.xlsx", OUTPUT_DIRECTORY, marker))

}

# Upload GMT file containing GO Slim mapping
#bp_gmt = gprofiler2::upload_GMT_file("//wsl.localhost/Ubuntu/home/alex/alex_files/markerproject_redux/misc_files/go_slim_p_json.gmt") # gp__QhP6_SV0W_5Zg
bp_gmt = "gp__QhP6_SV0W_5Zg"

## Single-marker BP enrichment
#markers = c('Cdc11', 'Dad2', 'Heh2', 'Hta2', 'Nop10', 'Nuf2', 'Om45', 'Pil1', 'Psr1', 
#      'Sac6', 'Sec21', 'Sec7', 'Snf7', 'Spf1', 'Vph1')
markers = c("Dad2", "Nuf2", "Pex3", "Pil1", "Psr1", "Sac6", "Sec21", "Sec7", "Snf7", "Spf1", "Tgl3")

for (marker in markers) {
  gost_enrich(bp_gmt, marker)
}


## Combined BP enrichment
