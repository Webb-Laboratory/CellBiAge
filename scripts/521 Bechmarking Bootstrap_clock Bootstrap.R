# REQUIRES COMPUTE CLUSTER TO RUN

library(tidyr)
library(Seurat)
library(Matrix)

#==================================================================================================
print("Load Exercise 10x Data (v3 and v2 Combined")
pb <- readRDS("./data/hypo.integrated.final.20210719.RDS") # 8 samples, 40064 cells
pb.umi <- t(pb[["RNA"]]@counts)
pb.genes <- colnames(pb.umi)

#==================================================================================================
print("Load Clock Training Object to match column structure")
svz <- readRDS("data/bootstrap_pseudocell_15_seed42.rds")
genes <- colnames(svz[1,5][[1]][[1]]) # use these genes to subset exercise data. 20948 genes.

#==================================================================================================
print("Format exercise data")
pb.missing <- setdiff(genes, pb.genes)
if (length(as.numeric(pb.missing)) > 0) {
  print("There are some genes in the training data
        that are not in test data. Filling with Zeros.")
  missing_df <- matrix(0, nrow(pb.umi), length(pb.missing))
  colnames(missing_df) <- pb.missing
  rownames(missing_df) <- rownames(pb.umi)
  pb.umi <- cbind(pb.umi, missing_df)
}
pb.umi <- pb.umi[, genes] # Reorder and cut to size to match clock training data format
pb.umi_vector <- as.data.frame(pb.umi)

# Fix metadata in seurat object



# Change Cell type labels
df <- tibble("Celltype" = as.character(pb@meta.data$group),
             "Sample" = as.character(pb@meta.data$orig.ident),
             "Mouse" = as.character(pb@meta.data$orig.ident),
             "Year" = as.character(pb@meta.data$stim),
             pb.umi_vector)


# Clean memory
rm(pb); rm(pb.umi); rm(svz)


# Convert to nested tidy form & nest
Celltypes <- c("Neuron", "Oligodendrocyte", "Astrocyte",
               "Microglia", "OPC")
by_celltype <- df %>%
    group_by(Celltype, Mouse, Sample, Year) %>%
    nest() %>%
    dplyr::filter(Celltype %in% Celltypes)
colnames(by_celltype)[5] <- "exData"

#==================================================================================================
print("Bootstrap and pseudobulk")

bootstrap.pseudocells <- function(df, size=15, n=100, replace="dynamic") {
    pseudocells <- c()
    # If dynamic then only sample with replacement if required due to shortage of cells
    if (replace == "dynamic") {
        if (nrow(df) <= size) {replace <- TRUE} else {replace <- FALSE}
    }
    for (i in c(1:n)) {
        batch <- df[sample(1:nrow(df), size = size, replace = replace), ]
        pseudocells <- rbind(pseudocells, colSums(batch))
    }
    colnames(pseudocells) <- colnames(df)
    return(as_tibble(pseudocells))
}

# Apply boostrap.pseudocells using map()

by_celltype <- by_celltype %>%
  mutate(exData = case_when(
    exData == "Aged" ~ "24",
    exData == "Young" ~ "3",
    TRUE ~ as.character(exData)
  ))

df2 <- by_celltype %>% mutate(pseudocell_ex = map(exData, bootstrap.pseudocells))
                      
print(dim(df2))            
# Remove single cell data; keep just key metadata and pseudocells
df2$exData <- NULL

df2 <- df2 %>% unnest(pseudocell_ex)
print(dim(df2))
# [1]  4000 20952
saveRDS(df2, "data/bootstrap_pseudocell_15_hypo.rds")
print("Done")
