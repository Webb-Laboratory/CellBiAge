library(enrichR)
library(viridis)

dbs <- c('GO_Biological_Process_2023','GO_Cellular_Component_2023',
         'GO_Molecular_Function_2023', 'KEGG_2019_Mouse')

wrapText <- function(x, len) {
  sapply(x, function(y) paste(strwrap(y, len), collapse = "\n"), USE.NAMES = FALSE)
}

celltype <- "all"

ranked <- read.csv("~/Downloads/Table1_ranked_genes.csv")

# run enrichR on different gene sets:
dbs <- c('Aging_Perturbations_from_GEO_down', 'Aging_Perturbations_from_GEO_up')

# cur_result <- enrichr(ranked$gene[1:20], dbs)
cur_result <- enrichr(ranked$gene[1:20], dbs)
cur_result <- enrichr(ranked$gene[1:200], dbs)
collapsed_output <- data.frame()

# collapse results into one dataframe
for(db in dbs){
  print(db)
  cur_result[[db]]$cluster <- celltype
  cur_result[[db]]$db <- db
  collapsed_output <- rbind(collapsed_output, cur_result[[db]])
}

collapsed_output <- collapsed_output %>% subset(Adjusted.P.value<0.05)

ranked_df <- ranked[, c('absolute_coef', 'gene')]
overlapped_genes <- ranked_df$gene[ranked_df$gene%in% ref_data_short$Gene.Symbol]
ref_data_short_ovelapped <- ranked_df[ranked_df$gene %in% overlapped_genes,]
ref_data_short_ovelapped$db <- "GeneAge"

overlapped_genes <- ranked_df$gene[ranked_df$gene%in% ref_atlas$Gene_short]
ref_atlas_ovelapped <- ranked_df[ranked_df$gene %in% overlapped_genes,]
ref_atlas_ovelapped$db <- "Digial Ageing Atlas"

overlapped_genes <- ranked_df$gene[ranked_df$gene%in% ref_scatlas$Symbol]
ref_scatlas_ovelapped <- ranked_df[ranked_df$gene %in% overlapped_genes,]
ref_scatlas_ovelapped$db <- "Aging Atlas"

overlapped <- rbind(ref_data_short_ovelapped, ref_atlas_ovelapped, ref_scatlas_ovelapped)
write.csv(overlapped, "~/Downloads/age_related_genes.csv")


