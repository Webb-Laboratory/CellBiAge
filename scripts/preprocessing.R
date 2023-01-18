# MLAging project DY@WebbLabBrown
# data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE188646
# from paper: https://doi.org/10.1038/s43587-022-00246-4

library(tidyverse)
library(Seurat)

##--1.read data from 10x and perform preprocessing----
Aged_1 <- Read10X(data.dir = '../filtered_feature_bc_matrix/Aged_1/filtered_feature_bc_matrix') 
Aged_2 <- Read10X(data.dir = '../filtered_feature_bc_matrix/Aged_2/filtered_feature_bc_matrix')
Aged_3 <- Read10X(data.dir = '../filtered_feature_bc_matrix/Aged_3/filtered_feature_bc_matrix')
Aged_4 <- Read10X(data.dir = '../filtered_feature_bc_matrix/Aged_4/filtered_feature_bc_matrix')

Young_1 <- Read10X(data.dir = '../filtered_feature_bc_matrix/Young_1/filtered_feature_bc_matrix')
Young_2 <- Read10X(data.dir = '../filtered_feature_bc_matrix/Young_2/filtered_feature_bc_matrix') 
Young_3 <- Read10X(data.dir = '../filtered_feature_bc_matrix/Young_3/filtered_feature_bc_matrix')
Young_4 <- Read10X(data.dir = '../filtered_feature_bc_matrix/Young_4/filtered_feature_bc_matrix')


Aged_1 <- CreateSeuratObject(counts = Aged_1, project = 'Aged_1', min.cells = 3, min.features = 200)
Aged_2 <- CreateSeuratObject(counts = Aged_2, project = 'Aged_2', min.cells = 3, min.features = 200)
Aged_3 <- CreateSeuratObject(counts = Aged_3, project = 'Aged_3', min.cells = 3, min.features = 200)
Aged_4 <- CreateSeuratObject(counts = Aged_4, project = 'Aged_4', min.cells = 3, min.features = 200)

Young_1 <- CreateSeuratObject(counts = Young_1, project = 'Young_1', min.cells = 3, min.features = 200)
Young_2 <- CreateSeuratObject(counts = Young_2, project = 'Young_2', min.cells = 3, min.features = 200)
Young_3 <- CreateSeuratObject(counts = Young_3, project = 'Young_3', min.cells = 3, min.features = 200)
Young_4 <- CreateSeuratObject(counts = Young_4, project = 'Young_4', min.cells = 3, min.features = 200)


Aged_1$stim <- 'Aged'  
Aged_2$stim <- 'Aged'
Aged_3$stim <- 'Aged'   
Aged_4$stim <- 'Aged'

Young_1$stim <- 'Young'
Young_2$stim <- 'Young'
Young_3$stim <- 'Young'
Young_4$stim <- 'Young'

# sample generated from batch1: test set
# sample generated from batch2: training set
Aged_1$batch <- '1'  
Aged_2$batch <- '1'
Young_1$batch <- '1'
Young_2$batch <- '1'  

Aged_3$batch <- '2'   
Aged_4$batch <- '2'
Young_3$batch <- '2'
Young_4$batch <- '2'


Aged_1[['percent.mt']] <- PercentageFeatureSet(Aged_1, pattern = '^mt-')
Aged_2[['percent.mt']] <- PercentageFeatureSet(Aged_2, pattern = '^mt-')
Aged_3[['percent.mt']] <- PercentageFeatureSet(Aged_3, pattern = '^mt-')
Aged_4[['percent.mt']] <- PercentageFeatureSet(Aged_4, pattern = '^mt-')

Young_1[['percent.mt']] <- PercentageFeatureSet(Young_1, pattern = '^mt-')
Young_2[['percent.mt']] <- PercentageFeatureSet(Young_2, pattern = '^mt-')
Young_3[['percent.mt']] <- PercentageFeatureSet(Young_3, pattern = '^mt-')
Young_4[['percent.mt']] <- PercentageFeatureSet(Young_4, pattern = '^mt-')

# Used different cutoffs for samples generated/sequenced with different methods
# the cutoffs were consistent with the published paper (https://doi.org/10.1038/s43587-022-00246-4)
Aged_1 <- subset(Aged_1, subset = nFeature_RNA > 200 & nFeature_RNA < 3000  & percent.mt < 10)
Aged_2 <- subset(Aged_2, subset = nFeature_RNA > 200 & nFeature_RNA < 3000  & percent.mt < 10)

Aged_3 <- subset(Aged_3, subset = nFeature_RNA > 200 & nFeature_RNA < 7500  & percent.mt < 10)
Aged_4 <- subset(Aged_4, subset = nFeature_RNA > 200 & nFeature_RNA < 7500  & percent.mt < 10)

Young_1 <- subset(Young_1, subset = nFeature_RNA > 200 & nFeature_RNA < 3000  & percent.mt < 10)
Young_2 <- subset(Young_2, subset = nFeature_RNA > 200 & nFeature_RNA < 3000  & percent.mt < 10)

Young_3 <- subset(Young_3, subset = nFeature_RNA > 200 & nFeature_RNA < 7500  & percent.mt < 10)
Young_4 <- subset(Young_4, subset = nFeature_RNA > 200 & nFeature_RNA < 7500  & percent.mt < 10)

# count normalization
hypo.list1 <- c(Young_1, Young_2, Aged_1, Aged_2)
for (i in 1:length(hypo.list1)) {
  hypo.list1[[i]] <- NormalizeData(hypo.list1[[i]], verbose = FALSE)
  hypo.list1[[i]] <- FindVariableFeatures(hypo.list1[[i]], selection.method = 'vst', nfeatures = 5000, verbose = FALSE)
}

hypo.list2 <- c(Young_3, Young_4, Aged_3, Aged_4)
for (i in 1:length(hypo.list2)) {
  hypo.list2[[i]] <- NormalizeData(hypo.list2[[i]], verbose = FALSE)
  hypo.list2[[i]] <- FindVariableFeatures(hypo.list2[[i]], selection.method = 'vst', nfeatures = 5000, verbose = FALSE)
}

# integrate training and test data separately 
hypo.anchors1 <- FindIntegrationAnchors(object.list = hypo.list1, dims = 1:30)
hypo.anchors2 <- FindIntegrationAnchors(object.list = hypo.list2, dims = 1:30)

hypo.integrated1 <- IntegrateData(anchorset = hypo.anchors1, dims = 1:30)
hypo.integrated2 <- IntegrateData(anchorset = hypo.anchors2, dims = 1:30)

DefaultAssay(hypo.integrated1) <- 'RNA'
DefaultAssay(hypo.integrated2) <- 'RNA'

set.seed(12345)
# standardize the normalized count 
hypo.integrated1 <- ScaleData(hypo.integrated1, verbose = FALSE)
hypo.integrated2 <- ScaleData(hypo.integrated2, verbose = FALSE)

##--2.select top 2k highly expressed genes----
# select the top2k highly expressed genes
# dataslot==counts --> raw counts
# a) dataslot==data --> LogNormalized
# b) dataslot==scale.data --> lognormalized + scaling (for each gene, standardized its express across cells)

topk_genes <- function(data, topk){
  df = as.matrix(rowSums(data))
  df <- as.matrix(df[order(df[,1],decreasing=TRUE),])
  topk <- rownames(df)[1:topk]
  return(topk) 
}

# a) dataslot=counts
sparse_lognorm_1 <- hypo.integrated1[['RNA']]@data
sparse_lognorm_2 <- hypo.integrated2[['RNA']]@data

# gene/feature intersection between the two sets
top2k_lognorm_intersect <- Reduce(intersect,list(topk_genes(sparse_lognorm_1, 2000),
                                                 topk_genes(sparse_lognorm_2, 2000)))

write.csv(t(as.matrix(sparse_lognorm_1[top2k_lognorm_intersect,])), 'test_heg2k_lognorm_intersect.csv')
write.csv(t(as.matrix(sparse_lognorm_2[top2k_lognorm_intersect,])), 'train_heg2k_lognorm_intersect.csv')

# b) dataslot==scale.data
sparse_std_1 <- hypo.integrated1[['RNA']]@scale.data
sparse_std_2 <- hypo.integrated2[['RNA']]@scale.data

write.csv(t(as.matrix(sparse_std_1[top2k_lognorm_intersect,])), 'test_heg2k_std_intersect.csv')
write.csv(t(as.matrix(sparse_std_2[top2k_lognorm_intersect,])), 'train_heg2k_std_intersect.csv')

##--3.integrate samples in the training and test sets separately (to remove batch effects), and then select top2k HVGs----
# to scale the data in the 'integrated' assay and perform CCA
integration <- function(hypo.integrated){
  DefaultAssay(hypo.integrated) <- 'integrated'
  hypo.integrated <- ScaleData(hypo.integrated, verbose = FALSE)
  hypo.integrated <- RunPCA(hypo.integrated, npcs = 30, verbose = FALSE)
  ElbowPlot(hypo.integrated)
  hypo.integrated <- FindNeighbors(hypo.integrated, dims = 1:30)
  hypo.integrated <- FindClusters(hypo.integrated, resolution = 1.5)
  hypo.integrated <- RunUMAP(hypo.integrated, reduction = 'pca', dims = 1:30)
  return(hypo.integrated)
}

hypo.integrated1 <- integration(hypo.integrated1)
hypo.integrated2 <- integration(hypo.integrated2)

# to plot the hypo.integrated
Idents(hypo.integrated2) <- 'stim'
DimPlot(hypo.integrated2)

saveRDS(hypo.integrated1, '../snmouse_objects/test_hypo_integrated_2k.RDS')
saveRDS(hypo.integrated2, '../snmouse_objects/train_hypo_integrated_2k.RDS')


top2k_hvg_1 <- rownames(hypo.integrated1@assays[['integrated']])
top2k_hvg_2 <- rownames(hypo.integrated2@assays[['integrated']])

top2k_hvg_intersect <- Reduce(intersect, list(top2k_hvg_1,top2k_hvg_2))

# a) use the LogNormalized count data in the RNA assay
write.csv(t(as.matrix(sparse_lognorm_1[top2k_hvg_intersect,])), 'test_hvg2k_lognorm_intersect.csv')
write.csv(t(as.matrix(sparse_lognorm_2[top2k_hvg_intersect,])), 'train_hvg2k_lognorm_intersect.csv')

# b) use the standardized and logNormalized count data in the RNA assay
write.csv(t(as.matrix(sparse_std_1[top2k_hvg_intersect,])), 'test_hvg2k_std_intersect.csv')
write.csv(t(as.matrix(sparse_std_2[top2k_hvg_intersect,])), 'train_hvg2k_std_intersect.csv')

# c) use the standardized and logNormalized count data in the integrated assay
# check: the scaled data in the RNA assay is different from the scaled data in the integrated data (less batch effects)
DefaultAssay(hypo.integrated2) <- 'integrated'
train_matrix <- as.matrix(GetAssayData(object = hypo.integrated2, slot = 'scale.data'))
train_matrix2 <- as.matrix(GetAssayData(object = hypo.integrated2, slot = 'data'))

# the two are different
# train_matrix[top2k_hvg_intersect,][1:3, 1:3]
# AAACCCAAGAACGTGC-1_1 AAACCCAAGCACCAGA-1_1 AAACCCAGTTGCCGAC-1_1
# Avp             0.7680657           -0.3595898            0.2369399
# Pmch           -0.5246414           -0.3722886            3.0902190
# Oxt             0.5149147           -0.3567146            0.1251261
# train_matrix2[top2k_hvg_intersect,][1:3, 1:3]
# AAACCCAAGAACGTGC-1_1 AAACCCAAGCACCAGA-1_1 AAACCCAGTTGCCGAC-1_1
# Avp            1.08595223            0.2441755            0.6894753
# Pmch           0.08114933            0.1991795            2.8816399
# Oxt            0.97357064            0.2452628            0.6478747

sparse_integ1 <- hypo.integrated1[['integrated']]@scale.data
sparse_integ2 <- hypo.integrated2[['integrated']]@scale.data

write.csv(t(as.matrix(sparse_integ1[top2k_hvg_intersect,])), 'test_hvg2k_std_integrated.csv')
write.csv(t(as.matrix(sparse_integ2[top2k_hvg_intersect,])), 'train_hvg2k_std_integrated.csv')

##--4.integrate samples using their top2k highly expressed genes, and use the scaled data from the integrated assay----
top2k_lognorm1 <- topk_genes(sparse_lognorm_1, 2000)
hypo.anchors1 <- FindIntegrationAnchors(object.list = hypo.list1, anchor.features = top2k_lognorm1, dims = 1:30)

hypo.integrated1.2 <- IntegrateData(anchorset = hypo.anchors1, dims = 1:30)
hypo.integrated1.2 <- ScaleData(hypo.integrated1.2, verbose = FALSE)
hypo.integrated1.2 <- integration(hypo.integrated1.2)

Idents(hypo.integrated1.2) <- 'orig.ident'
DimPlot(hypo.integrated1.2)

sparse_integ1.2 <- hypo.integrated1.2[['integrated']]@scale.data
write.csv(t(as.matrix(sparse_integ1.2[top2k_lognorm_intersect,])), 'test_heg2k_std_integrated.csv')

top2k_lognorm2 <- topk_genes(sparse_lognorm_2, 2000)
hypo.anchors2 <- FindIntegrationAnchors(object.list = hypo.list2, anchor.features = top2k_lognorm2, dims = 1:30)

hypo.integrated2.2 <- IntegrateData(anchorset = hypo.anchors2, dims = 1:30)
hypo.integrated2.2 <- ScaleData(hypo.integrated2.2, verbose = FALSE)
hypo.integrated2.2 <- integration(hypo.integrated2.2)

Idents(hypo.integrated2.2) <- 'orig.ident'
DimPlot(hypo.integrated2.2)

sparse_integ2.2 <- hypo.integrated2.2[['integrated']]@scale.data
write.csv(t(as.matrix(sparse_integ2.2[top2k_lognorm_intersect,])), 'train_heg2k_std_integrated.csv')

##--misc and plotting----
top2k_hveg_intersect <- Reduce(intersect, list(top2k_lognorm_intersect,top2k_hvg_intersect))
# length(top2k_hveg_intersect)
# [1] 255
# only 255 genes were shared between top2k_heg and top2k hvgs, both intersect

Idents(hypo.integrated1) <- 'orig.ident'
# p1 <- DimPlot(hypo.integrated1, cols = c("#4998A1", "#6DACB3", "#DB5D6C", "#E27D89"))
# Idents(hypo.integrated1) <- 'stim'
p2 <- DimPlot(hypo.integrated1, cols = c("#25838E", "#D43D4F")) + 
  ggtitle('test set')+
  theme(plot.title = element_text(hjust = 0.5))

Idents(hypo.integrated2) <- 'orig.ident'
# p3 <- DimPlot(hypo.integrated2, cols = c("#92C1C6", "#B6D5D9", "#E99EA7", "#F0BEC4")) 
# Idents(hypo.integrated2) <- 'stim'
p4 <- DimPlot(hypo.integrated2, cols = c("#25838E", "#D43D4F")) + 
  ggtitle('training set')+
  theme(plot.title = element_text(hjust = 0.5)) +
  NoLegend()

p4|p2 

Idents(hypo.integrated1.2) <- 'stim'
p5 <- DimPlot(hypo.integrated1.2, cols = c("#25838E", "#D43D4F")) + 
  ggtitle('test set -- integrated on HEGs')+
  theme(plot.title = element_text(hjust = 0.5))

Idents(hypo.integrated2.2) <- 'stim'
p6 <- DimPlot(hypo.integrated2.2, cols = c("#25838E", "#D43D4F")) + 
  ggtitle('training set -- integrated on HEGs')+
  theme(plot.title = element_text(hjust = 0.5)) +
  NoLegend()

p6|p5 

 
# hypo.combined <- merge(hypo.integrated1, y = hypo.integrated2, 
#                        add.cell.ids = c("test", "train"), merge.data = TRUE, project = "snmouse")


##--sessionInfo()----
# R version 4.2.1 (2022-06-23)
# Platform: aarch64-apple-darwin20 (64-bit)
# Running under: macOS Monterey 12.4
# 
# Matrix products: default
# LAPACK: /Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib/libRlapack.dylib
# 
# locale:
#   [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
# 
# attached base packages:
#   [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# other attached packages:
#   [1] sp_1.5-0           SeuratObject_4.1.0 Seurat_4.1.1       forcats_0.5.2      stringr_1.4.1      dplyr_1.0.9        purrr_0.3.4       
# [8] readr_2.1.2        tidyr_1.2.0        tibble_3.1.8       ggplot2_3.3.6      tidyverse_1.3.2   
# 
# loaded via a namespace (and not attached):
#   [1] googledrive_2.0.0     Rtsne_0.16            colorspace_2.0-3      deldir_1.0-6          ellipsis_0.3.2        ggridges_0.5.3       
# [7] fs_1.5.2              spatstat.data_2.2-0   rstudioapi_0.14       farver_2.1.1          leiden_0.4.2          listenv_0.8.0        
# [13] ggrepel_0.9.1         fansi_1.0.3           lubridate_1.8.0       xml2_1.3.3            codetools_0.2-18      splines_4.2.1        
# [19] polyclip_1.10-0       jsonlite_1.8.0        broom_1.0.0           ica_1.0-3             cluster_2.1.4         dbplyr_2.2.1         
# [25] png_0.1-7             rgeos_0.5-9           uwot_0.1.14           spatstat.sparse_2.1-1 sctransform_0.3.4     shiny_1.7.2          
# [31] compiler_4.2.1        httr_1.4.4            backports_1.4.1       lazyeval_0.2.2        assertthat_0.2.1      Matrix_1.4-1         
# [37] fastmap_1.1.0         gargle_1.2.0          cli_3.3.0             later_1.3.0           htmltools_0.5.3       tools_4.2.1          
# [43] igraph_1.3.4          gtable_0.3.0          glue_1.6.2            reshape2_1.4.4        RANN_2.6.1            Rcpp_1.0.9           
# [49] scattermore_0.8       cellranger_1.1.0      vctrs_0.4.1           nlme_3.1-159          progressr_0.10.1      lmtest_0.9-40        
# [55] spatstat.random_2.2-0 globals_0.16.0        rvest_1.0.3           mime_0.12             miniUI_0.1.1.1        lifecycle_1.0.1      
# [61] irlba_2.3.5           goftest_1.2-3         googlesheets4_1.0.1   future_1.27.0         MASS_7.3-58.1         zoo_1.8-10           
# [67] scales_1.2.1          spatstat.core_2.4-4   spatstat.utils_2.3-1  hms_1.1.2             promises_1.2.0.1      parallel_4.2.1       
# [73] RColorBrewer_1.1-3    gridExtra_2.3         reticulate_1.25       pbapply_1.5-0         rpart_4.1.16          stringi_1.7.8        
# [79] rlang_1.0.4           pkgconfig_2.0.3       matrixStats_0.62.0    lattice_0.20-45       tensor_1.5            ROCR_1.0-11          
# [85] labeling_0.4.2        patchwork_1.1.2       htmlwidgets_1.5.4     cowplot_1.1.1         tidyselect_1.1.2      parallelly_1.32.1    
# [91] RcppAnnoy_0.0.19      plyr_1.8.7            magrittr_2.0.3        R6_2.5.1              generics_0.1.3        DBI_1.1.3            
# [97] mgcv_1.8-40           pillar_1.8.1          haven_2.5.1           withr_2.5.0           fitdistrplus_1.1-8    abind_1.4-5          
# [103] survival_3.4-0        future.apply_1.9.0    modelr_0.1.9          crayon_1.5.1          KernSmooth_2.23-20    utf8_1.2.2           
# [109] spatstat.geom_2.4-0   plotly_4.10.0         tzdb_0.3.0            grid_4.2.1            readxl_1.4.1          data.table_1.14.2    
# [115] reprex_2.0.2          digest_0.6.29         xtable_1.8-4          httpuv_1.6.5          munsell_0.5.0         viridisLite_0.4.1  