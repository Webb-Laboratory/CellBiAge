library(Seurat)

setwd("/gpfs/data/awebb/dyu20/cell2location/MLAging/data/svz_10x/")
# train_y57o57 <- readRDS('../svz_processed/train_y57o57.RDS')

#####-----------preprocess--------
files <- list.files("./", recursive = FALSE)

object_list <- c()
for (i in (1:length(files))) {
  print(files[i])
  data <- Read10X(paste0(files[i], "/"))
  obj <- CreateSeuratObject(counts = data, project = files[i], min.cells = 3, min.features = 200)
  obj[['percent.mt']] <- PercentageFeatureSet(obj, pattern = '^mt-')
  print(dim(obj))
  object_list <- c(object_list, obj)
}

set.seed(12345)
for (i in (1:length(files))) {
  print(paste(files[i], "before filteration", dim(object_list[[i]])[2]))
  object_list[[i]] <- subset(object_list[[i]])
  hig_ft <- quantile(object_list[[i]]$nFeature_RNA, probs = 0.98)
  print(hig_ft)
  # object_list[[i]] <- subset(object_list[[i]], subset = nFeature_RNA < 7500  & percent.mt < 10)
  object_list[[i]] <- subset(object_list[[i]], subset = nFeature_RNA < hig_ft  & percent.mt < 10)
  
  print(paste("after filteration", dim(object_list[[i]])[2]))
  object_list[[i]] <- NormalizeData(object_list[[i]], verbose = FALSE)
  object_list[[i]] <- FindVariableFeatures(object_list[[i]], selection.method = 'vst', nfeatures = 5000, verbose = FALSE)
}

svz.nonint <- Reduce(merge, object_list)
svz.nonint <- ScaleData(svz.nonint, verbose = FALSE)
svz.nonint <- FindVariableFeatures(svz.nonint, selection.method = 'vst', nfeatures = 5000, verbose = FALSE)
svz.nonint <- RunPCA(svz.nonint, npcs = 30, verbose = FALSE)
svz.nonint <- FindNeighbors(svz.nonint, dims = 1:30)
svz.nonint <- RunUMAP(svz.nonint, reduction = 'pca', dims = 1:30)
svz.nonint$batch <- "2"
svz.nonint@meta.data[(svz.nonint$orig.ident %in% c("Y1", "Y2", "O2", "Y3", "Y4", "O3", "O4")), 'batch'] <- "1"
saveRDS(svz.nonint, "../svz_processed/svz.nonint.RDS")

library(patchwork)
Idents(szv.nonint) <- "batch"
p1 <- DimPlot(szv.nonint) + NoAxes()
p2 <- DimPlot(szv.nonint, split.by = "batch") + NoAxes()
p1 + p2 + plot_layout(ncol = 2, widths = c(1, 2), heights = c(1, 1))

svz.nonint$batch <- "2"
svz.nonint@meta.data[(svz.nonint$orig.ident %in% c("Y1", "Y2", "O2", "Y3", "Y4", "O3", "O4")), 'batch'] <- "1"
# saveRDS(svz.nonint, "../svz_processed/svz.nonint.RDS")

test_34$batch <- "2"
test_34@meta.data[(test_34$orig.ident %in% c("Y3", "Y4", "O3", "O4")), 'batch'] <- "1"
Idents(test_34) <- "batch"
p1 <- DimPlot(test_34) + NoAxes() + NoLegend()
p2 <- DimPlot(test_34, split.by = "batch") + NoAxes() 
p1 + p2 + plot_layout(ncol = 2, widths = c(1, 2), heights = c(1, 1))

test_integ_by_train_ctl$batch <- "2"
test_integ_by_train_ctl@meta.data[(test_integ_by_train_ctl$orig.ident %in% c("Y1", "Y2", "O2")), 'batch'] <- "1"
Idents(test_integ_by_train_ctl) <- "batch"
p1 <- DimPlot(test_integ_by_train_ctl) + NoAxes()+ NoLegend()
p2 <- DimPlot(test_integ_by_train_ctl, split.by = "batch") + NoAxes()
p1 + p2 + plot_layout(ncol = 2, widths = c(1, 2), heights = c(1, 1))

train_57 <- c(object_list[[12]], object_list[[14]], object_list[[4]], object_list[[6]])

test_12 <- c(object_list[[12]], object_list[[14]], object_list[[4]], object_list[[6]],
             object_list[[8]], object_list[[9]], object_list[[1]])

test_34 <- c(object_list[[12]], object_list[[14]], object_list[[4]], object_list[[6]], 
             object_list[[10]], object_list[[11]], object_list[[2]], object_list[[3]])

test_68 <- c(object_list[[12]], object_list[[14]], object_list[[4]], object_list[[6]], 
             object_list[[13]], object_list[[15]], object_list[[5]], object_list[[7]])

integration <- function(seurat.integrated){
  seurat.anchors <- FindIntegrationAnchors(object.list = seurat.integrated, dims = 1:30)
  seurat.integrated <- IntegrateData(anchorset = seurat.anchors, dims = 1:30)
  
  DefaultAssay(seurat.integrated) <- 'RNA'
  seurat.integrated <- ScaleData(seurat.integrated, verbose = FALSE)
  
  DefaultAssay(seurat.integrated) <- 'integrated'
  seurat.integrated <- ScaleData(seurat.integrated, verbose = FALSE)
  seurat.integrated <- RunPCA(seurat.integrated, npcs = 30, verbose = FALSE)
  ElbowPlot(seurat.integrated)
  seurat.integrated <- FindNeighbors(seurat.integrated, dims = 1:30)
  seurat.integrated <- FindClusters(seurat.integrated, resolution = 1.5)
  seurat.integrated <- RunUMAP(seurat.integrated, reduction = 'pca', dims = 1:30)
  return(seurat.integrated)
}
##############################
# repeat
#############################

rename_cells <- function(input_object){
  for(n in (1:length(input_object))){
    cn = unlist(lapply(strsplit(colnames(input_object[[n]]), "-", fixed = T), function(x) x[1]))
    id = unique(input_object[[n]]$orig.ident)
    print(id)
    input_object[[n]] = RenameCells(input_object[[n]],
                                new.names = paste0(cn, "-", gsub("_", "-", id, fixed = T)))
  }
  return(input_object)
}
train_57 <- rename_cells(train_57)
test_12 <- rename_cells(test_12)
test_34 <- rename_cells(test_34)
test_58 <- rename_cells(test_58)

train_57 <- integration(train_57)
saveRDS(train_57, "../svz_processed/train_57.RDS")

test_12 <- integration(test_12)
saveRDS(test_12, "../svz_processed/test_12.RDS")
# it's the same as the test_ctlinteg_by_train.RDS

test_34 <- integration(test_34)
saveRDS(test_34, "../svz_processed/test_34.RDS")

test_68 <- integration(test_68)
saveRDS(test_68, "../svz_processed/test_68.RDS")
#####-----------preprocess--------
train_57 <- readRDS("../svz_processed/train_57.RDS") 
test_integ_by_train_ctl <- readRDS('../svz_processed/test_ctlinteg_by_train.RDS')

test_34 <- readRDS("../svz_processed/test_34.RDS") 
test_68 <- readRDS("../svz_processed/test_68.RDS") 
szv.nonint <- readRDS("../svz_processed/svz.nonint.RDS") 


svz <- readRDS("../seurat.SVZ.annotated.2020-04-27.rds") 

meta_df_orig <- svz@meta.data
cn <- unlist(lapply(strsplit(rownames(meta_df_orig), "_", fixed = T), function(x) x[1]))
rownames(meta_df_orig) <- paste0(cn, "-", gsub("_", "-", unlist(meta_df_orig$ID), fixed = T))

top2k_hvg_train <- rownames(train_57@assays[['integrated']])
top2k_hvg_test_34 <- rownames(test_34@assays[['integrated']])
top2k_hvg_test_68 <- rownames(test_68@assays[['integrated']])
top2k_hvg_test_clt <- rownames(test_integ_by_train_ctl@assays[['integrated']])


top2k_hvg_intersect <- Reduce(intersect, list(top2k_hvg_train,top2k_hvg_test_34, top2k_hvg_test_68, top2k_hvg_test_clt))

top2k_hvg_train <- train_57[['integrated']]@scale.data
top2k_hvg_test_clt <- test_integ_by_train_ctl[['integrated']]@scale.data
top2k_hvg_test_34 <- test_34[['integrated']]@scale.data
top2k_hvg_test_68 <- test_68[['integrated']]@scale.data

library(stringr)
finalize_df <- function(input_df,exclude_l, O1, O2){
  integ <- t(as.data.frame(input_df[top2k_hvg_intersect,]))
  animal <- data.frame(animal = as.vector(str_extract(rownames(integ), "(?<=-)[^-]+$")))
  integ <- cbind(integ, animal)
  integ <- integ[!integ$animal %in% exclude_l,]
  
  target <- data.frame(target = rep(0, nrow(integ)))
  integ <- cbind(integ, target)
  
  integ[(integ$animal==O1) | (integ$animal==O2),"target"] <- 1
  print(table(integ$animal, integ$target))
  return (integ)
}

integ1_train <- finalize_df(top2k_hvg_train, NULL, "O5", "O7")
# 0    1
# O5    0 6038
# O7    0 7706
# Y5 8834    0
# Y7 7083    0

# the newest
# 0    1
# O5    0 5921
# O7    0 7560
# Y5 8666    0
# Y7 6951    0

integ2_test <- finalize_df(top2k_hvg_test_clt, c("Y5", "Y7", "O5", "O7"), "O2", "O2")
# 0    1
# O2    0 4394
# Y1 2369    0
# Y2 4372    0

integ2_test_34 <- finalize_df(top2k_hvg_test_34, c("Y5", "Y7", "O5", "O7"), "O3", "O4")
# 0    1
# O3    0 4124
# O4    0 1934
# Y3 5760    0
# Y4 5204    0
integ2_test_68 <- finalize_df(top2k_hvg_test_68, c("Y5", "Y7", "O5", "O7"), "O6", "O8")
# 0    1
# O6    0 5341
# O8    0 6664
# Y6 6963    0
# Y8 7872    0

add_cell_type <- function(input_df,last_element){
  input_df$rn <- rownames(input_df)
  input_df_celltype <- merge(input_df, meta_df_orig[,22:23], by = "rn", all = FALSE)
  input_df_celltype$rn <- NULL
  colnames(input_df_celltype)[last_element] <- "major_group"
  return (input_df_celltype)
}

meta_df_orig$rn <- rownames(meta_df_orig)

integ1_train_celltype <- add_cell_type(integ1_train, 1674)
integ2_test_celltype <- add_cell_type(integ2_test, 1674)
integ2_test_34_celltype <- add_cell_type(integ2_test_34, 1674)
integ2_test_68_celltype <- add_cell_type(integ2_test_68, 1674)

dim(integ1_train_celltype)
# 27939  1634
# [1] 27380  1674
dim(integ2_test_celltype)
# 10377  1634
dim(integ2_test_34_celltype)
# 15874  1634
dim(integ2_test_68_celltype)
# 25492  1634

write.csv(integ1_train_celltype, '../svz_processed/svz_ctl_train_cell_sep3integ_batch1.csv')
write.csv(integ2_test_celltype, '../svz_processed/svz_ctl_test_cell_sep3integ_batch2.csv')

write.csv(integ2_test_34_celltype, '../svz_processed/svz_ex_test_34_cell_sep3integ_batch2.csv')
write.csv(integ2_test_68_celltype, '../svz_processed/svz_ex_test_68_cell_sep3integ_batch2.csv')

