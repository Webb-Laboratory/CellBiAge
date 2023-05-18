library(tidyverse)
library(glmnet)
library(purrr)
library(ggplot2)
library(ROCR)
library(PRROC)

models <- readRDS("data/models_all_bootstrap.rds")
colnames(models)[1] <- "Celltype"
models$lognormalized <- NULL
#==================================================================================================
# Load hypo bootstrapCells

ex_data <- readRDS("data/bootstrap_pseudocell_15_hypo.rds")

meta <- as.data.frame(df2[, c(1:4)])
umi <- df2[, -c(1:4)]
normed <- sweep(umi, MARGIN = 1, FUN = "/", STATS = rowSums(umi))
logged <- log1p(normed * 10000)

by_celltype <- as_tibble(cbind(meta, logged)) %>%
            group_by(Celltype) %>%
            nest()

models <- models %>%
  mutate(Celltype = case_when(
    Celltype == "Oligodendro" ~ "Oligodendrocyte",
    Celltype == "Astrocyte_qNSC" ~ "Astrocyte",
    TRUE ~ as.character(Celltype)
  ))

by_celltype <-  dplyr::inner_join(by_celltype, models, by = "Celltype")

by_celltype[1,2][[1]][[1]][1:5,1:4]

custom_add_predictions <- function(data, model) {
    predictions <- predict(model, newx = as.matrix(data[,-(1:3)]), s = "lambda.min")
    p <- as_tibble(data.frame("Pred" = predictions[,1],
                              "Sample" = data[,1],
                              "Mouse" = data[,2],
                              "Year" = data[,3]))
    return(p)
}

# Add predictions 
by_celltype <- by_celltype %>%
             mutate(Predictions = map2(data, model, custom_add_predictions))

output <- by_celltype %>% select(-c(data, model)) %>% unnest(Predictions)

range(output$Pred)

# saveRDS(output, "data/hypo_predictions.rds")
output <- readRDS("data/hypo_predictions.rds")

#==================================================================================================
output$Year <- factor(output$Year, levels = c("Young", "Aged"))

output <- output %>%
  mutate(Year = case_when(
    Year == "Young" ~ "0",
    Year == "Aged" ~ "1",
    TRUE ~ as.character(Year)
  ))

ggplot(output, aes(x = Celltype, y = Pred, fill = Year)) +
  geom_violin() +
  labs(x = "Celltype", y = "Pred") +
  ggtitle("Violin Plot of Pred by Celltype and Year")

output_test <- output[output$Sample %in% c("Young_1", "Young_2", "Aged_1", "Aged_2"), ]
ggplot(output_test, aes(x = Celltype, y = Pred, fill = Year)) +
  geom_violin() +
  labs(x = "Celltype", y = "Pred") +
  ggtitle("Violin Plot of Pred by Celltype and Year")

# Split the data frame by Celltype
df_list <- split(output_test, output_test$Celltype)

# Initialize a list to store the pr_curves for each Celltype
pr_curves <- list()

# Loop over each Celltype and calculate pr_curve
for (celltype in names(df_list)) {
  subset_df <- df_list[[celltype]]
  pr_curve <- pr.curve(scores.class0 = subset_df$Pred, weights.class0 = as.numeric(subset_df$Year), curve = TRUE)
  pr_curves[[celltype]] <- pr_curve
}

# Assuming you have a list of pr_curves named pr_curves, with entries for "Oligodendrocyte", "Astrocyte", and "Microglia"

par(mfrow = c(1, 3), mar = c(4, 4, 1, 1), pty = "s")

plot(pr_curves[["Oligodendrocyte"]], col = "black") 
plot(pr_curves[["Astrocyte"]], col = "black")
plot(pr_curves[["Microglia"]], col = "black")

# Reset the graphical parameters to default
par(mfrow = c(1, 1))
