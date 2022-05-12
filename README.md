# 2470 Final Project
Doudou Yu, Guanjie Linghu, Manlin Li, Yihuan Hu

### Introduction:

In this group project, we applied deep learning to predict cellular age using the single-cell RNA sequencing (scRNAseq) dataset. This is a binary classification problem (young or aged). We are interested in this topic because the global aged population has increased tremendously and aging is one of the greatest risk factors for cancer, neurodegenerative diseases, etc. However, it is very hard to develop anti-aging interventions, partially due to the difficulties in accessing cellular age, especially for single cells. With the evolution of scRNAseq, we are able to understand transcriptomics at single-cell resolution. The motivation for our project is to learn young and aged cellular signatures using deep learning, which can be used to assess whether a specific anti-aging intervention is effective or not.

### Methodology:

Our main model is based on Multilayer Perceptron (MLP), and we implemented various variant of it including PCA + MLP, Autoencoder + MLP, Deepcount Autoencoder (DCA) + MLP, Binarized Data + MLP, and Capsule + MLP.

### Result:

With binarized data, the best-performed model achieved an accuracy of 0.96 and an AUC of 0.98.
