# MLAging
[Webb laboratory](https://www.webblabatbrown.com/) and [Singh laboratory](https://rsinghlab.org/) @Brown

ML applications in predicting cellcular age using female mouse hypothalamus single nuclei RNAseq. 

### Models tested:
- Logistic regression with regularization
- Tree-based models
- Support Vector Machine Classifier (SVC)
- Multi-Layer Perceptron (MLP)


### Virtual Enviroments:

- For MLP: ```requirments_mlp.txt```
Other required packages for MLP using GPU:
```
module load cuda/11.7.1
module load cudnn/8.2.0
```
To implement keras tuner for group-based cross validation (MLP): 

in terminal: 
``` 
cd scripts
python3 mlp_kt_4cv_console.py
```

- For ML models (logistic regression, tree-based models, and SVC):

```environment_ml.yml```

### [Dataset](https://drive.google.com/drive/folders/1AxRl1PlOIWvgR9lBwkHN-pPNbX9ELou_?usp=sharing)
[Hajdarovic, K. H., Yu, D., Hassell, L. A., Evans, S. A., Packer, S., Neretti, N., & Webb, A. E. (2022). Single-cell analysis of the aging female mouse hypothalamus. *Nature Aging*, 2(7), 662-678.](https://www.nature.com/articles/s43587-022-00246-4):
- raw: ```filtered_feature_bc_matrix/``` folder
- input training and testing count matrix ```.csv```

### [Preliminary Figures](https://drive.google.com/file/d/1eoS2lvJQm9viL1mqS9ujvOfcdE419Cn4/view?usp=sharing)



