# Efficient Multimodal Tiny Cycle-Attention-Attention-Transformer Toward Cancer Diagnosis



## Abstract:

In this work, we present a Cycle-Attention Transformer (Cy-Atten) framework that learns a more interpretable and robust joint representation for multimodal cancer survival analysis and is proven to effectively reduce the parameter redundancy via a cycling mechanism with sparse attention. The Cy-Atten yields a theoretical cost of $O(N)$ for adding a modality. As a comparison, an existing multimodal fusion method based on Kronecker Product, PaF \cite{03chen}, yields a theoretical cost of $O((m+1)^N)$ (m is a constant) for adding a modality. Empirically in the experiments on two canser datasets from the TCGA database, the parameters in our fusion part are only of the order of $\mathbf{10^{3}}$ considering three modalities while the parameters in the fusion part of PaF are of the order of $\mathbf{10^6}$. In addition to being lightweight, the Cy-Atten outperforms several state-of-the-art multimodal fusion methods in survival outcome prediction and grade classification of two cancers from the TCGA database.

![overall_architecture](overall_architecture.png)



## **Dataset:**

We use glioma and clear cell renal cell carcinoma (CCRCC) data from the TCGA. TCGA is a well-known cancer database, which has a lot of genomic information and cell slice image data. The datasets used in our work are the Glioma dataset from the TCGA-GBMLGG and CCRCC dataset from the TCGA-KIRC. The histology images, cell graphs and genomic features used in our work are consistent with those of Pathomic Fusion [^1]. The data can be downloaded from the [following link](https://drive.google.com/drive/u/1/folders/1swiMrz84V3iuzk8x99vGIBd5FCVncOlf). Checkpoints and corrected graph data can be downloaded from [here](https://drive.google.com/drive/folders/1jBNjmARYAxgraCctt5qZAVmdH-vuVF5H). Because the cell graphs of Pathomic Fusion could not be used directly due to version incompatibility of some packages, we stored the processed ones in folders named 'graph_GBMLGG' and 'graph_KIRC'. You need to place the compressed file named 'graph_GBMLGG' ('graph_KIRC') in the second link inside the 'data/TCGA_GBMLGG' ('data/TCGA_KIRC') of the first link and unzip it.



## Enviroment Setup

Our models are trained with NVIDIA GPUs (NVIDIA RTX A2000, NVIDIA RTX A4000, NVIDIA A30 and NVIDIA A40), torch = 1.8.1 and torch_geometric = 2.0.4.  Richard J. Chen's cell graphs are stored with torch_geometric = 1.3.0, so there may be incompatibility issues when reading.



## Train and Test

### Representation Learning

You can choose the task type by selecting 'surv' or 'grade' at the 'exp_name' and 'task'. Similarly, you can change the mode by changing 'mode' and 'model_name'. The following commands are used to train SAGE for survival outcome prediction and grade classification, respectively. Detailed training parameters are described in the article.

```
python train_cv.py  --exp_name surv_15_rnaseq --task surv --mode graph --model_name graph --niter 10 --niter_decay 150 --batch_size 32 --lr 0.0001 --begin_k 1 --init_type max --reg_type none --lambda_reg 0 --gpu_ids 0
```

```
python train_cv.py  --exp_name grad_15 --task grad --mode graph --model_name graph --niter 10 --niter_decay 190 --batch_size 32 --lr 0.0003 --init_type max --reg_type none --lambda_reg 0 --gpu_ids 0 --label_dim 3 --begin_k 1 --act LSM 
```



### Multimodal Models

The following commands are used to train Cro-Atten (PathOmic) for survival outcome prediction and grade classification, respectively. You can choose Cro-Atten (GraphOmic) by changing the 'mode' and 'model_name' according to your needs. Detailed training parameters are described in the article.

```
python train_cv.py --exp_name surv_15_rnaseq --task surv --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 150 --lr 0.0003 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 0 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320 --batch_size 32 --finetune 1 --myfusion CrossAttention --optimizer_type adam --Tfnum 4 --lastnum 0 --begin_k 1 --lr_policy cosine --reg_type none --input_size_path 224 --position_C 1 --position_S 1 --use_conv_stem 1 --use_sparsemax 1
```

```
python train_cv.py --exp_name grad_15 --task grad --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 190 --batch_size 32 --lr 0.0003 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 0 --gpu_ids 0 --path_gate 0 --omic_scale 2 --act LSM --label_dim 3 --finetune 1 --myfusion CrossAttention --Tfnum 6 --lastnum 0  --use_conv_stem 1 --use_sparsemax 1 --position_C 1 --position_S 1 --input_size_path 224 --path_dim 32 --omic_dim 32 --input_size_omic 80 --begin_k 1 --lr_policy cosine --reg_type none
```

The parameters in the following commands are used to train Cy-Atten (PathGraphOmic) for survival outcome prediction and grade classification, respectively. You can choose Tri-Co-Atten or MulT by changing the 'mode' and 'model_name' according to your needs. Detailed training parameters are described in the article.

```
python train_cv.py --exp_name surv_15_rnaseq --task surv --mode pathgraphomic --model_name pathgraphomic_fusion --niter 10 --niter_decay 190 --batch_size 32 --finetune 1 --lr 0.0003 --beta1 0.5 --lr_policy cosine --fusion_type pofusion_A --mmhid 192 --use_bilinear 1 --Tfnum 1 --lastnum 0 --gpu_ids 0 --optimizer_type adam --reg_type omic --omic_gate 0 --grph_scale 2 --input_size_path 224 --use_rnaseq 1 --use_sparsemax 1 --input_size_omic 320 --myfusion Cy_Atten --graph_model SAGE --begin_k 1 --position_C 1 --position_G 1 --position_S 1
```

```
python train_cv.py --exp_name grad_15 --task grad --mode pathgraphomic --model_name pathgraphomic_fusion --niter 10 --niter_decay 190 --lr 0.0003 --batch_size 32 --finetune 1 --beta1 0.5 --fusion_type pofusion_B --mmhid 192 --use_bilinear 1 --gpu_ids 0 --path_gate 0 --act LSM --label_dim 3 --Tfnum 1 --lastnum 0 --optimizer_type adam --reg_type omic --input_size_path 224 --position_C 1 --position_G 1 --position_S 1 --myfusion Cy_Atten --graph_model SAGE --lr_policy cosine --use_sparsemax 1 --begin_k 1
```

## Partial Results


|      Model      | C-Index (Glioma) | AUC (Glioma) | AP (Glioma) | F1-micro (Glioma) | F1-GradeIV (Glioma) | C-Index (CCRCC) |
| :-------------: | :--------------: | :----------: | :---------: | :---------------: | :-----------------: | :-------------: |
| Cro-Atten (PO)  |      0.856       |    0.923     |    0.860    |       0.771       |        0.940        |      0.745      |
| Cro-Atten (GO)  |      0.845       |    0.913     |    0.845    |       0.717       |        0.926        |      0.743      |
|  Tri-Co-Atten   |      0.838       |    0.918     |    0.856    |       0.751       |        0.923        |      0.757      |
|      MulT       |      0.857       |    0.923     |    0.863    |       0.756       |        0.930        |      0.740      |
| Cy-Atten (Ours) |      0.863       |    0.929     |    0.873    |       0.772       |        0.941        |      0.765      |


[^1]: Chen, R. J., Lu, M. Y., Wang, J., Williamson, D. F., Rodig, S. J., Lindeman, N. I., & Mahmood, F. (2020). Pathomic fusion: an integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis. *IEEE Transactions on Medical Imaging*, *41*(4), 757-770.


