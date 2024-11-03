# DeepB<sup>3</sup>P<sup>3</sup>: masked peptide transformer for low-data peptide drug discovery
<p align="center">
  <img src="./fig/flowchart.png">
</p>

### Installation
Please see requirements.txt.

### Datasets
| Source | Total number | BBBPs | non-BBBPs  |
| ------ | ------------ | ----- | ---------- |
| B3Pred Training set   | 2367  | 215 | 2152 |
| B3Pred Testing set    | 592   | 54  | 538  |

### Masking peptides for small data challenge
The size of drug discovery datasets can be extremely limited due to the high cost of the experiments ([1](https://pubs.acs.org/doi/10.1021/acscentsci.6b00367),[2](https://pubs.acs.org/doi/10.1021/acs.chemrev.3c00189)). However, the training of modern neural networks typically requires large-scale high-quality data. In this paper, we introduce 'masked peptide' that can significantly overcome this issue (Fig. (A)).

Unlike other data augmentation methods, our masking peptide technique does not involve any substitution, insertion, or deletion, but it can significantly change the latent distribution, as follows.
<p align="center">
  <img src="./fig/tsne.png">
</p>

### Training
```bash
mkdir temp
python DeepB3P3.py \
    --train_path 'bbbp/d3_train_a1x8.txt' \
    --test_path 'bbbp/d3_test_a1x8.txt' \
    --result_path 'temp/d3_test.pred.txt' \
    --log_path 'temp/d3_test.txt.log' \
    --max_length 75 \
    --conv1_kernel 10 \
    --conv2_kernel 10 \
    --regCLASS --LR 0.001 --EVALUATE_ALL --NUM_EPOCHS 50
```
Or experiment with multiple magnitudes of data augmentation using a single script.
```bash 
mkdir collect
bash run.sh
```
### Analysis
Pretrained model files: [Google Drive](https://drive.google.com/file/d/1OiLLq8UKR1_d833OXIEFZoIIzItwMcpv/view?usp=sharing).
Please download the file (163MB) and unzip to 'DeepB3P3/collect/8/max75'. Then follow the jupyter notebook 'DeepB3P3_Analysis.ipynb'.
### Reference
```
@article{ma2023prediction,
  title={A prediction model for blood-brain barrier penetrating peptides based on masked peptide transformers with dynamic routing},
  author={Ma, Chunwei and Wolfinger, Russ},
  journal={Briefings in Bioinformatics},
  volume={24},
  number={6},
  pages={bbad399},
  year={2023},
  publisher={Oxford University Press}
}
```
Please let [me](mailto:horsepurve@gmail.com) know if you have any questions about this research.
