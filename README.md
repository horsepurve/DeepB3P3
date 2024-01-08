# DeepB<sup>3</sup>P<sup>3</sup>: masked peptide transformer for low-data peptide drug discovery
<p align="center">
  <img src="./fig/flowchart.png">
</p>

### Installation
See requirements.txt.

### Datasets

### Training
```bash
python DeepB3P3.py \
    --train_path $train_path \
    --test_path $test_path \
    --result_path $result_path \
    --log_path $log_path \
    --max_length 75 \
    --conv1_kernel 10 \
    --conv2_kernel 10 \
    --regCLASS --LR 0.001 --EVALUATE_ALL --NUM_EPOCHS 50
```
Or experiment with multiple magnitudes of data augmentation:
```bash 
mkdir collect
bash run.sh
```
### Analysis

### Reference

