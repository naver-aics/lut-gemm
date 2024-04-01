# LUT-GEMM

## Model Quantization Examples

Run the following commands to get the binary matrices and scaling factor matrices from the pre-trained weights.

``` sh
python quant_model_bcq.py \
    --model_name_or_path facebook/opt-125m \
    --qbits 4 \
    --group_size 128
```

``` sh
python quant_model_rtn.py \
    --model_name_or_path facebook/opt-125m \
    --qbits 4 \
    --group_size 128
```