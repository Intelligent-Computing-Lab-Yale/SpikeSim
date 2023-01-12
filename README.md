# SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks
### This repository contains the Pytorch-based evaluation codes for [SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks]. https://arxiv.org/pdf/2210.12899.pdf
 
The repository consists of two hardware evaluation tools: 1) Non-ideality Computation Engine (NICE) and 2) Energy-Latency-Area (ELA) Tool. It also contains the code for quantization-aware SNN training. For reference, we have also provided a pre-trained model path for a 4-bit quantized VGG9 SNN on CIFAR10 dataset. 

## Quantization-aware (weights only) SNN Training
```shell
cd SNN_train_infer_quantization_ela
python train.py --lr 0.001 --encode 'd' --arch 'vgg9' --T 5 --quant 4
```
## Hardware-realistic Inference using the NICE
```shell
cd NICE_Evaluation
python hw_inference.py --num_steps 5 --arch 'vgg9' --batch_size 128 --b_size 4 --ADC_precision 4 --quant 4
```
## Hardware-realistic energy-latency-area evaluation
```shell
cd SNN_train_infer_quantization_ela
python ela_spikesim.py 

## Variable Description 
________________________________________________________________________________________
| Variable     | Type | Length            | Description                                |
|--------------|------|-------------------|--------------------------------------------|
| in_ch_list   | list | No. of SNN Layers | Layer-wise input channel count             |
| out_ch_list  | list | No. of SNN Layers | Layer-wise output channel count            |
| in_dim_list  | list | No. of SNN Layers | Layer-wise input feature size              |
| out_dim_list | list | No. of SNN Layers | Layer-wise output feature size             |
| xbar_size    | int  | -                 | Crossbar Size                              |
| kernel_size  | int  | -                 | SNN Kernel Size                            | 
| pe_per_tile  | int  | -                 | No. of Processing Engines (PE) in one tile |
| time_steps   | int  | -                 | No. of Time Steps                          |
| clk_freq     | int  | -                 | Clock Frequency in MHz                     | 
----------------------------------------------------------------------------------------

```
## Citation
Please consider citing our paper:

```
@article{moitra2022spikesim,
  title={SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks},
  author={Moitra, Abhishek and Bhattacharjee, Abhiroop and Kuang, Runcong and Krishnan, Gokul and Cao, Yu and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2210.12899},
  year={2022}
}
```
## Ackonwledgements

Code for SNN training with quantization has been adapted from https://github.com/jiecaoyu/XNOR-Net-PyTorch 
