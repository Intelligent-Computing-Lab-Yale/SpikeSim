# SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks
### This repository contains the Pytorch-based evaluation codes for [SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks]. https://arxiv.org/pdf/2210.12899.pdf
 
The repository consists of two hardware evaluation tools: 1) Non-ideality Computation Engine (NICE) and 2) Energy-Latency-Area (ELA) Tool. It also contains the code for quantization-aware SNN training. For reference, we have also provided a pre-trained model path for a 4-bit quantized VGG9 SNN on CIFAR10 dataset. 

## Quantization-aware (weights only) SNN Training
```shell
python train.py --lr 0.001 --encode 'd' --arch 'vgg9' --T 5 --quant 4
```
## Hardware-realistic Inference using the NICE
```shell
python hw_inference.py --num_steps 5 --arch 'vgg9' --batch_size 128 --b_size 4 --ADC_precision 4 --quant 4
```
## Hardware-realistic energy-latency-area evaluation
```shell
python ela_spikesim.py 

## Variable Description 
```
## Ackonwledgements

Code for SNN training with quantization has been adapted from https://github.com/jiecaoyu/XNOR-Net-PyTorch 
