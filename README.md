# SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks
### This repository contains the Pytorch-based evaluation codes for [SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks]. https://arxiv.org/pdf/2210.12899.pdf
 
Currently this repository contains the energy, latency, area (ELA) evaluation code for 4-bit quantized VGG9 SNN model pre-trained on CIFAR10 and implemented on 64x64 4-bit SRAM and 1-bit RRAM crossbars. It also contains the code for quantization-aware SNN training. For reference, we have also provided a pre-trained model path for a 4-bit quantized VGG9 SNN on CIFAR10 dataset. 

TO DO: This Repository is under work and we will be adding the remainder codes for hardware-realistic accuracy evaluation (NICE Engine) and ELA evaluation for other crossbar sizes and technology nodes.

## Ackonwledgements

Code for SNN training with quantization has been adapted from https://github.com/jiecaoyu/XNOR-Net-PyTorch 
