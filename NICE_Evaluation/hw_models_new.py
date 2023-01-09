import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import numpy as np
import math
import random
import numpy.linalg as LA
from torch.autograd import Variable
import torch.nn.functional as F
import mesh_utils as mesh
import section_ip_wts as section

class Surrogate_BP_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), (torch.sign(inp)+1)*0.5)



class SNN_VGG5_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=28, num_cls=10, bntt_flag=False):
        super(SNN_VGG5_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.bntt_flag = bntt_flag

        print(">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print("***** time step per batchnorm".format(self.batch_num))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList(
            [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList(
            [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear((self.img_size // 4) * (self.img_size // 4) * 128, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList(
            [nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt_fc]
        self.pool_list = [self.pool1, False, self.pool2]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(self.num_steps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                if self.bntt_flag is True:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](
                        self.conv_list[i](out_prev))
                else:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + (self.conv_list[i](out_prev))

                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()

                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            if self.bntt_flag is True:
                mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            else:
                mem_fc1 = self.leak_mem * mem_fc1 + (self.fc1(out_prev))

            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps

        return out_voltage

def quantize(value, n_bits):
    x = value
    v0 = 1
    v1 = 2
    v2 = -0.5
    y = 2 ** n_bits - 1
    # print(y)
    x = x.add(v0).div(v1)
    # print(x)
    x = x.mul(y).round_()
    x = x.div(y)
    x = x.add(v2)
    x = x.mul(v1)

    return x

def signed_hw_conv_mod(A, B, ideal_current, ADC_prec, neg_W, spike_inp, unfold_size, neg_wt_bits, weight_size, factor):
    # print(B.size(), A.size())
    sol = torch.solve(B, A)
    pc_size = sol[0][:, -1, :].size()
    ip_size = spike_inp.size()
    # print('hellow')
    # print(pc_size)
    p_currents = torch.reshape(sol[0][:, -1, :], shape=(ip_size[0], weight_size[0], -1, pc_size[1]))
    # p_currents = float((2**4)-1) /  7.841899945321117e-05 * p_currents
    # p_currents = float((2 ** 5) - 1) / 0.0001672938655001829 * p_currents
    p_currents = float((2 ** ADC_prec) - 1) / ideal_current * p_currents
    # p_currents = torch.clamp(p_currents, 0, 64)
    # p_currents = quantize(p_currents, 5)
    # p_currents = torch.round(p_currents * (2 ** 4 - 1))
    # p_currents = p_currents / (2 ** 4 - 1)
    # print(p_currents)


    # print(p_currents.size())
    conv_h = p_currents.sum(dim=2)
    convh_size = conv_h.size()
    # print(conv_h.size())
    # conv_h = torch.reshape(conv_h, shape=(convh_size[0], spike_inp.size(0), -1))
    # conv_h = torch.transpose(conv_h, 0, 1)

    ############################################################################
    conv_func_negs = torch.nn.Conv2d(weight_size[1], weight_size[0], weight_size[2], stride=1, padding=1, bias=False).cuda()
    neg_W = neg_W.float().cuda()
    conv_func_negs.weight = torch.nn.Parameter(neg_W)
    # print(spike_inp.size(), neg_W.size())

    number_negs = conv_func_negs(spike_inp.float())
    # print(number_negs.size())
    number_negs = torch.reshape(number_negs, shape=(ip_size[0], weight_size[0], spike_inp.size(2),spike_inp.size(2)))
    # print(number_negs.size())
    conv_h = torch.reshape(conv_h, shape=(ip_size[0], weight_size[0], spike_inp.size(2), spike_inp.size(2)))
    current_digital = float((2**ADC_prec)-1) /  ideal_current * conv_h  # 1250e-6 * conv_h  #510e-6 4080e-6
    # print(f'current {current_digital.size()}, number_negs {number_negs.size()}')

    current_digital = conv_h
    sub_after_shift = current_digital - (2 ** (neg_wt_bits)) * number_negs
    # sub_after_shift = sub_after_shift / (2**factor)

    return sub_after_shift, current_digital


def hw_conv(A, conv_layer, frac_bit, batch_size, b_size, out_prev, xbar_size,
            ideal_current, ADC_precision, neg_W, neg_bits, batch, t, layer):
    # A = A_list[i].to("cuda")
    # weight_size = self.conv_list[i].weight.data.size()

    weight_size = conv_layer.weight.data.size()

    # print(weight_size)
    # print(A.size())
    # frac_bits = -1

    factor = 2 ** frac_bit
    # print(f' conv {i} factor {factor}')
    conv_out = torch.tensor([]).cuda()

    for ba in range(int(batch_size / b_size)):
        # print(ba)

        # if batch == 0 and t == 0:

        B_in, B_rep_size = section.create_B_in1(out_prev[ba * b_size:ba * b_size + b_size, :, :, :], xbar_size,
                                                weight_size[3], weight_size)
        B_in = B_in.to("cuda")
        # sub_after_shift, current_digital = models.signed_hw_conv_mod(A.cuda(), B_in.cuda(), ideal_current,
        #                                                              ADC_precision, neg.cuda(), a.cuda(),
        #                                                              B_rep_size, neg_wt_bits, wt_size,
        #                                                              1)  ## 1332e-6

        # if t == 0 and batch == 0 and ba == 0:
        #     print('saving')
        sub_after_shift, current_digital = signed_hw_conv_mod(A, B_in, ideal_current,
                                                              ADC_precision, neg_W,
                                                              out_prev[ba * b_size:ba * b_size + b_size, :, :, :],
                                                              B_rep_size, neg_bits, weight_size,
                                                              1)  ## 1332e-6

        # print(f' sub_after_shift {sub_after_shift.size()}, current_digital {current_digital.size()}')
        # print('hi')
        sub_after_shift = sub_after_shift / factor
        # print('hello')
        conv_out = torch.cat((conv_out, sub_after_shift))

        if batch == 0 and ba == 0:
            # print(f'saving {i} size {sub_after_shift.size()}')
            torch.save(sub_after_shift, './hw_outs_new/conv' + str(layer) + '_output_hw' + str(t))
            torch.save(conv_layer(out_prev[ba * b_size:ba * b_size + b_size, :, :, :]),
                       './hw_outs_new/conv' + str(layer) + '_out_sw' + str(t))

    return conv_out

class SNN_VGG9_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32, num_cls=10, bntt_flag=False):
        super(SNN_VGG9_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.bntt_flag = bntt_flag

        print(">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print("***** time step per batchnorm".format(self.batch_num))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList(
            [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList(
            [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList(
            [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList(
            [nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList(
            [nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList(
            [nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear((self.img_size // 8) * (self.img_size // 8) * 256, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList(
            [nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7,
                          self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp, batch, A_list, neg_W_list, xbar_size, n_bits, neg_bits, b_size):

        batch_size = inp.size(0)
        ideal_current = 0.0013331229907048428 #0.00012750000000000242  # 0.0013323579954395343 #0.00021250000000000929 #0.0012749999999998585 #0.0013323579954395343  #4080e-6 #0.0013323579954395343
        ADC_precision = 8
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(self.num_steps):

            # addDeviceVariation_SA(wts, dev_param)

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp
            # frac_bits = [4,4,4,3,3,3,3] #[0, 0, 0, 0, 1, 1, 1]
            frac_bits = [3,3,3,3,3,3,3]
            for i in range(len(self.conv_list)):
                conv_out = hw_conv(A_list[i].cuda(), self.conv_list[i], frac_bits[i], batch_size, b_size, out_prev, xbar_size,
                        ideal_current, ADC_precision, neg_W_list[i].cuda(), neg_bits[i], batch, t, i)

            # for i in range(len(self.conv_list)):
            #     A = A_list[i].to("cuda")
            #     weight_size = self.conv_list[i].weight.data.size()
            #     # print(weight_size)
            #     # print(A.size())
            #     # frac_bits = -1
            #
            #     factor = 2 ** frac_bits[i]
            #     # print(f' conv {i} factor {factor}')
            #     conv_out = torch.tensor([]).cuda()
            #
            #     for ba in range(int(batch_size/b_size)):
            #         # print(ba)
            #
            #         # if batch == 0 and t == 0:
            #
            #         B_in, B_rep_size = section.create_B_in1(out_prev[ba*b_size:ba*b_size+b_size,:,:,:], xbar_size, weight_size[3], weight_size)
            #         B_in = B_in.to("cuda")
            #         # sub_after_shift, current_digital = models.signed_hw_conv_mod(A.cuda(), B_in.cuda(), ideal_current,
            #         #                                                              ADC_precision, neg.cuda(), a.cuda(),
            #         #                                                              B_rep_size, neg_wt_bits, wt_size,
            #         #                                                              1)  ## 1332e-6
            #
            #         # if t == 0 and batch == 0 and ba == 0:
            #         #     print('saving')
            #         sub_after_shift, current_digital = signed_hw_conv_mod(A, B_in, ideal_current,
            #                                                                      ADC_precision, neg_W_list[i].cuda(), out_prev[ba*b_size:ba*b_size+b_size,:,:,:],
            #                                                                      B_rep_size, neg_bits[i], weight_size,
            #                                                                      1)  ## 1332e-6
            #
            #         # print(f' sub_after_shift {sub_after_shift.size()}, current_digital {current_digital.size()}')
            #         # print('hi')
            #         sub_after_shift = sub_after_shift/factor
            #         # print('hello')
            #         conv_out = torch.cat((conv_out, sub_after_shift))
            #         if batch == 0 and ba == 0:
            #             # print(f'saving {i} size {sub_after_shift.size()}')
            #             torch.save(sub_after_shift, './hw_outs_new/conv'+str(i)+'_output_hw' + str(t))
            #             torch.save(self.conv_list[i](out_prev[ba*b_size:ba*b_size+b_size,:,:,:]), './hw_outs_new/conv'+str(i)+'_out_sw' + str(t))
                if self.bntt_flag is True:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](conv_out)
                else:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + (self.conv_list[i](out_prev))

                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()

                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            if self.bntt_flag is True:
                mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            else:
                mem_fc1 = self.leak_mem * mem_fc1 + (self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps

        return out_voltage







class VGG9_Direct(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32, num_cls=10, input_dim=3):
        super(VGG9_Direct, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.arch = "SNN"

        print(">>>>>>>>>>>>>>>>>>> VGG 9_Direct Coding >>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        # Test
        # self.drop1 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear((self.img_size // 8) * (self.img_size // 8) * 256, 1024, bias=bias_flag)
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7,
        ]

        self.pool_list = [
            False,
            self.pool1,
            False,
            self.pool2,
            False,
            False,
            self.pool3,
        ]

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)

    def forward(self, inp, batch, A_list, neg_W_list, xbar_size, n_bits, neg_bits, b_size, ADC_precision):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(
            batch_size, 128, self.img_size // 2, self.img_size // 2
        ).cuda()
        mem_conv4 = torch.zeros(
            batch_size, 128, self.img_size // 2, self.img_size // 2
        ).cuda()
        mem_conv5 = torch.zeros(
            batch_size, 256, self.img_size // 4, self.img_size // 4
        ).cuda()
        mem_conv6 = torch.zeros(
            batch_size, 256, self.img_size // 4, self.img_size // 4
        ).cuda()
        mem_conv7 = torch.zeros(
            batch_size, 256, self.img_size // 4, self.img_size // 4
        ).cuda()
        mem_conv_list = [
            mem_conv1,
            mem_conv2,
            mem_conv3,
            mem_conv4,
            mem_conv5,
            mem_conv6,
            mem_conv7,
        ]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        # Direct coding - static input from conv1

        static_input = self.conv1(inp)
        batch_size = inp.size(0)

        # ADC_precision = 4
        if ADC_precision == 8:
            ideal_current = 0.0013323580124351578
        elif ADC_precision == 4:
            ideal_current = 7.837400073149083e-05 #0.0013331229907048428  # 0.00012750000000000242  # 0.0013323579954395343 #0.00021250000000000929 #0.0012749999999998585 #0.0013323579954395343  #4080e-6 #0.0013323579954395343

        frac_bits = [3,3,3,3,3,3,3]
        # static_input = hw_conv(A_list[0].cuda(), self.conv_list[0], frac_bits[0], batch_size, b_size, inp, xbar_size,
        #                 ideal_current, ADC_precision, neg_W_list[i].cuda(), neg_bits[i], batch, t, i)
        for t in range(self.num_steps):
            # Charging and firing (lif for conv1)
            mem_conv_list[0] = (1 - self.leak_mem) * mem_conv_list[0] + self.leak_mem * static_input
            mem_thr = (mem_conv_list[0] / self.conv_list[0].threshold) - 1.0
            out = self.spike_fn(mem_thr)

            # Soft reset
            rst = torch.zeros_like(mem_conv_list[0]).cuda()
            rst[mem_thr > 0] = self.conv_list[0].threshold
            mem_conv_list[0] = mem_conv_list[0] - rst
            out_prev = out.clone()

            for i in range(1, len(self.conv_list)):
                mem_conv_list[i] = hw_conv(A_list[i].cuda(), self.conv_list[i], frac_bits[i], batch_size, b_size, out_prev, xbar_size,
                        ideal_current, ADC_precision, neg_W_list[i].cuda(), neg_bits[i], batch, t, i)

                # mem_conv_list[i] = (1 - self.leak_mem) * mem_conv_list[i] + self.leak_mem * self.conv_list[i](out_prev)
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()
                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            # Test
            # out = self.drop1(out_prev)
            # out_prev = out.clone()

            mem_fc1 = (1 - self.leak_mem) * mem_fc1 + self.leak_mem * self.fc1(out_prev)

            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps

        return out_voltage


class SNN_VGG11_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32, num_cls=10):
        super(SNN_VGG11_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps

        print(">>>>>>>>>>>>>>>>> VGG11 >>>>>>>>>>>>>>>>>>>>>>>")
        print("***** time step per batchnorm".format(self.batch_num))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList(
            [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList(
            [nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList(
            [nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList(
            [nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList(
            [nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList(
            [nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt8 = nn.ModuleList(
            [nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 4096, bias=bias_flag)
        self.bntt_fc = nn.ModuleList(
            [nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(4096, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7,
                          self.bntt8, self.bntt_fc]
        self.pool_list = [self.pool1, self.pool2, False, self.pool3, False, self.pool4, False, self.pool5]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv3 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv5 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
        mem_conv6 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
        mem_conv7 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv8 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7, mem_conv8]

        mem_fc1 = torch.zeros(batch_size, 4096).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(self.num_steps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()

                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps

        return out_voltage