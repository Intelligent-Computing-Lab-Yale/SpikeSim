import numpy as np
import torch
import mesh_utils as mesh

def section_kernel_wts(kernel_ip, xbar_size):
    size = kernel_ip.shape
    #     xbar_grp_size = np.ceil(size[1]/xbar_size)

    kernel_list = []
    kernel_out = torch.zeros(size=(size[2] * size[2], size[1], size[0]))
    k = 0
    for x in range(size[2]):
        for y in range(size[2]):
            asymmetric_xbar = kernel_ip[:, :, x, y]
            #             print(torch.transpose(asymmetric_xbar,0,1).size())
            asymmetric_xbar = torch.transpose(asymmetric_xbar, 0, 1)
            kernel_out[k, :, :] = asymmetric_xbar
            k = k + 1
    #     kernel_array_partitioned = np.asarray(kernel_list)
    shape = kernel_out.size()
    #     print(shape)
    if shape[1] % xbar_size != 0:
        a = shape[1] % xbar_size
        kernel_out = torch.cat((kernel_out, torch.zeros(size=(9, xbar_size - a, shape[2]))), dim=1)

    kernel_out = torch.transpose(kernel_out, 0, 2)
    #     print(kernel_out[0,:,:])
    return kernel_out


def section_kernel_wts_greater(kernel_ip, xbar_size):
    ##### WHEN INPUT FEATURES ARE GREATER THAN THE XBAR SIZE
    # print(f'xbar wts {xbar_size}')
    size = kernel_ip.shape
    #     xbar_grp_size = np.ceil(size[1]/xbar_size)
    no_xbars_ip_channel = int(np.ceil((size[1] / xbar_size)))
    kernel_out = torch.zeros(size=(size[2] * size[2] * no_xbars_ip_channel, xbar_size, size[0]))
    k = 0
    for x in range(size[2]):
        for y in range(size[2]):
            for z in range(no_xbars_ip_channel):
                if z * xbar_size + xbar_size <= size[1]:
                    asymmetric_xbar = kernel_ip[:, z * xbar_size:z * xbar_size + xbar_size, x, y]
                else:
                    asymmetric_xbar = kernel_ip[:, z * xbar_size:size[1], x, y]
                    #                     print(asymmetric_xbar.size(), torch.zeros(size=(size[0],xbar_size-size[1]%xbar_size,1,1)).size())
                    asymmetric_xbar = torch.cat(
                        (asymmetric_xbar, torch.zeros(size=(size[0], xbar_size - size[1] % xbar_size))), dim=1)
                #             print(torch.transpose(asymmetric_xbar,0,1).size())
                asymmetric_xbar = torch.transpose(asymmetric_xbar, 0, 1)
                # print(asymmetric_xbar.size())
                kernel_out[k, :, :] = asymmetric_xbar
                k = k + 1
    #     kernel_array_partitioned = np.asarray(kernel_list)
    shape = kernel_out.size()
    #     print(shape)
    if shape[1] % xbar_size != 0:
        a = shape[1] % xbar_size
        kernel_out = torch.cat((kernel_out, torch.zeros(size=(9, xbar_size - a, shape[2]))), dim=1)

    #     kernel_out = torch.transpose(kernel_out) #, 0,2)
    #     print(kernel_out[0,:,:])
    return kernel_out


##############.  Inputs are 4D Tensors  [batch_size, inchannels, dim, dim]

def section_inputs(inputs, stride, input_size, padding, ip_channel, k, xbar_size):
    pad = torch.nn.ZeroPad2d(padding)
    #     padded_ip = np.pad(ip, ((0,0),(padding,padding),(padding,padding)), 'constant')
    padded_ip = pad(inputs)
    padded_size = padded_ip.size()
    #     print(padded_ip)
    output_size = (np.floor((input_size + 2 * padding - k) / stride) + 1).astype(int)

    #     input_section_out = torch.zeros(size=(padded_size[0], output_size, output_size, padded_size[1], k, k))
    #     for x in range(padded_size[2]-k+1):
    #         for y in range(padded_size[2]-k+1):

    #             section = padded_ip[:,:,x:x+k,y:y+k]
    #             input_section_out[:,x,y,:,:,:] = section

    input_section_out = torch.zeros(size=(output_size, output_size, padded_size[0], padded_size[1], k * k))
    #     input_section_out = torch.zeros(size=(output_size, output_size, padded_size[0], xbar_size, k*k))

    for x in range(padded_size[2] - k + 1):
        for y in range(padded_size[2] - k + 1):
            section = padded_ip[:, :, x:x + k, y:y + k]
            section_size = section.size()
            section_reshape = torch.reshape(section, (section_size[0], section_size[1], k * k))
            #             print(section_reshape.size())
            input_section_out[x, y, :, :, :] = section_reshape

    input_section_out = torch.transpose(input_section_out, 1, 2)
    input_section_out = torch.transpose(input_section_out, 0, 1)

    return input_section_out


def section_inputs_greater(inputs, stride, input_size, padding, ip_channel, k, xbar_size):
    pad = torch.nn.ZeroPad2d(padding)
    #     padded_ip = np.pad(ip, ((0,0),(padding,padding),(padding,padding)), 'constant')
    padded_ip = pad(inputs)
    padded_size = padded_ip.size()
    #     print(padded_ip)
    output_size = (np.floor((input_size + 2 * padding - k) / stride) + 1).astype(int)
    # print(f' padded {padded_ip[:,:,0:3,0:3]}')
    #     input_section_out = torch.zeros(size=(padded_size[0], output_size, output_size, padded_size[1], k, k))
    #     for x in range(padded_size[2]-k+1):
    #         for y in range(padded_size[2]-k+1):

    #             section = padded_ip[:,:,x:x+k,y:y+k]
    #             input_section_out[:,x,y,:,:,:] = section
    no_xbars_ip_channel = int(np.ceil(float(padded_size[1]) / xbar_size))
    #     no_xbars_ip_channel = no_xbars_ip_channel[1]
    input_section_out = torch.zeros(
        size=(output_size, output_size, no_xbars_ip_channel, padded_size[0], xbar_size, k * k))
    #     input_section_out = torch.zeros(size=(output_size, output_size, padded_size[0], xbar_size, k*k))

    for x in range(padded_size[2] - k + 1):
        for y in range(padded_size[2] - k + 1):
            #             print(f' padded_section {padded_ip[:,:,x:x+k,y:y+k]}')
            for z in range(no_xbars_ip_channel):
                if z * xbar_size + xbar_size <= padded_size[1]:
                    section = padded_ip[:, z * xbar_size:z * xbar_size + xbar_size, x:x + k, y:y + k]
                #                     print(f' section {section}')
                else:
                    section = padded_ip[:, z * xbar_size:padded_size[1], x:x + k, y:y + k]
                    sec_size = section.size()
                    #                     print(sec_size)

                    section = torch.cat(
                        (section, torch.zeros(sec_size[0], (xbar_size - sec_size[1]), sec_size[2], sec_size[3]).int()),
                        dim=1)

                section_size = section.size()
                #                 print(section_size)
                section_reshape = torch.reshape(section, (section_size[0], section_size[1], k * k))
                #                 print(section_reshape.size())
                #                 print(z)
                #                 print(section_reshape.size())
                input_section_out[x, y, z, :, :, :] = section_reshape

    input_section_out = torch.transpose(input_section_out, 2, 3)
    input_section_out = torch.transpose(input_section_out, 1, 2)
    input_section_out = torch.transpose(input_section_out, 0, 1)

    return input_section_out


def section_inputs_greater1(inputs, stride, input_size, padding, ip_channel, k, xbar_size):
    # print(f'xbar ip {xbar_size}')

    pad = torch.nn.ZeroPad2d(padding)
    padded_ip = pad(inputs)
    #     padded_ip = inputs
    padded_size = padded_ip.size()
    output_size = (np.floor((input_size + 2 * padding - k) / stride) + 1).astype(int)
    no_xbars_ip_channel = int(np.ceil(float(padded_size[1]) / xbar_size))
    input_section_out = torch.zeros(
        size=(output_size, output_size, no_xbars_ip_channel * k * k, padded_size[0], xbar_size))

    for x in range(padded_size[2] - k + 1):
        for y in range(padded_size[2] - k + 1):
            section = padded_ip[:, :, x:x + k, y:y + k]
            size_sec = section.size()
            kl = 0
            for kx in range(size_sec[2]):
                for ky in range(size_sec[2]):
                    for z in range(no_xbars_ip_channel):
                        if z * xbar_size + xbar_size <= size_sec[1]:
                            asymmetric_xbar = section[:, z * xbar_size:z * xbar_size + xbar_size, kx, ky]
                        else:
                            asymmetric_xbar = section[:, z * xbar_size:size_sec[1], kx, ky]
                            asymmetric_xbar = torch.cat(
                                (asymmetric_xbar, torch.zeros(size=(size_sec[0], xbar_size - size_sec[1] % xbar_size))),
                                dim=1)
                        #                         asymmetric_xbar = torch.transpose(asymmetric_xbar,0,1)
                        input_section_out[x, y, kl, :, :] = asymmetric_xbar
                        kl = kl + 1

    input_section_out = torch.transpose(input_section_out, 2, 3)
    input_section_out = torch.transpose(input_section_out, 1, 2)
    input_section_out = torch.transpose(input_section_out, 0, 1)

    return input_section_out

def create_A_arr(sec_wts, xbar_size, base_r, n_bits):
    sec_wts1 = torch.transpose(sec_wts, 1, 2)
    sec_wts1 = torch.transpose(sec_wts1, 0, 1)
    wt_size = sec_wts1.size()
    # print(wt_size)
    A_arr = torch.zeros(size=(wt_size[0], wt_size[1], wt_size[2], wt_size[2]))
    for i in range(wt_size[0]):
        for j in range(wt_size[1]):
            #         print(sec_wts[i,j,:].size())
            A_arr[i, j, :, :] = mesh.create_A(weight_arr=sec_wts1[i, j, :].int(), n_rows=xbar_size, base_r=base_r, n_bits=n_bits)
            ### Calculate the negative weights here
    # print(A_arr.size())

    return A_arr

def create_cat_ip(sec_inp):
    sec_size = sec_inp.size()
    # print(sec_size)
    sec_inp_reshape = torch.reshape(sec_inp, shape=(sec_size[0]*sec_size[1] * sec_size[2], sec_size[3], sec_size[4]))
    # print(sec_inp_reshape.size())
    cat_inputs = sec_inp_reshape

    return cat_inputs

def create_cat_ip1(sec_inp):
    sec_size = sec_inp.size()
    # print('Hi')
    sec_inp_reshape = torch.reshape(sec_inp, shape=(sec_size[0], sec_size[1] * sec_size[2], sec_size[3], sec_size[4]))
    # print(sec_inp_reshape.size())
    cat_inputs = sec_inp_reshape

    return cat_inputs

def create_A_mat_unsigned(weights, n_bits, xbar_size, base_r):
    # n_bits = 4
    # xbar_size = 64
    # batch_s = 1
    # base_r = 6.25e3  # 19.19e3

    # W_temp = torch.round(weights * 2 ** n_bits)
    # mask = (W_temp < 0) * 1
    # mask = mask.float()
    # W_bits = W_temp + (mask * 16)
    # neg_W = W_bits >= 2 ** (n_bits - 1)
    sec_wts = section_kernel_wts_greater(weights.cpu(), xbar_size)

    A_arr = create_A_arr(sec_wts, xbar_size, base_r)
    A_size = A_arr.size()
    A = torch.reshape(A_arr, shape=(A_size[0] * A_size[1], A_size[2], A_size[3]))
    # A = A.repeat()
    return A

def create_A_mat1(weights, n_bits, xbar_size, base_r, b_size):
    # n_bits = 4
    # xbar_size = 64
    # batch_s = 1
    # base_r = 6.25e3  # 19.19e3
    # weights_u = weights.abs()
    # W_bits_u = torch.round(weights_u * 2 ** (n_bits-1))
    # W_bits = torch.round(weights * 2 ** (n_bits-1))

    wt_uq = torch.round(weights * (2 ** (n_bits-1)))
    mask_u = wt_uq < 0
    mask_u = mask_u * (2**(n_bits-2))
    # print(type(mask_u), type(wt_uq))
    wt_uq = mask_u + wt_uq
    # print(weights[0,:,:,:], W_bits_u[0,:,:,:], W_bits[0,:,:,:])
    # W_bits = weights * 2 ** (n_bits-1)
    # mask = (W_temp < 0) * 1
    # mask = mask.float()
    # W_bits = W_temp + (mask * 16)
    wt_q = torch.round(weights * (2 ** (n_bits - 1)))
    neg_W = wt_q < 0
    sec_wts = section_kernel_wts_greater(wt_uq.cpu(), xbar_size)

    A_arr = create_A_arr(sec_wts, xbar_size, base_r, n_bits)
    A_size = A_arr.size()
    A = torch.reshape(A_arr, shape=(A_size[0] * A_size[1], A_size[2], A_size[3]))
    A = A.repeat(b_size, 1,1)
    return neg_W, A, wt_uq

def create_A_mat_mod(weights, n_bits, neg_wt_bits, xbar_size, base_r, b_size):
    # n_bits = 4
    # xbar_size = 64
    # batch_s = 1
    # base_r = 6.25e3  # 19.19e3
    # weights_u = weights.abs()
    # W_bits_u = torch.round(weights_u * 2 ** (n_bits-1))
    # W_bits = torch.round(weights * 2 ** (n_bits-1))

    wt_uq = torch.round(weights * (2 ** (n_bits-1)))
    wt_raw_q = wt_uq
    mask_u = wt_uq < 0
    mask_u = mask_u * (2**(neg_wt_bits))
    # print(type(mask_u), type(wt_uq))
    wt_uq = mask_u + wt_uq
    # print(weights[0,:,:,:], W_bits_u[0,:,:,:], W_bits[0,:,:,:])
    # W_bits = weights * 2 ** (n_bits-1)
    # mask = (W_temp < 0) * 1
    # mask = mask.float()
    # W_bits = W_temp + (mask * 16)
    wt_q = torch.round(weights * (2 ** (n_bits - 1)))
    neg_W = wt_q < 0
    # neg_W = weights < 0
    sec_wts = section_kernel_wts_greater(wt_uq.cpu(), xbar_size)

    A_arr = create_A_arr(sec_wts, xbar_size, base_r, n_bits)
    A_size = A_arr.size()
    A = torch.reshape(A_arr, shape=(A_size[0] * A_size[1], A_size[2], A_size[3]))
    A = A.repeat(b_size, 1,1)
    return neg_W, A, wt_uq, wt_raw_q

def create_B_in(spike_inp, xbar_size, kernel_size, A_size0):
    ip_size = spike_inp.size()
    sec_inp = section_inputs_greater1(spike_inp.cpu(), stride=1, input_size=ip_size[2], padding=1,
                                              ip_channel=ip_size[1], k=kernel_size, xbar_size=xbar_size)
    cat_inputs = create_cat_ip(sec_inp)
    B_size = cat_inputs.size()

    B = torch.transpose(cat_inputs, 0, 1)
    B = torch.transpose(B, 1, 2)
    B_rep = B.repeat(A_size0, 1, 1, 1)
    # print(B_rep.size())
    B_rep_size = B_rep.size()
    B_rep = torch.reshape(B_rep, shape=(B_rep_size[0] * B_rep_size[1], B_rep_size[2], B_rep_size[3]))
    # print(f'Brep_size {B_rep.size()}')
    B_in_size = B_rep.size()
    temp = B_rep[:, 0:B_in_size[1] - 1, :] - B_rep[:, 1:B_in_size[1], :]
    # print(temp.size())
    B_rep[:, 0:B_in_size[1] - 1, :] = temp
    B_in = B_rep * -0.1

    return B_in, B_rep_size

def create_B_in1(spike_inp, xbar_size, kernel_size, wt_size):
    ip_size = spike_inp.size()
    sec_inp = section_inputs_greater1(spike_inp.cpu(), stride=1, input_size=ip_size[2], padding=1,
                                              ip_channel=ip_size[1], k=kernel_size, xbar_size=xbar_size)
    cat_inputs = create_cat_ip1(sec_inp)
    B_size = cat_inputs.size()

    B = torch.transpose(cat_inputs, 1, 2)
    # print(f'B1 {B.size()}')
    B = torch.transpose(B, 2, 3)
    # print(f'B2 {B.size()}')
    B_rep = B.repeat(1, wt_size[0], 1, 1)
    # print(f'Brep {B_rep.size()}')
    # cat_t = torch.tensor([])
    # for i in range(B.size(0)):
    #     t = B[i,:,:,:].repeat(4,1,1,1)
    #     cat_t = torch.cat((cat_t, t))
    # print(cat_t.size())
    # print(B_rep.size())
    B_rep_size = B_rep.size()
    B_rep = torch.reshape(B_rep, shape=(B_rep_size[0] * B_rep_size[1], B_rep_size[2], B_rep_size[3]))
    # print(f'Brep_size {B_rep.size()}')
    B_in_size = B_rep.size()
    temp = B_rep[:, 0:B_in_size[1] - 1, :] - B_rep[:, 1:B_in_size[1], :]
    # print(temp.size())
    B_rep[:, 0:B_in_size[1] - 1, :] = temp
    B_in = B_rep * -0.1

    return B_in, B_rep_size
def create_B_in_unsigned(spike_inp, xbar_size, kernel_size, wt_size, padding):
    ip_size = spike_inp.size()
    sec_inp = section_inputs_greater1(spike_inp.cpu(), stride=1, input_size=ip_size[2], padding=padding,
                                              ip_channel=ip_size[1], k=kernel_size, xbar_size=xbar_size)
    cat_inputs = create_cat_ip1(sec_inp)
    B_size = cat_inputs.size()

    B = torch.transpose(cat_inputs, 1, 2)
    # print(f'B1 {B.size()}')
    B = torch.transpose(B, 2, 3)
    # print(f'B2 {B.size()}')
    B_rep = B.repeat(1, wt_size[0], 1, 1)
    # print(f'Brep {B_rep.size()}')
    # cat_t = torch.tensor([])
    # for i in range(B.size(0)):
    #     t = B[i,:,:,:].repeat(4,1,1,1)
    #     cat_t = torch.cat((cat_t, t))
    # print(cat_t.size())
    # print(B_rep.size())
    B_rep_size = B_rep.size()
    B_rep = torch.reshape(B_rep, shape=(B_rep_size[0] * B_rep_size[1], B_rep_size[2], B_rep_size[3]))
    # print(f'Brep_size {B_rep.size()}')
    B_in_size = B_rep.size()
    temp = B_rep[:, 0:B_in_size[1] - 1, :] - B_rep[:, 1:B_in_size[1], :]
    # print(temp.size())
    B_rep[:, 0:B_in_size[1] - 1, :] = temp
    B_in = B_rep * -0.1

    return B_in, B_rep_size
# wt_size=wt.size()
# B = torch.transpose(cat_inputs, 1, 2)
# print(f'B1 {B.size()}')
# B = torch.transpose(B, 2, 3)
# print(f'B2 {B.size()}')
# B_rep = B.repeat(1, wt_size[0], 1, 1)
# # print(f'Brep {B_rep.size()}')
# # cat_t = torch.tensor([])
# # for i in range(B.size(0)):
# #     t = B[i,:,:,:].repeat(4,1,1,1)
# #     cat_t = torch.cat((cat_t, t))
# # print(cat_t.size())
# print(B_rep.size())
# B_rep_size = B_rep.size()
# B_rep = torch.reshape(B_rep, shape=(B_rep_size[0] * B_rep_size[1], B_rep_size[2], B_rep_size[3]))
# print(f'Brep2 {B_rep.size()}')