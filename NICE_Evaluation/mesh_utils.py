import numpy as np
import matplotlib.pyplot as plt
import torch

def calc_reff(weight, base_r, n_bits):
    bin_val = np.binary_repr(weight, width=n_bits)
    # print(weight,bin_val)
    bin_val = (np.array(list(bin_val), dtype=int))
    bin_val = torch.from_numpy((bin_val))
    infty = 1e8
    if n_bits == 1:
        if int(bin_val[0]) == 1:
            Reff = base_r
        else:
            Reff = 200e3
    if n_bits == 4:
        R8, R4, R2, R1 = infty, infty, infty, infty
        if int(bin_val[0]) == 1:
            R8 = base_r / 8.0

        if int(bin_val[1]) == 1:
            R4 = base_r / 4.0
        if int(bin_val[2]) == 1:
            R2 = base_r / 2.0
        if int(bin_val[3]) == 1:
            R1 = base_r

        # if bin_val.sum() > 0:
        Reff = 1 / (1 / R8 + 1 / R4 + 1 / R2 + 1 / R1)
        # elif bin_val.sum() == 0:
        #     Reff = infty
    elif n_bits == 8:
        R128, R64, R32, R16, R8, R4, R2, R1 = infty, infty, infty, infty, infty, infty, infty, infty
        if int(bin_val[0]) == 1:
            R128 = base_r / 128.0

        if int(bin_val[1]) == 1:
            R64 = base_r / 64.0
        if int(bin_val[2]) == 1:
            R32 = base_r / 32.0
        if int(bin_val[3]) == 1:
            R16 = base_r / 16.
        if int(bin_val[4]) == 1:
            R8 = base_r / 8.
        if int(bin_val[5]) == 1:
            R4 = base_r / 4.
        if int(bin_val[6]) == 1:
            R2 = base_r / 2.
        if int(bin_val[7]) == 1:
            R1 = base_r / 1.

        Reff = 1 / (1 / R128 + 1 / R64 + 1 / R32 + 1 / R16 + 1 / R8 + 1 / R4 + 1 / R2 + 1 / R1)
    elif n_bits == 8:
        R128, R64, R32, R16, R8, R4, R2, R1 = infty, infty, infty, infty, infty, infty, infty, infty
        if int(bin_val[0]) == 1:
            R128 = base_r / 128.0

        if int(bin_val[1]) == 1:
            R64 = base_r / 64.0
        if int(bin_val[2]) == 1:
            R32 = base_r / 32.0
        if int(bin_val[3]) == 1:
            R16 = base_r / 16.
        if int(bin_val[4]) == 1:
            R8 = base_r / 8.
        if int(bin_val[5]) == 1:
            R4 = base_r / 4.
        if int(bin_val[6]) == 1:
            R2 = base_r / 2.
        if int(bin_val[7]) == 1:
            R1 = base_r / 1.

        Reff = 1 / (1 / R128 + 1 / R64 + 1 / R32 + 1 / R16 + 1 / R8 + 1 / R4 + 1 / R2 + 1 / R1)


    return Reff


def compute_Reff_column(weight_arr, n_rows, base_r, n_bits):
    Reff_column = torch.zeros(size=(n_rows, 1))
    for idx, i in enumerate(weight_arr):
        Reff_column[idx, :] = calc_reff(weight_arr[idx], base_r, n_bits)
    #         if i == 0:
    #             Reff_column[idx] = calc_reff(,base_r)
    #         else:
    #             Reff_column[idx] = calc_reff(weight_arr[idx], base_r)

    return Reff_column


def compute_current(weight_arr, vin_arr, base_r, n_rows):
    #     R = np.ones(shape=(64))
    R = compute_Reff_column(weight_arr, n_rows, base_r)
    r = 5
    n = n_rows
    #     print(R)
    A = torch.zeros(size=(n, n))
    R = torch.from_numpy(R)
    A[0][0] = -R[0] - R[1] - r
    A[0][1] = R[1]
    for i in range(1, n - 1):
        A[i][i] = -R[i] - r - R[i + 1]
        A[i][i - 1] = R[i]
        A[i][i + 1] = R[i + 1]
    A[n - 1][n - 2] = R[n - 1]
    A[n - 1][n - 1] = -R[n - 1] - r

    B = torch.zeros(n)
    for idx in range(len(vin_arr) - 1):
        if vin_arr[idx] == 0 and vin_arr[idx + 1] == 1:
            B[idx] = 0.1
        elif vin_arr[idx] == 1 and vin_arr[idx + 1] == 0:
            B[idx] = -0.1
    if vin_arr[len(vin_arr) - 1] == 1:
        B[len(vin_arr) - 1] = -0.1
    else:
        B[len(vin_arr) - 1] = 0
    #     print(B)
    #     print(R)
    print(A.size())
    print(torch.solve(A, B))
    return torch.solve(A, B)[-1]  # torch.solve(np.array(A),np.array(B))[-1]


def create_A(weight_arr, base_r, n_rows, n_bits):
    R = compute_Reff_column(weight_arr, n_rows, base_r, n_bits)
    #     print(R)
    r = 0
    n = n_rows
    #     print(R)
    A = torch.zeros(size=(n, n))
    A[0][0] = -R[0] - R[1] - r
    A[0][1] = R[1]
    for i in range(1, n - 1):
        A[i][i] = -R[i] - r - R[i + 1]
        A[i][i - 1] = R[i]
        A[i][i + 1] = R[i + 1]
    A[n - 1][n - 2] = R[n - 1]
    A[n - 1][n - 1] = -R[n - 1] - r
    return A


def create_B(vin_arr, n_rows):
    n = n_rows
    B = torch.zeros(n)
    for idx in range(len(vin_arr) - 1):
        if vin_arr[idx] == 0 and vin_arr[idx + 1] == 1:
            B[idx] = 0.1
        elif vin_arr[idx] == 1 and vin_arr[idx + 1] == 0:
            B[idx] = -0.1
    if vin_arr[len(vin_arr) - 1] == 1:
        B[len(vin_arr) - 1] = -0.1
    else:
        B[len(vin_arr) - 1] = 0

    return B


def compute_current_for_ADC(weight_arr, vin_arr, base_r, n_rows, n_bits):
    #     R = np.ones(shape=(64))
    R = compute_Reff_column(weight_arr, n_rows, base_r, n_bits)
    r = 0
    n = n_rows
    #     print(R)
    A = np.zeros(shape=(n, n))
    A[0][0] = -R[0] - R[1] - r
    A[0][1] = R[1]
    for i in range(1, n - 1):
        A[i][i] = -R[i] - r - R[i + 1]
        A[i][i - 1] = R[i]
        A[i][i + 1] = R[i + 1]
    A[n - 1][n - 2] = R[n - 1]
    A[n - 1][n - 1] = -R[n - 1] - r

    B = np.zeros(shape=(n))
    for idx in range(len(vin_arr) - 1):
        if vin_arr[idx] == 0 and vin_arr[idx + 1] == 1:
            B[idx] = 0.1
        elif vin_arr[idx] == 1 and vin_arr[idx + 1] == 0:
            B[idx] = -0.1
    if vin_arr[len(vin_arr) - 1] == 1:
        B[len(vin_arr) - 1] = -0.1
    else:
        B[len(vin_arr) - 1] = 0
    #     print(B)
    #     print(R)
    return np.linalg.solve(np.array(A), np.array(B))[-1]

