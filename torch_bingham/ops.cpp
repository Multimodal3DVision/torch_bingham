/*
   torch_bingham module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Haowen Deng
*/

#include <torch/extension.h>
#include <iostream>
#include <memory>

#include "bingham.h"
#include "bingham/bingham_constants.h"
#include "bingham/bingham_constant_tables.h"

torch::Tensor F_lookup_3d(torch::Tensor z)
{
    auto z_cpu = z.to(torch::device(torch::kCPU));
    auto acc_z = z_cpu.accessor<float, 2>();
    auto o = torch::zeros({acc_z.size(0)});
    auto acc_o = o.accessor<float, 1>();
    double *Z(new double[3]);
    for (auto i = 0; i < acc_z.size(0); i++)
    {
        Z[0] = static_cast<double>(acc_z[i][0]);
        Z[1] = static_cast<double>(acc_z[i][1]);
        Z[2] = static_cast<double>(acc_z[i][2]);

        double y0 = sqrt(-Z[0]);
        double y1 = sqrt(-Z[1]);
        double y2 = sqrt(-Z[2]);

        int n = BINGHAM_TABLE_LENGTH;
        double ymin = bingham_table_range[0];
        double ymax = bingham_table_range[n - 1];

        // get the Z-table cell between (i0,i1), (j0,j1), and (k0,k1)

        int i0, j0, k0, i1, j1, k1;

        if (y0 <= ymin)
            i1 = 1;
        else if (y0 >= ymax)
            i1 = n - 1;
        else
            i1 = binary_search(y0, (double *)bingham_table_range, n);
        i0 = i1 - 1;

        if (y1 <= ymin)
            j1 = 1;
        else if (y1 >= ymax)
            j1 = n - 1;
        else
            j1 = binary_search(y1, (double *)bingham_table_range, n);
        j0 = j1 - 1;

        if (y2 <= ymin)
            k1 = 1;
        else if (y2 >= ymax)
            k1 = n - 1;
        else
            k1 = binary_search(y2, (double *)bingham_table_range, n);
        k0 = k1 - 1;

        // use trilinear interpolation between the 8 corners
        y0 -= bingham_table_range[i0];
        y1 -= bingham_table_range[j0];
        y2 -= bingham_table_range[k0];

        double d0 = bingham_table_range[i1] - bingham_table_range[i0];
        double d1 = bingham_table_range[j1] - bingham_table_range[j0];
        double d2 = bingham_table_range[k1] - bingham_table_range[k0];

        double F000 = bingham_F_table_get(i0, j0, k0);
        double F001 = bingham_F_table_get(i0, j0, k1);
        double F010 = bingham_F_table_get(i0, j1, k0);
        double F011 = bingham_F_table_get(i0, j1, k1);
        double F100 = bingham_F_table_get(i1, j0, k0);
        double F101 = bingham_F_table_get(i1, j0, k1);
        double F110 = bingham_F_table_get(i1, j1, k0);
        double F111 = bingham_F_table_get(i1, j1, k1);

        // interpolate over k
        double F00 = F000 + y2 * (F001 - F000) / d2;
        double F01 = F010 + y2 * (F011 - F010) / d2;
        double F10 = F100 + y2 * (F101 - F100) / d2;
        double F11 = F110 + y2 * (F111 - F110) / d2;

        // interpolate over j
        double F0 = F00 + y1 * (F01 - F00) / d1;
        double F1 = F10 + y1 * (F11 - F10) / d1;

        // interpolate over i
        double F = F0 + y0 * (F1 - F0) / d0;

        acc_o[i] = static_cast<float>(F);
    }
    delete[] Z;
    o = o.to(z.device(), z.dtype());
    return o;
}

torch::Tensor dF_lookup_3d(torch::Tensor z)
{
    auto z_cpu = z.to(torch::device(torch::kCPU));
    auto acc_z = z_cpu.accessor<float, 2>();
    auto o = torch::zeros({acc_z.size(0), 3});
    auto acc_o = o.accessor<float, 2>();
    double *Z(new double[3]);
    for (auto i = 0; i < acc_z.size(0); i++)
    {
        Z[0] = static_cast<double>(acc_z[i][0]);
        Z[1] = static_cast<double>(acc_z[i][1]);
        Z[2] = static_cast<double>(acc_z[i][2]);

        double y0 = sqrt(-Z[0]);
        double y1 = sqrt(-Z[1]);
        double y2 = sqrt(-Z[2]);

        int n = BINGHAM_TABLE_LENGTH;
        double ymin = bingham_table_range[0];
        double ymax = bingham_table_range[n - 1];

        // get the Z-table cell between (i0,i1), (j0,j1), and (k0,k1)

        int i0, j0, k0, i1, j1, k1;

        if (y0 <= ymin)
            i1 = 1;
        else if (y0 >= ymax)
            i1 = n - 1;
        else
            i1 = binary_search(y0, (double *)bingham_table_range, n);
        i0 = i1 - 1;

        if (y1 <= ymin)
            j1 = 1;
        else if (y1 >= ymax)
            j1 = n - 1;
        else
            j1 = binary_search(y1, (double *)bingham_table_range, n);
        j0 = j1 - 1;

        if (y2 <= ymin)
            k1 = 1;
        else if (y2 >= ymax)
            k1 = n - 1;
        else
            k1 = binary_search(y2, (double *)bingham_table_range, n);
        k0 = k1 - 1;

        // use trilinear interpolation between the 8 corners
        y0 -= bingham_table_range[i0];
        y1 -= bingham_table_range[j0];
        y2 -= bingham_table_range[k0];

        double d0 = bingham_table_range[i1] - bingham_table_range[i0];
        double d1 = bingham_table_range[j1] - bingham_table_range[j0];
        double d2 = bingham_table_range[k1] - bingham_table_range[k0];

        float df[3];
        int a;
        // note the order here is descending !!! z0 > z1 > z2
        for (a = 2; a >= 0; a--)
        {
            double dF000 = bingham_dF_table_get(a, i0, j0, k0);
            double dF001 = bingham_dF_table_get(a, i0, j0, k1);
            double dF010 = bingham_dF_table_get(a, i0, j1, k0);
            double dF011 = bingham_dF_table_get(a, i0, j1, k1);
            double dF100 = bingham_dF_table_get(a, i1, j0, k0);
            double dF101 = bingham_dF_table_get(a, i1, j0, k1);
            double dF110 = bingham_dF_table_get(a, i1, j1, k0);
            double dF111 = bingham_dF_table_get(a, i1, j1, k1);

            // interpolate over k
            double dF00 = dF000 + y2 * (dF001 - dF000) / d2;
            double dF01 = dF010 + y2 * (dF011 - dF010) / d2;
            double dF10 = dF100 + y2 * (dF101 - dF100) / d2;
            double dF11 = dF110 + y2 * (dF111 - dF110) / d2;

            // interpolate over j
            double dF0 = dF00 + y1 * (dF01 - dF00) / d1;
            double dF1 = dF10 + y1 * (dF11 - dF10) / d1;

            // interpolate over i
            double dF = dF0 + y0 * (dF1 - dF0) / d0;
            acc_o[i][2 - a] = static_cast<float>(dF);
        }
    }
    delete[] Z;
    o = o.to(z.device(), z.dtype());
    return o;
}

torch::Tensor bingham_prob(torch::Tensor q, torch::Tensor l, torch::Tensor q_gt)
{
    /*
    q:  batch_size \times 4, the quaternions predicted by the network
    l:  batch_size \times 3, the lambda values predicted by the network
    q_gt batch_size \times 4, the ground truth quaternionsi

    The transposed version of V(q) is:
    V(q)' = {
        q0, q1, q2, q3;
        -q1, q0, -q3, q2;
        -q2, q3, q0, -q1;
        q3, q2, -q1, -q0
    }

    each row would take dot product with the ground truth quatenion.
    Since the first value of lambda is always 0, so we don't have to calculate it for the first row

    */
    auto l_cpu = l.to(torch::device(torch::kCPU));
    auto l_acc = l_cpu.accessor<float, 2>();

    torch::Tensor range_tensor(torch::zeros({l_acc.size(0), 3}));
    torch::Tensor d_tensor(torch::zeros({l_acc.size(0), 3}));
    torch::Tensor F_tensor(torch::zeros({l_acc.size(0), 8}));

    auto range_tensor_acc = range_tensor.accessor<float, 2>();
    auto d_tensor_acc = d_tensor.accessor<float, 2>();
    auto F_tensor_acc = F_tensor.accessor<float, 2>();

    auto gt0 = torch::narrow(q_gt, 1, 0, 1);
    auto gt1 = torch::narrow(q_gt, 1, 1, 1);
    auto gt2 = torch::narrow(q_gt, 1, 2, 1);
    auto gt3 = torch::narrow(q_gt, 1, 3, 1);

    auto q0 = torch::narrow(q, 1, 0, 1);
    auto q1 = torch::narrow(q, 1, 1, 1);
    auto q2 = torch::narrow(q, 1, 2, 1);
    auto q3 = torch::narrow(q, 1, 3, 1);

    auto a1 = -gt0 * q1 + gt1 * q0 - gt2 * q3 + gt3 * q2;
    auto a2 = -gt0 * q2 + gt1 * q3 + gt2 * q0 - gt3 * q1;
    auto a3 = gt0 * q3 + gt1 * q2 - gt2 * q1 - gt3 * q0;

    for (auto i = 0; i < l_acc.size(0); i++)
    {
        double y0 = sqrt(-l_acc[i][0]);
        double y1 = sqrt(-l_acc[i][1]);
        double y2 = sqrt(-l_acc[i][2]);

        int n = BINGHAM_TABLE_LENGTH;
        double ymin = bingham_table_range[0];
        double ymax = bingham_table_range[n - 1];

        // get the Z-table cell between (i0,i1), (j0,j1), and (k0,k1)

        int i0, j0, k0, i1, j1, k1;

        if (y0 <= ymin)
            i1 = 1;
        else if (y0 >= ymax)
            i1 = n - 1;
        else
            i1 = binary_search(y0, (double *)bingham_table_range, n);
        i0 = i1 - 1;

        if (y1 <= ymin)
            j1 = 1;
        else if (y1 >= ymax)
            j1 = n - 1;
        else
            j1 = binary_search(y1, (double *)bingham_table_range, n);
        j0 = j1 - 1;

        if (y2 <= ymin)
            k1 = 1;
        else if (y2 >= ymax)
            k1 = n - 1;
        else
            k1 = binary_search(y2, (double *)bingham_table_range, n);
        k0 = k1 - 1;

        range_tensor_acc[i][0] = bingham_table_range[i0];
        range_tensor_acc[i][1] = bingham_table_range[j0];
        range_tensor_acc[i][2] = bingham_table_range[k0];

        double d0 = bingham_table_range[i1] - bingham_table_range[i0];
        double d1 = bingham_table_range[j1] - bingham_table_range[j0];
        double d2 = bingham_table_range[k1] - bingham_table_range[k0];

        d_tensor_acc[i][0] = d0;
        d_tensor_acc[i][1] = d1;
        d_tensor_acc[i][2] = d2;

        double F000 = bingham_F_table_get(i0, j0, k0);
        double F001 = bingham_F_table_get(i0, j0, k1);
        double F010 = bingham_F_table_get(i0, j1, k0);
        double F011 = bingham_F_table_get(i0, j1, k1);
        double F100 = bingham_F_table_get(i1, j0, k0);
        double F101 = bingham_F_table_get(i1, j0, k1);
        double F110 = bingham_F_table_get(i1, j1, k0);
        double F111 = bingham_F_table_get(i1, j1, k1);

        F_tensor_acc[i][0] = F000;
        F_tensor_acc[i][1] = F001;
        F_tensor_acc[i][2] = F010;
        F_tensor_acc[i][3] = F011;
        F_tensor_acc[i][4] = F100;
        F_tensor_acc[i][5] = F101;
        F_tensor_acc[i][6] = F110;
        F_tensor_acc[i][7] = F111;
    }

    range_tensor = range_tensor.to(l.device(), l.dtype());
    d_tensor = d_tensor.to(l.device(), l.dtype());
    F_tensor = F_tensor.to(l.device(), l.dtype());

    auto y = (torch::sqrt(-l) - range_tensor) / d_tensor;

    auto y0 = torch::narrow(y, 1, 0, 1);
    auto y1 = torch::narrow(y, 1, 1, 1);
    auto y2 = torch::narrow(y, 1, 2, 1);

    auto F000 = torch::narrow(F_tensor, 1, 0, 1);
    auto F001 = torch::narrow(F_tensor, 1, 1, 1);
    auto F010 = torch::narrow(F_tensor, 1, 2, 1);
    auto F011 = torch::narrow(F_tensor, 1, 3, 1);
    auto F100 = torch::narrow(F_tensor, 1, 4, 1);
    auto F101 = torch::narrow(F_tensor, 1, 5, 1);
    auto F110 = torch::narrow(F_tensor, 1, 6, 1);
    auto F111 = torch::narrow(F_tensor, 1, 7, 1);

    // interpolate over k
    auto F00 = F000 + y2 * (F001 - F000);
    auto F01 = F010 + y2 * (F011 - F010);
    auto F10 = F100 + y2 * (F101 - F100);
    auto F11 = F110 + y2 * (F111 - F110);

    // interpolate over j
    auto F0 = F00 + y1 * (F01 - F00);
    auto F1 = F10 + y1 * (F11 - F10);

    // interpolate over i
    auto F = F0 + y0 * (F1 - F0);

    auto a = torch::cat({a1, a2, a3}, 1);
    auto L = torch::sum(l * a * a, 1, true) - torch::log(F);

    return L;
}

torch::Tensor bingham_prob_m(torch::Tensor q, torch::Tensor l, torch::Tensor q_gt)
{
    /*
    q:  batch_size \times 16, the quaternions predicted by the network
    l:  batch_size \times 3, the lambda values predicted by the network
    q_gt batch_size \times 4, the ground truth quaternionsi

    The transposed version of V(q) is:
    V(q)' = {
        q0, q1, q2, q3;
        -q1, q0, -q3, q2;
        -q2, q3, q0, -q1;
        q3, q2, -q1, -q0
    }

    V(1) = {
        q0, q4, q8, q12;
        q1, q5, q9, q13;
        q2, q6, q10, q14;
        q3, q7, q11, q15
    }

    V is a matrix here, for convinience, we still use q to represent it. But it will have q0 - q15 instead.

    each row would take dot product with the ground truth quatenion.
    Since the first value of lambda is always 0, so we don't have to calculate it for the first row

    */
    auto l_cpu = l.to(torch::device(torch::kCPU));
    auto l_acc = l_cpu.accessor<float, 2>();

    torch::Tensor range_tensor(torch::zeros({l_acc.size(0), 3}));
    torch::Tensor d_tensor(torch::zeros({l_acc.size(0), 3}));
    torch::Tensor F_tensor(torch::zeros({l_acc.size(0), 8}));

    auto range_tensor_acc = range_tensor.accessor<float, 2>();
    auto d_tensor_acc = d_tensor.accessor<float, 2>();
    auto F_tensor_acc = F_tensor.accessor<float, 2>();

    auto gt0 = torch::narrow(q_gt, 1, 0, 1);
    auto gt1 = torch::narrow(q_gt, 1, 1, 1);
    auto gt2 = torch::narrow(q_gt, 1, 2, 1);
    auto gt3 = torch::narrow(q_gt, 1, 3, 1);

    auto q0 = torch::narrow(q, 1, 0, 1);
    auto q1 = torch::narrow(q, 1, 1, 1);
    auto q2 = torch::narrow(q, 1, 2, 1);
    auto q3 = torch::narrow(q, 1, 3, 1);
    auto q4 = torch::narrow(q, 1, 4, 1);
    auto q5 = torch::narrow(q, 1, 5, 1);
    auto q6 = torch::narrow(q, 1, 6, 1);
    auto q7 = torch::narrow(q, 1, 7, 1);
    auto q8 = torch::narrow(q, 1, 8, 1);
    auto q9 = torch::narrow(q, 1, 9, 1);
    auto q10 = torch::narrow(q, 1, 10, 1);
    auto q11 = torch::narrow(q, 1, 11, 1);
    auto q12 = torch::narrow(q, 1, 12, 1);
    auto q13 = torch::narrow(q, 1, 13, 1);
    auto q14 = torch::narrow(q, 1, 14, 1);
    auto q15 = torch::narrow(q, 1, 15, 1);

    auto a1 = gt0 * q1 + gt1 * q5 + gt2 * q9 + gt3 * q13;
    auto a2 = gt0 * q2 + gt1 * q6 + gt2 * q10 + gt3 * q14;
    auto a3 = gt0 * q3 + gt1 * q7 + gt2 * q11 + gt3 * q15;

    for (auto i = 0; i < l_acc.size(0); i++)
    {
        double y0 = sqrt(-l_acc[i][0]);
        double y1 = sqrt(-l_acc[i][1]);
        double y2 = sqrt(-l_acc[i][2]);

        int n = BINGHAM_TABLE_LENGTH;
        double ymin = bingham_table_range[0];
        double ymax = bingham_table_range[n - 1];

        // get the Z-table cell between (i0,i1), (j0,j1), and (k0,k1)

        int i0, j0, k0, i1, j1, k1;

        if (y0 <= ymin)
            i1 = 1;
        else if (y0 >= ymax)
            i1 = n - 1;
        else
            i1 = binary_search(y0, (double *)bingham_table_range, n);
        i0 = i1 - 1;

        if (y1 <= ymin)
            j1 = 1;
        else if (y1 >= ymax)
            j1 = n - 1;
        else
            j1 = binary_search(y1, (double *)bingham_table_range, n);
        j0 = j1 - 1;

        if (y2 <= ymin)
            k1 = 1;
        else if (y2 >= ymax)
            k1 = n - 1;
        else
            k1 = binary_search(y2, (double *)bingham_table_range, n);
        k0 = k1 - 1;

        range_tensor_acc[i][0] = bingham_table_range[i0];
        range_tensor_acc[i][1] = bingham_table_range[j0];
        range_tensor_acc[i][2] = bingham_table_range[k0];

        double d0 = bingham_table_range[i1] - bingham_table_range[i0];
        double d1 = bingham_table_range[j1] - bingham_table_range[j0];
        double d2 = bingham_table_range[k1] - bingham_table_range[k0];

        d_tensor_acc[i][0] = d0;
        d_tensor_acc[i][1] = d1;
        d_tensor_acc[i][2] = d2;

        double F000 = bingham_F_table_get(i0, j0, k0);
        double F001 = bingham_F_table_get(i0, j0, k1);
        double F010 = bingham_F_table_get(i0, j1, k0);
        double F011 = bingham_F_table_get(i0, j1, k1);
        double F100 = bingham_F_table_get(i1, j0, k0);
        double F101 = bingham_F_table_get(i1, j0, k1);
        double F110 = bingham_F_table_get(i1, j1, k0);
        double F111 = bingham_F_table_get(i1, j1, k1);

        F_tensor_acc[i][0] = F000;
        F_tensor_acc[i][1] = F001;
        F_tensor_acc[i][2] = F010;
        F_tensor_acc[i][3] = F011;
        F_tensor_acc[i][4] = F100;
        F_tensor_acc[i][5] = F101;
        F_tensor_acc[i][6] = F110;
        F_tensor_acc[i][7] = F111;
    }

    range_tensor = range_tensor.to(l.device(), l.dtype());
    d_tensor = d_tensor.to(l.device(), l.dtype());
    F_tensor = F_tensor.to(l.device(), l.dtype());

    auto y = (torch::sqrt(-l) - range_tensor) / d_tensor;

    auto y0 = torch::narrow(y, 1, 0, 1);
    auto y1 = torch::narrow(y, 1, 1, 1);
    auto y2 = torch::narrow(y, 1, 2, 1);

    auto F000 = torch::narrow(F_tensor, 1, 0, 1);
    auto F001 = torch::narrow(F_tensor, 1, 1, 1);
    auto F010 = torch::narrow(F_tensor, 1, 2, 1);
    auto F011 = torch::narrow(F_tensor, 1, 3, 1);
    auto F100 = torch::narrow(F_tensor, 1, 4, 1);
    auto F101 = torch::narrow(F_tensor, 1, 5, 1);
    auto F110 = torch::narrow(F_tensor, 1, 6, 1);
    auto F111 = torch::narrow(F_tensor, 1, 7, 1);

    // interpolate over k
    auto F00 = F000 + y2 * (F001 - F000);
    auto F01 = F010 + y2 * (F011 - F010);
    auto F10 = F100 + y2 * (F101 - F100);
    auto F11 = F110 + y2 * (F111 - F110);

    // interpolate over j
    auto F0 = F00 + y1 * (F01 - F00);
    auto F1 = F10 + y1 * (F11 - F10);

    // interpolate over i
    auto F = F0 + y0 * (F1 - F0);

    auto a = torch::cat({a1, a2, a3}, 1);
    auto L = torch::sum(l * a * a, 1, true) - torch::log(F);

    return L;
}

torch::Tensor bingham_entropy(torch::Tensor z)
{
    auto F = F_lookup_3d(z);
    auto dF = dF_lookup_3d(z);
    auto e = torch::log(F) - torch::sum(dF * z, -1) / F;
    return e;
}

PYBIND11_MODULE(torch_bingham, m)
{
    m.def("F_lookup_3d", &F_lookup_3d, "bingham_F_looup_3d");
    m.def("dF_lookup_3d", &dF_lookup_3d, "bingham_dF_looup_3d");
    m.def("bingham_entropy", &bingham_entropy, "compute bingham entropy");
    m.def("bingham_prob", &bingham_prob, "bingham probability");
    m.def("bingham_prob_m", &bingham_prob_m, "bingham probability, input V is a matrix");
}
