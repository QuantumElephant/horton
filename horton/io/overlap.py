from collections import OrderedDict

import numpy as np
from overlap_helper import tfs


# void GBasis::compute_two_index(double* output, GB2Integral* integral) {
#     IterGB2 iter = IterGB2(this);
#     iter.update_shell();
#     do {
#         integral->reset(iter.shell_type0, iter.shell_type1, iter.r0, iter.r1);
#         iter.update_prim();
#         do {
#             integral->add(iter.con_coeff, iter.alpha0, iter.alpha1, iter.scales0, iter.scales1);
#         } while (iter.inc_prim());
#         integral->cart_to_pure();
#         iter.store(integral->get_work(), output);
#     } while (iter.inc_shell());
# }

# void IterGB2::update_shell() {
#     // Update fields that depend on shell and related counters.
#     nprim0 = gbasis->nprims[ishell0];
#     nprim1 = gbasis->nprims[ishell1];
#     // Update indexes in output array
#     ibasis0 = basis_offsets[ishell0];
#     ibasis1 = basis_offsets[ishell1];
#     // update centers
#     r0 = gbasis->centers + 3*gbasis->shell_map[ishell0];
#     r1 = gbasis->centers + 3*gbasis->shell_map[ishell1];
#     // update shell types
#     shell_type0 = gbasis->shell_types[ishell0];
#     shell_type1 = gbasis->shell_types[ishell1];
#     // reset contraction counters
#     iprim0 = 0;
#     iprim1 = 0;
# }

# void IterGB2::update_prim() {
#     // Update fields that depend on primitive counters.
#     alpha0 = gbasis->alphas[oprim0 + iprim0];
#     alpha1 = gbasis->alphas[oprim1 + iprim1];
#     con_coeff = gbasis->con_coeffs[oprim0 + iprim0]*
#                 gbasis->con_coeffs[oprim1 + iprim1];
#     scales0 = gbasis->get_scales(oprim0 + iprim0);
#     scales1 = gbasis->get_scales(oprim1 + iprim1);
# }
def compute_overlap(centers, shell_map, nprims, shell_types, alphas, con_coeffs):
    nshell = len(shell_types)
    nbasis = sum([get_shell_nbasis(i) for i in shell_types])
    nscales = sum([get_shell_nbasis(abs(s)) * p for s, p in zip(shell_types, nprims)])

    ps_offsets = []
    last = 0
    for i in shell_types:
        ps_offsets.append(last)
        last += get_shell_nbasis(i)
    ps_offsets.append(nbasis)
    integral = np.zeros((nbasis, nbasis))
    scales, scales_offsets = init_scales(alphas, nshell, nprims, nscales, shell_types)

    oprim0 = 0
    for ishell0 in range(nshell):
        nprim0 = nprims[ishell0]
        r0 = centers[shell_map[ishell0], :]
        shell_type0 = shell_types[ishell0]

        oprim1 = 0
        for ishell1 in range(0, ishell0 + 1):
            nprim1 = nprims[ishell1]
            r1 = centers[shell_map[ishell1], :]
            shell_type1 = shell_types[ishell1]

            # print "integrals within shell", shell_type0, shell_type1,
            result = np.zeros((get_shell_nbasis(abs(shell_type0)),  # In cartesian coordinates
                               get_shell_nbasis(abs(shell_type1))),  # In cartesian coordinates
                              dtype=float)
            for iprim0 in range(nprim0):
                alpha0 = alphas[oprim0 + iprim0]
                scales0 = scales[scales_offsets[oprim0 + iprim0]:]
                for iprim1 in range(nprim1):
                    alpha1 = alphas[oprim1 + iprim1]
                    con_coeff = con_coeffs[oprim0 + iprim0] * con_coeffs[oprim1 + iprim1]
                    scales1 = scales[scales_offsets[oprim1 + iprim1]:]
                    # print "scales starting:", oprim0 + iprim0, oprim1 + iprim1,  scales0[0], scales1[0]
                    result += add_overlap(con_coeff, alpha0, alpha1, scales0, scales1, r0, r1,
                                          shell_type0, shell_type1)
            #         print result.ravel()
            #         print " "
            # print "-" * 50
            # cart to pure
            if shell_type0 < -1:
                result = np.dot(tfs[abs(shell_type0)], result)
            if shell_type1 < -1:
                result = np.dot(result, tfs[abs(shell_type1)].T)

            # print "offsets", (ps_offsets[ishell0], ps_offsets[ishell0 + 1]), (ps_offsets[ishell1],ps_offsets[ishell1 + 1])
            # print ps_offsets
            # store result
            integral[ps_offsets[ishell0]:ps_offsets[ishell0 + 1],
            ps_offsets[ishell1]:ps_offsets[ishell1 + 1]] = result
            integral[ps_offsets[ishell1]:ps_offsets[ishell1 + 1],
            ps_offsets[ishell0]:ps_offsets[ishell0 + 1]] = result.T

            oprim1 += nprim1
        oprim0 += nprim0

    # integral = np.tril(integral, -1).T
    return integral


# void GBasis::init_scales() {
#     long n[3], counter=0, oprim=0;
#     double alpha;
#     for (long ishell=0; ishell < nshell; ishell++) {
#         for (long iprim=0; iprim < nprims[ishell]; iprim++) {
#             scales_offsets[oprim + iprim] = counter;
#             alpha = alphas[oprim + iprim];
#             n[0] = abs(shell_types[ishell]);
#             n[1] = 0;
#             n[2] = 0;
#             do {
#                 scales[counter] = normalization(alpha, n);
#                 counter += 1;
#             } while (iter_pow1_inc(n));
#         }
#         oprim += nprims[ishell];
#     }
# }

def init_scales(alphas, nshell, nprims, nscales, shell_types):
    counter, oprim = 0, 0
    scales = np.zeros(nscales)
    scales_offsets = np.zeros(sum(nprims), dtype=int)

    for s in range(nshell):
        for p in range(nprims[s]):
            scales_offsets[oprim + p] = counter
            alpha = alphas[oprim + p]
            for n in get_iter_pow(abs(shell_types[s])):
                scales[counter] = gob_cart_normalization(alpha, n)
                counter += 1
        oprim += nprims[s]

    # print "scales: ", scales
    return scales, scales_offsets


# const double gob_cart_normalization(const double alpha, const long* n) {
#     return sqrt(pow(4.0*alpha, n[0]+n[1]+n[2])*pow(2.0*alpha/M_PI, 1.5)
#            /(fac2(2*n[0]-1)*fac2(2*n[1]-1)*fac2(2*n[2]-1)));
# }

def gob_cart_normalization(alpha, n):  # from utils
    # sqrt(pow(4.0*alpha, n[0]+n[1]+n[2])*pow(2.0*alpha/M_PI, 1.5)
    #        /(fac2(2*n[0]-1)*fac2(2*n[1]-1)*fac2(2*n[2]-1)));

    vfac2 = np.vectorize(fac2)
    return np.sqrt((4 * alpha) ** sum(n) * (2 * alpha / np.pi) ** 1.5 / np.prod(vfac2(2 * n - 1)))


def add_overlap(coeff, alpha0, alpha1, scales0, scales1, r0, r1, shell_type0, shell_type1):
    # gamma_inv = 1.0/(alpha0 + alpha1);
    # pre = coeff*exp(-alpha0*alpha1*gamma_inv*dist_sq(r0, r1));
    # compute_gpt_center(alpha0, r0, alpha1, r1, gamma_inv, gpt_center);
    # i2p.reset(abs(shell_type0), abs(shell_type1));
    # do {
    #     work_cart[i2p.offset] += pre*(
    #         gb_overlap_int1d(i2p.n0[0], i2p.n1[0], gpt_center[0] - r0[0], gpt_center[0] - r1[0], gamma_inv)*
    #         gb_overlap_int1d(i2p.n0[1], i2p.n1[1], gpt_center[1] - r0[1], gpt_center[1] - r1[1], gamma_inv)*
    #         gb_overlap_int1d(i2p.n0[2], i2p.n1[2], gpt_center[2] - r0[2], gpt_center[2] - r1[2], gamma_inv)*
    #         scales0[i2p.ibasis0]*scales1[i2p.ibasis1]
    #     );
    # } while (i2p.inc());

    gamma_inv = 1.0 / (alpha0 + alpha1)
    pre = coeff * np.exp(-alpha0 * alpha1 * gamma_inv * dist_sq(r0, r1))
    gpt_center = compute_gpt_center(alpha0, r0, alpha1, r1, gamma_inv)

    nshell0 = get_shell_nbasis(abs(shell_type0))  # In cartesian coordinates
    nshell1 = get_shell_nbasis(abs(shell_type1))  # In cartesian coordinates
    result = np.zeros([nshell0, nshell1], dtype=float)
    for n0, s0 in zip(get_iter_pow(abs(shell_type0)), range(nshell0)):
        for n1, s1 in zip(get_iter_pow(abs(shell_type1)), range(nshell1)):
            result[s0, s1] = pre * np.prod(
                vec_gb_overlap_int1d(n0, n1, gpt_center - r0, gpt_center - r1,
                                     gamma_inv)) * scales0[s0] * scales1[s1]
            # print np.ravel_multi_index((s0, s1), result.shape), "|", result[s0, s1], "|", pre, vec_gb_overlap_int1d(n0, n1, gpt_center - r0, gpt_center - r1,
            #                          gamma_inv), scales0[s0], scales1[s1]

    return result


def get_shell_nbasis(shell):
    if shell > 0:  # Cartesian
        return (shell + 1) * (shell + 2) / 2
    elif shell == -1:
        raise ValueError
    else:  # Pure
        return -2 * shell + 1


def dist_sq(r0, r1):
    return sum((r0 - r1) ** 2)


def compute_gpt_center(alpha0, r0, alpha1, r1, gamma_inv):
    return gamma_inv * (alpha0 * r0 + alpha1 * r1)


# double gb_overlap_int1d(long n0, long n1, double pa, double pb, double gamma_inv) {
#     long k, kmax;
#     double result;
#
#     kmax = (n0+n1)/2;
#     result = 0.0;
#     for (k=0; k<=kmax; k++) {
#         result += fac2(2*k-1)*gpt_coeff(2*k, n0, n1, pa, pb)*pow(0.5*gamma_inv,k);
#     }
#     return sqrt(M_PI*gamma_inv)*result;
# }

def gb_overlap_int1d(n0, n1, pa, pb, gamma_inv):
    kmax = (n0 + n1) / 2
    result = 0.0
    for k in range(kmax + 1):
        result += fac2(2 * k - 1) * gpt_coeff(2 * k, n0, n1, pa, pb) * np.power(0.5 * gamma_inv, k)

    return np.sqrt(np.pi * gamma_inv) * result


vec_gb_overlap_int1d = np.vectorize(gb_overlap_int1d)


def fac2(n):
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result


# double gpt_coeff(long k, long n0, long n1, double pa, double pb) {
#     long i0, i1;
#     double result;
#
#     result = 0;
#     i0 = k-n1;
#     if (i0<0) i0 = 0;
#     i1 = k - i0;
#     do {
#         result += binom(n0, i0)*binom(n1, i1)*pow(pa, n0-i0)*pow(pb, n1-i1);
#         i0++;
#         i1--;
#     } while ((i1 >= 0)&&(i0 <= n0));
#     return result;
# }

def gpt_coeff(k, n0, n1, pa, pb):
    result = 0.0
    i0 = k - n1
    if i0 < 0:
        i0 = 0
    i1 = k - i0

    while True:
        result += binom(n0, i0) * binom(n1, i1) * np.power(pa, n0 - i0) * np.power(pb, n1 - i1)
        i0 += 1
        i1 -= 1
        if not (i1 >= 0 and i0 <= n0):
            break
    return result


# long binom(long n, long m) {
#     long numer = 1, denom = 1;
#     while (n > m) {
#         numer *= n;
#         denom *= (n-m);
#         n--;
#     }
#     return numer/denom;
# }

def binom(n, m):
    numer, denom = 1, 1
    while n > m:
        numer *= n
        denom *= n - m
        n -= 1
    return numer / denom


# -----------------------------------------------------------------------------

# int iter_pow1_inc(long* n) {
#     // Modify the indexes in place as to move to the next combination of powers
#     // withing one angular momentum.
#     // Note: shell_type = n[0] + n[1] + n[2];
#     if (n[1] == 0) {
#       if (n[0] == 0) {
#         n[0] = n[2];
#         n[2] = 0;
#         return 0;
#       } else {
#         n[1] = n[2] + 1;
#         n[2] = 0;
#         n[0]--;
#       }
#     } else {
#       n[1]--;
#       n[2]++;
#     }
#     return 1;
# }

# def iter_pow1_inc(n):
#     if n[1] == 0:
#         if n[0] == 0:
#             n[0] = n[2]
#             n[2] = 0
#             return 0
#         else:
#             n[1] = n[2] + 1
#             n[2] = 0
#             n[0] -= 1
#     else:
#         n[1] -= 1
#         n[2] += 1
#     return 1
#
#
# # void IterPow2::reset(long shell_type0, long shell_type1) {
# #     shell_type0 = shell_type0;
# #     shell_type1 = shell_type1;
# #     n0[0] = shell_type0;
# #     n0[1] = 0;
# #     n0[2] = 0;
# #     n1[0] = shell_type1;
# #     n1[1] = 0;
# #     n1[2] = 0;
# #     ibasis0 = 0;
# #     ibasis1 = 0;
# #     offset = 0;
# # }
#
#
# # int IterPow2::inc() {
# #     // Increment indexes of shell 1
# #     int result;
# #     result = iter_pow1_inc(n1);
# #     if (result) {
# #         offset++;
# #         ibasis1++;
# #     } else {
# #         // Increment indexes of shell 0
# #         result = iter_pow1_inc(n0);
# #         ibasis1 = 0;
# #         if (result) {
# #             offset++;
# #             ibasis0++;
# #         } else {
# #             offset = 0;
# #             ibasis0 = 0;
# #         }
# #     }
# #     return result;
# # }
# class Iterpow2(object):
#     def __init__(self, shell_type0, shell_type1):
#         self.shell_type0 = shell_type0
#         self.shell_type1 = shell_type1
#
#         self.n0 = [shell_type0, 0, 0]
#         self.n1 = [shell_type1, 0, 0]
#
#         self.ibasis0 = 0
#         self.ibasis1 = 0
#         self.offset = 0
#
#     def __repr__(self):
#         return "({}, {}) n0:{} n1:{} ibasis0:{} ibasis1:{} offset={}".format(self.shell_type0,
#                                                                              self.shell_type1,
#                                                                              self.n0, self.n1,
#                                                                              self.ibasis0,
#                                                                              self.ibasis1,
#                                                                              self.offset)
#
#     def inc(self):
#         result = iter_pow1_inc(self.n1)
#         if result:
#             self.offset += 1
#             self.ibasis1 += 1
#         else:
#             result = iter_pow1_inc(self.n0)
#             self.ibasis1 = 0
#             if result:
#                 self.offset += 1
#                 self.ibasis0 += 1
#             else:
#                 self.offset = 0
#                 self.ibasis0 = 0
#         return result
#
#
# ip2 = Iterpow2(2, 2)
# res = 1
# while res:
#     print ip2
#     res = ip2.inc()

def get_iter_pow(n):
    for nx in range(n, -1, -1):
        for ny in range(n - nx, -1, -1):
            nz = n - nx - ny
            yield np.array((nx, ny, nz), dtype=int)


# # test_load_fchk_hf_sto3g_num
# d = OrderedDict([('centers', np.array([[0., 0., 0.19048439],
#                                        [0., 0., -1.71435955]])),
#                  ('shell_map', np.array([0, 0, 0, 1])),
#                  ('nprims', np.array([3, 3, 3, 3])), ('shell_types', np.array([0, 0, 1, 0])),
#                  ('alphas', np.array([166.679134, 30.3608123, 8.21682067, 6.46480325,
#                                       1.50228124, 0.48858849, 6.46480325, 1.50228124,
#                                       0.48858849, 3.42525091, 0.62391373, 0.1688554])),
#                  ('con_coeffs',
#                   np.array([0.15432897, 0.53532814, 0.44463454, -0.09996723, 0.39951283,
#                             0.70011547, 0.15591627, 0.60768372, 0.39195739, 0.15432897,
#                             0.53532814, 0.44463454]))])

# # test_load_azirine_cc
# d = OrderedDict([('centers', np.array([[-8.24047836e-01, -1.42796461e+00, -8.58966424e-06],
#                                        [1.52641681e+00, 2.35709498e-01, -8.58966424e-06],
#                                        [-1.05795747e+00, 9.14299426e-01, -8.58966424e-06],
#                                        [2.56217426e+00, 2.23789419e-01, -1.75443033e+00],
#                                        [2.56206608e+00, 2.23545044e-01, 1.75460212e+00],
#                                        [-2.16666151e+00, 2.64836424e+00, -8.58966424e-06]])), (
#                  'shell_map',
#                  np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5])),
#                  ('nprims', np.array([6, 3, 3, 1, 1, 6, 3, 3, 1, 1, 6, 3, 3, 1, 1, 3, 1, 3, 1, 3, 1])),
#                  ('shell_types',
#                   np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])),
#                  ('alphas', np.array([4.17351146e+03, 6.27457911e+02, 1.42902093e+02,
#                                    4.02343293e+01, 1.28202129e+01, 4.39043701e+00,
#                                    1.16263619e+01, 2.71627981e+00, 7.72218397e-01,
#                                    1.16263619e+01, 2.71627981e+00, 7.72218397e-01,
#                                    2.12031498e-01, 2.12031498e-01, 3.04752488e+03,
#                                    4.57369518e+02, 1.03948685e+02, 2.92101553e+01,
#                                    9.28666296e+00, 3.16392696e+00, 7.86827235e+00,
#                                    1.88128854e+00, 5.44249258e-01, 7.86827235e+00,
#                                    1.88128854e+00, 5.44249258e-01, 1.68714478e-01,
#                                    1.68714478e-01, 3.04752488e+03, 4.57369518e+02,
#                                    1.03948685e+02, 2.92101553e+01, 9.28666296e+00,
#                                    3.16392696e+00, 7.86827235e+00, 1.88128854e+00,
#                                    5.44249258e-01, 7.86827235e+00, 1.88128854e+00,
#                                    5.44249258e-01, 1.68714478e-01, 1.68714478e-01,
#                                    1.87311370e+01, 2.82539436e+00, 6.40121692e-01,
#                                    1.61277759e-01, 1.87311370e+01, 2.82539436e+00,
#                                    6.40121692e-01, 1.61277759e-01, 1.87311370e+01,
#                                    2.82539436e+00, 6.40121692e-01, 1.61277759e-01])), (
#                  'con_coeffs', np.array([0.00183477, 0.01399463, 0.06858655, 0.23224087, 0.46906995,
#                                          0.3604552, -0.11496118, -0.16911748, 1.14585195,
#                                          0.06757974,
#                                          0.3239073, 0.74089514, 1., 1., 0.00183474,
#                                          0.01403732, 0.06884262, 0.23218444, 0.46794135, 0.36231199,
#                                          -0.11933242, -0.16085415, 1.14345644, 0.06899907,
#                                          0.31642396,
#                                          0.74430829, 1., 1., 0.00183474, 0.01403732,
#                                          0.06884262, 0.23218444, 0.46794135, 0.36231199,
#                                          -0.11933242,
#                                          -0.16085415, 1.14345644, 0.06899907, 0.31642396,
#                                          0.74430829,
#                                          1., 1., 0.0334946, 0.23472695, 0.81375733,
#                                          1., 0.0334946, 0.23472695, 0.81375733, 1.,
#                                          0.0334946, 0.23472695, 0.81375733, 1.]))])

# test_load_fchk_o2_cc_pvtz_pure_num
# d = OrderedDict([('centers', np.array([[0.00000000e+00, 0.00000000e+00, 1.09122830e+00],
#                                        [1.33636924e-16, 0.00000000e+00, -1.09122830e+00]])), (
#                  'shell_map',
#                  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])), (
#                  'nprims',
#                  np.array([7, 7, 1, 1, 3, 1, 1, 1, 1, 1, 7, 7, 1, 1, 3, 1, 1, 1, 1, 1])),
#                  ('shell_types', np.array([0, 0, 0, 0, 1, 1, 1, -2, -2, -3, 0, 0, 0, 0, 1, 1, 1,
#                                            -2, -2, -3])),
#                  ('alphas', np.array([1.53300000e+04, 2.29900000e+03, 5.22400000e+02,
#                                       1.47300000e+02, 4.75500000e+01, 1.67600000e+01,
#                                       6.20700000e+00, 1.53300000e+04, 2.29900000e+03,
#                                       5.22400000e+02, 4.75500000e+01, 1.67600000e+01,
#                                       6.20700000e+00, 6.88200000e-01, 1.75200000e+00,
#                                       2.38400000e-01, 3.44600000e+01, 7.74900000e+00,
#                                       2.28000000e+00, 7.15600000e-01, 2.14000000e-01,
#                                       2.31400000e+00, 6.45000000e-01, 1.42800000e+00,
#                                       1.53300000e+04, 2.29900000e+03, 5.22400000e+02,
#                                       1.47300000e+02, 4.75500000e+01, 1.67600000e+01,
#                                       6.20700000e+00, 1.53300000e+04, 2.29900000e+03,
#                                       5.22400000e+02, 4.75500000e+01, 1.67600000e+01,
#                                       6.20700000e+00, 6.88200000e-01, 1.75200000e+00,
#                                       2.38400000e-01, 3.44600000e+01, 7.74900000e+00,
#                                       2.28000000e+00, 7.15600000e-01, 2.14000000e-01,
#                                       2.31400000e+00, 6.45000000e-01, 1.42800000e+00])),
#                  ('con_coeffs', np.array([5.19808943e-04, 4.02025621e-03, 2.07128267e-02,
#                                           8.10105536e-02, 2.35962985e-01, 4.42653446e-01,
#                                           3.57064423e-01, 9.13376623e-06, 6.07362596e-05,
#                                           2.68782282e-04, -6.96940030e-03, -6.06456900e-02,
#                                           -1.65519536e-01, 1.07151369e+00, 1.00000000e+00,
#                                           1.00000000e+00, 4.11634896e-02, 2.57762836e-01,
#                                           8.02419274e-01, 1.00000000e+00, 1.00000000e+00,
#                                           1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
#                                           5.19808943e-04, 4.02025621e-03, 2.07128267e-02,
#                                           8.10105536e-02, 2.35962985e-01, 4.42653446e-01,
#                                           3.57064423e-01, 9.13376623e-06, 6.07362596e-05,
#                                           2.68782282e-04, -6.96940030e-03, -6.06456900e-02,
#                                           -1.65519536e-01, 1.07151369e+00, 1.00000000e+00,
#                                           1.00000000e+00, 4.11634896e-02, 2.57762836e-01,
#                                           8.02419274e-01, 1.00000000e+00, 1.00000000e+00,
#                                           1.00000000e+00, 1.00000000e+00, 1.00000000e+00]))])
#
# #test_load_fchk_o2_cc_pvtz_cart_num
# d = OrderedDict([('centers', np.array([[0.00000000e+00, 0.00000000e+00, 1.09122830e+00],
#                                        [1.33636924e-16, 0.00000000e+00, -1.09122830e+00]])), (
#                  'shell_map',
#                  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])), (
#                  'nprims',
#                  np.array([7, 7, 1, 1, 3, 1, 1, 1, 1, 1, 7, 7, 1, 1, 3, 1, 1, 1, 1, 1])), (
#                  'shell_types',
#                  np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])),
#                  ('alphas', np.array([1.53300000e+04, 2.29900000e+03, 5.22400000e+02,
#                                       1.47300000e+02, 4.75500000e+01, 1.67600000e+01,
#                                       6.20700000e+00, 1.53300000e+04, 2.29900000e+03,
#                                       5.22400000e+02, 4.75500000e+01, 1.67600000e+01,
#                                       6.20700000e+00, 6.88200000e-01, 1.75200000e+00,
#                                       2.38400000e-01, 3.44600000e+01, 7.74900000e+00,
#                                       2.28000000e+00, 7.15600000e-01, 2.14000000e-01,
#                                       2.31400000e+00, 6.45000000e-01, 1.42800000e+00,
#                                       1.53300000e+04, 2.29900000e+03, 5.22400000e+02,
#                                       1.47300000e+02, 4.75500000e+01, 1.67600000e+01,
#                                       6.20700000e+00, 1.53300000e+04, 2.29900000e+03,
#                                       5.22400000e+02, 4.75500000e+01, 1.67600000e+01,
#                                       6.20700000e+00, 6.88200000e-01, 1.75200000e+00,
#                                       2.38400000e-01, 3.44600000e+01, 7.74900000e+00,
#                                       2.28000000e+00, 7.15600000e-01, 2.14000000e-01,
#                                       2.31400000e+00, 6.45000000e-01, 1.42800000e+00])),
#                  ('con_coeffs', np.array([5.19808943e-04, 4.02025621e-03, 2.07128267e-02,
#                                           8.10105536e-02, 2.35962985e-01, 4.42653446e-01,
#                                           3.57064423e-01, 9.13376623e-06, 6.07362596e-05,
#                                           2.68782282e-04, -6.96940030e-03, -6.06456900e-02,
#                                           -1.65519536e-01, 1.07151369e+00, 1.00000000e+00,
#                                           1.00000000e+00, 4.11634896e-02, 2.57762836e-01,
#                                           8.02419274e-01, 1.00000000e+00, 1.00000000e+00,
#                                           1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
#                                           5.19808943e-04, 4.02025621e-03, 2.07128267e-02,
#                                           8.10105536e-02, 2.35962985e-01, 4.42653446e-01,
#                                           3.57064423e-01, 9.13376623e-06, 6.07362596e-05,
#                                           2.68782282e-04, -6.96940030e-03, -6.06456900e-02,
#                                           -1.65519536e-01, 1.07151369e+00, 1.00000000e+00,
#                                           1.00000000e+00, 4.11634896e-02, 2.57762836e-01,
#                                           8.02419274e-01, 1.00000000e+00, 1.00000000e+00,
#                                           1.00000000e+00, 1.00000000e+00, 1.00000000e+00]))])


# np.set_printoptions(threshold=np.nan)
# np.save("test/test_overlap", compute_overlap(*d.values()))
# [[  9.99999999e-01   2.37989883e-01   0.00000000e+00   0.00000000e+00  -1.48726293e-17   3.98036642e-02]
#  [  2.37989883e-01   1.00000000e+00   0.00000000e+00   0.00000000e+00  -2.15936741e-17   3.90178135e-01]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
#  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00   0.00000000e+00   0.00000000e+00]
#  [ -1.48726293e-17  -2.15936741e-17   0.00000000e+00   0.00000000e+00   1.00000000e+00  -3.16721125e-01]
#  [  3.98036642e-02   3.90178135e-01   0.00000000e+00   0.00000000e+00   -3.16721125e-01   9.99999999e-01]]


# [[  9.99999995e-01   2.37989884e-01  -2.35472394e-17  -4.33531238e-17   0.00000000e+00   0.00000000e+00]
#  [  2.37989884e-01   1.00000001e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
#  [ -2.35472394e-17  -4.33531238e-17   4.46884699e+00   0.00000000e+00   3.79028507e-01   3.41881504e+00]
#  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.64464483e+02  -6.32552678e+00   0.00000000e+00]
#  [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   2.57473548e+01   0.00000000e+00]
#  [  3.79028507e-01   3.41881504e+00  -6.32552678e+00   0.00000000e+00   0.00000000e+00   6.03734498e+01]]
# ----------------------------------------------------------------------------------
