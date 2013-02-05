# -*- coding: utf-8 -*-
# Horton is a Density Functional Theory program.
# Copyright (C) 2011-2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--

from horton.newton import *
from horton import *
import numpy as np

from matplotlib import pyplot

def get_water_sto3g_hf(lf=None):
    if lf is None:
        lf = DenseLinalgFactory()
    fn = context.get_fn('test/water_sto3g_hf_g03.log')
    overlap, kinetic, nuclear_attraction, electronic_repulsion = load_operators_g09(fn, lf)
    coeffs = np.array([
        9.94099882E-01, 2.67799213E-02, 3.46630004E-03, -1.54676269E-15,
        2.45105601E-03, -6.08393842E-03, -6.08393693E-03, -2.32889095E-01,
        8.31788042E-01, 1.03349385E-01, 9.97532839E-17, 7.30794097E-02,
        1.60223990E-01, 1.60223948E-01, 1.65502862E-08, -9.03020258E-08,
        -3.46565859E-01, -2.28559667E-16, 4.90116062E-01, 4.41542336E-01,
        -4.41542341E-01, 1.00235366E-01, -5.23423149E-01, 6.48259144E-01,
        -5.78009326E-16, 4.58390414E-01, 2.69085788E-01, 2.69085849E-01,
        8.92936017E-17, -1.75482465E-16, 2.47517845E-16, 1.00000000E+00,
        5.97439610E-16, -3.70474007E-17, -2.27323914E-17, -1.35631600E-01,
        9.08581133E-01, 5.83295647E-01, -4.37819173E-16, 4.12453695E-01,
        -8.07337352E-01, -8.07337875E-01, 5.67656309E-08, -4.29452066E-07,
        5.82525068E-01, -6.76605679E-17, -8.23811720E-01, 8.42614916E-01,
        -8.42614243E-01
    ]).reshape(7,7).T
    epsilons = np.array([
        -2.02333942E+01, -1.26583942E+00, -6.29365088E-01, -4.41724988E-01,
        -3.87671783E-01, 6.03082408E-01, 7.66134805E-01
    ])
    wfn = ClosedShellWFN(nep=5, lf=lf, nbasis=7)
    wfn.expansion.coeffs[:] = coeffs
    wfn.expansion.energies[:] = epsilons
    return lf, overlap, kinetic, nuclear_attraction, electronic_repulsion, wfn

def get_be_ss_hf(lf=None):
    if lf is None:
        lf = DenseLinalgFactory()
    fn = context.get_fn('test/be_hf_ss_g09.log')
    overlap, kinetic, nuclear_attraction, electronic_repulsion = load_operators_g09(fn, lf)
    coeffs = np.array([
       9.92898192E-01,  2.61376621E-02, -2.93880686E-01,  1.03514709E+00
     ]).reshape(2,2).T
    epsilons = np.array([
        -4.48399211E+00, -2.54037694E-01
    ])
    wfn = ClosedShellWFN(nep=2, lf=lf, nbasis=2)
    wfn.expansion.coeffs[:] = coeffs
    wfn.expansion.energies[:] = epsilons
    return lf, overlap, kinetic, nuclear_attraction, electronic_repulsion, wfn

def _lg_init(intgrl_init, nep, ee_rep=None):
    lf, overlap, kinetic, nuclear_attraction, electronic_repulsion, wfn = intgrl_init()
    nbasis = overlap.nbasis

    if ee_rep is not None:
        if isinstance(ee_rep,np.ndarray):
            nbasis = ee_rep.shape[0]
            electronic_repulsion = lf.create_two_body(nbasis)
            electronic_repulsion._array = ee_rep
        else:
            nbasis = ee_rep
            electronic_repulsion._array = electronic_repulsion._array[0:nbasis, 0:nbasis, 0:nbasis, 0:nbasis]

    Vee = HF_dVee(electronic_repulsion, nbasis, lf)
    W = []

    lg = old_lagrangian(kinetic, [nuclear_attraction, nuclear_attraction], overlap, [nep,nep], Vee, W, debug = False) # TODO: check normalization

    return lf, lg, nbasis, wfn

def Hessian(r):
    x = r[0]
    y = r[1]
    z = r[2]

    z2=z**2
    y2=y**2
    x2=x**2

    Hess = np.array([[2.*y2*z2, 4.*x*y*z2, 4.*x*y2*z],
                    [4.*x*y*z2, 2.*x2*z2, 4.*x2*y*z],
                    [4.*x*y2*z, 4.*x2*y*z, 2.*x2*y2]])

    return Hess

def simpleHessian(r):
    
    Hess = np.eye(3)*2
    return Hess

def saddleHessian(r):
    
    Hess = np.array([[2,0],[0,-2]])
    return Hess

def Gradient(r):
    x = r[0]
    y = r[1]
    z = r[2]

    z2=z**2
    y2=y**2
    x2=x**2

    grad = np.array([2*x*y2*z2, 2*x2*y*z2, 2*x2*y2*z])
    return grad

def simpleGradient(r):
    x = r[0]
    y = r[1]
    z = r[2]
    
    grad = np.array([2*x, 2*y, 2*z])
    return grad

def saddleGradient(r):
    x = r[0]
    y = r[1]
    
    grad = np.array([2*x, -2*y])
    return grad

def test_really_stupid_newton():
    mn = reallyStupidNewton()
    x = mn.minimize([np.array([10.,4.,3.])], simpleHessian, simpleGradient, stop = 1e-15)
    assert (x < 1e-4).all()
#test_really_stupid_newton()

def test_conditioned_really_stupid_newton():
    mn = reallyStupidNewton()
    x = mn.conditioned_minimize([np.array([3.,5.,4.])], simpleHessian, simpleGradient)
    assert (x[0] < 1e-4).all()
#test_conditioned_really_stupid_newton()

def _rebuild(dx,x,lin_x = None, mask = None):
    xk1 = dx + x[0]
    xk1 = [xk1]
    
    lin_x = xk1
    
    return xk1, lin_x

def test_trustR_really_stupid_newton():
    mn = reallyStupidNewton()
    x = mn.trustR_conditioned_minimize([np.array([3.,5.,4.])], simpleHessian, simpleGradient,_rebuild)
#    x = mn.trustR_conditioned_minimize([np.array([3.,5.,4.])], simpleHessian, simpleGradient)    
    assert (x[0][0] < 1e-4).all()
#test_trustR_really_stupid_newton()

def test_trustR_stupid_newton_saddle():
    mn = reallyStupidNewton()
    x = mn.trustR_conditioned_minimize([np.array([3.,5.])], saddleHessian, saddleGradient,_rebuild)
#    x = mn.trustR_conditioned_minimize([np.array([3.,5.,4.])], simpleHessian, simpleGradient)    
    assert (x[0][0] < 1e-4).all()
#test_trustR_stupid_newton_saddle()
    
    
def test_H2O_DM():
    mn = reallyStupidNewton()
    lf, lg, nbasis, wfn = _lg_init(get_water_sto3g_hf, 5)
    
    dm = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm)
    dm._array = np.array([[0.99502,-0.26321,0,0,0,0,0],
                          [0.020205,1.02905,0,0,0,0,0],
                          [0,0,0,0.5,0.5,0,0],
                          [0,0,0.5,0.5,0,0,0],
                          [0,0,0.5,0,0.5,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])
    
    
    dm2 = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm2); H0_inv = np.load("Hinv_exact_dm.npy")
    dm2._array = np.array([[0.99502,-0.26321,0,0,0,0,0],
                          [0.020205,1.02905,0,0,0,0,0],
                          [0,0,0,0.5,0.5,0,0],
                          [0,0,0.5,0.5,0,0,0],
                          [0,0,0.5,0,0.5,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]]); H0_inv = np.load("Hinv_new_promol_dm.npy"); H0 = np.load("H0_new_promol_dm.npy")
    
    b = lf.create_one_body(nbasis)
    b._array = np.array([[20.9936376792,3.9803152974,3.3990643654,3.3990643654,3.0922099245,0,0],
                          [3.9803152974,0.7546529148,0.6444499089,0.6444499089,0.586271453,0,0],
                          [3.3990643654,0.6444499089,0.55034,0.55034,0.5006574242,0,0],
                          [3.3990643654,0.6444499089,0.55034,0.55034,0.5006574242,0,0],
                          [3.0922099245,0.586271453,0.5006574242,0.5006574242,0.45546,0,0],
                          [0,0,0,0,0,0.30802,0],
                          [0,0,0,0,0,0,0.30802]])
#    b._array = -1*np.ones([nbasis,nbasis])

    b2 = lf.create_one_body(nbasis)
    b2._array = np.array([[20.9936376792,3.9803152974,3.3990643654,3.3990643654,3.0922099245,0,0],
                          [3.9803152974,0.7546529148,0.6444499089,0.6444499089,0.586271453,0,0],
                          [3.3990643654,0.6444499089,0.55034,0.55034,0.5006574242,0,0],
                          [3.3990643654,0.6444499089,0.55034,0.55034,0.5006574242,0,0],
                          [3.0922099245,0.586271453,0.5006574242,0.5006574242,0.45546,0,0],
                          [0,0,0,0,0,0.30802,0],
                          [0,0,0,0,0,0,0.30802]])
#    b2._array = -1*np.ones([nbasis,nbasis])

    mu = lf.create_one_body(1)
    mu._array = np.ones(1)*-0.45546
    mua = lf.create_one_body(1)
    mua._array = np.ones(1)*0.37726
    
    D = [dm,dm2]
    B = [b,b2]
    Mu = [mu,mua]
    
#    Dmask = np.ones(dm._array.size*4+2)
    #Dmask[0:97] = 0
    Dmask = None
    
#    lg.S._array = np.eye(lg.nbasis)
    
    H0 = lg.fdiff_ordered_lambda_hessian(D,B,Mu)
    np.save("H0_new_promol_dm", H0)
#    
#    step = 1e-2
#    step_range = np.arange(1e-5, 0.5, step)
#    fdiffs = []
#    norms = []
#    
#    mask = np.ones(H0.shape[0])
##    mask[0:196] = 0
#    
#    g0 = lg.fdiff_ordered_lambda_gradient(D,B,Mu)
#    for i in step_range:
#        dx_temp = np.ones(H0.shape[0])*i
#        dx_temp = dx_temp*mask
#        x1 = cp.deepcopy([D,B,Mu]) 
#        x1 = lg.rebuild(dx_temp, x1) 
#        g1 = lg.fdiff_ordered_lambda_gradient(*x1)
#        
#        fdiff_check = np.linalg.norm(np.dot(H0,dx_temp) - (g1-g0))
#        print("fdiff check", fdiff_check)
#        fdiffs.append(fdiff_check)
#        norms.append(np.linalg.norm(dx_temp))
#    
#    pyplot.plot(step_range, fdiffs)
##    pyplot.plot(norms,fdiffs)
#    pyplot.show()
    
    dx_temp = np.ones(H0.shape[0])*0.005
#    
    g0 = lg.fdiff_new_idem_gradient(D,B,Mu)
    x1 = cp.deepcopy([D,B,Mu]) 
    x1 = lg.rebuild(dx_temp, x1) 
    g1 = lg.fdiff_ordered_lambda_gradient(*x1)
    
    print("fdiff check", np.linalg.norm(np.dot(H0,dx_temp) - (g1-g0)))
    
    H0_inv = np.linalg.pinv(H0, 1e-4)
    np.save("Hinv_new_promol_dm", H0_inv)
    
    x = mn.line_SR1_minimize([D, B, Mu], lg.fdiff_ordered_lambda_gradient, H0_inv, lg.rebuild, 10e-5, Dmask, lg.grad_off, lg)
#    x,lin_x = mn.trustR_conditioned_minimize([D, B, Mu], lg.fdiff_ordered_lambda_hessian, lg.fdiff_ordered_lambda_gradient, lg.rebuild, 10e-5, Dmask, lg.grad_off, lg, 0.5)
    assert (abs(lg.energy(x[0]) + 84.212859533) < 1e-4).all()
    
test_H2O_DM()

def test_Be_DM():
    mn = reallyStupidNewton()
    lf, lg, nbasis, wfn = _lg_init(get_be_ss_hf, 2)
    
    dm = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm)
        
    dm2 = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm2)
    
#    dm._array[0,0] = dm._array[0,0]
#    dm2._array[0,0] = dm2._array[0,0]
    
    b = lf.create_one_body(nbasis)
    b._array = np.array([[4.2622083134,-0.1967473524],
                          [-0.1967473524,1.0830411716]])
    
    b2 = lf.create_one_body(nbasis)
    b2._array = np.array([[4.2622083134,-0.1967473524],
                          [-0.1967473524,1.0830411716]])
    
    mu = lf.create_one_body(1)
    mu._array = np.ones(1)*-0.1
    mua = lf.create_one_body(1)
    mua._array = np.ones(1)*-0.1
    
    D = [dm,dm2]
    B = [b,b2]
    Mu = [mu,mua]
    
    H0 = lg.fdiff_new_idem_hessian(D,B,Mu)
    
    dx_temp = np.ones(H0.shape[0])*0.005
    
    g0 = lg.fdiff_ordered_lambda_gradient(D,B,Mu)
    x1 = cp.deepcopy([D,B,Mu]) 
    x1 = lg.rebuild(dx_temp, x1) 
    g1 = lg.fdiff_ordered_lambda_gradient(*x1)
    
    print("fdiff check", np.linalg.norm(np.dot(H0,dx_temp) - (g1-g0)))
    
    H0_inv = np.linalg.pinv(H0)
    
#    H0 = 1*np.eye(lg.grad_off[-1])
#    H0[dm._array.size*2:, dm._array.size*2:] *= -1
#    H0_inv = np.zeros([lg.grad_off[-1],lg.grad_off[-1]])
#    H0_inv += 1/np.diag(H0)
    
    Dmask = None
    
    x = mn.line_SR1_minimize([D, B, Mu], lg.fdiff_new_idem_gradient, H0_inv, lg.rebuild, 10e-5, Dmask, lg.grad_off, lg)
#    x,lin_x = mn.trustR_conditioned_minimize([D, B, Mu], lg.fdiff_ordered_lambda_hessian, lg.fdiff_ordered_lambda_gradient, lg.rebuild, 10e-5, Dmask, lg.grad_off, lg)
    assert (abs(lg.energy(x[0]) - (-14.35188047620196)) < 1e-4).all()
    
#test_Be_DM()

def test_dummy_DM():
    mn = reallyStupidNewton()
    lf, lg, nbasis, wfn = _lg_init(get_be_ss_hf, 2)
    
    dm = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm)
    dm._array = np.ones(1)*5
        
    dm2 = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm2)
    dm2._array = np.ones(1)*5
    
    b = lf.create_one_body(nbasis)
    b._array = np.array([[4.2622083134,-0.1967473524],
                          [-0.1967473524,1.0830411716]])*3
    b._array = np.ones(1)*8
    
    b2 = lf.create_one_body(nbasis)
    b2._array = np.array([[4.2622083134,-0.1967473524],
                          [-0.1967473524,1.0830411716]])*3
    b2._array = np.ones(1)*8
    
    mu = lf.create_one_body(1)
    mu._array = np.ones(1)*-0.1
    mua = lf.create_one_body(1)
    mua._array = np.ones(1)*-0.1
    
    lg.nbasis = 1
    lg.recalc_offsets()
    lg.inputShape = (1)
    
    D = [dm,dm2]
    B = [b,b2]
    Mu = [mu,mua]
    
    H0 = lg.fdiff_dummy_hessian(D,B,Mu)
    
    dx_temp = np.ones(H0.shape[0])*0.005
    
    g0 = lg.fdiff_dummy_gradient(D,B,Mu)
    x1 = cp.deepcopy([D,B,Mu]) 
    x1 = lg.rebuild(dx_temp, x1) 
    g1 = lg.fdiff_dummy_gradient(*x1)
    
    print("fdiff check", np.linalg.norm(np.dot(H0,dx_temp) - (g1-g0)))
    
    H0_inv = np.linalg.pinv(H0)
    
#    H0 = 1*np.eye(lg.grad_off[-1])
#    H0[dm._array.size*2:, dm._array.size*2:] *= -1
#    H0_inv = np.zeros([lg.grad_off[-1],lg.grad_off[-1]])
#    H0_inv += 1/np.diag(H0)
    
    Dmask = None
    
    x = mn.line_SR1_minimize([D, B, Mu], lg.fdiff_dummy_gradient, H0_inv, lg.rebuild, 10e-5, Dmask, lg.grad_off, lg)
#    x,lin_x = mn.trustR_conditioned_minimize([D, B, Mu], lg.fdiff_dummy_hessian, lg.fdiff_dummy_gradient, lg.rebuild, 10e-5, Dmask, lg.grad_off, lg)
    assert abs(lg.dummy_lagrangian(*x)) < 1e-4
    
    print(lg.dummy_lagrangian(*x), np.linalg.norm(lg.fdiff_dummy_gradient(*x)))
#test_dummy_DM()

