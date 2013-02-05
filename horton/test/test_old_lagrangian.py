import numpy as np
from horton import *
from horton.newton import reallyStupidNewton as newt

np.set_printoptions(precision=8)


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

def _lg_init(ee_rep=None):
    lf, overlap, kinetic, nuclear_attraction, electronic_repulsion, wfn = get_water_sto3g_hf()
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

    lg = lagrangian(kinetic, [nuclear_attraction, nuclear_attraction], overlap, [5,5], Vee, W, debug = False) # TODO: check normalization

    return lf, lg, nbasis, wfn

def stupid_fn(*args):
    print("USING STUPID FUNCTION")
    result =0
    for i in args:
        for j in np.nditer(i._array):
            print(j)
            result += j**2
    return result

def test_calc_fdiff_hess(lf, lg, *args):
    h = 10e-4

#    fn = stupid_fn
    fn = lg.ugly_lagrangian

    result = []

    tmpFwdArgs = cp.deepcopy(args)
    tmpBackArgs = cp.deepcopy(args)
    tmpFwdBackArgs = cp.deepcopy(args)
    tmpBkFwdArgs = cp.deepcopy(args)

    tmpFwd = list(tmpFwdArgs)
    tmpBack = list(tmpBackArgs)
    tmpFwdBack = list(tmpFwdBackArgs)
    tmpBkFwd = list(tmpBkFwdArgs)

    for iKey,i in enumerate(args): # iterating over D, B, Mu
        inner_result = []
        for kKey,k in enumerate(args): # iterating over D, B, Mu again
            print(iKey, kKey)

            intermediate_result = np.zeros([i._array.size, k._array.size])

            it = np.nditer(intermediate_result, flags=['multi_index'])
            while not it.finished:
#                dim1_index = np.unravel_index(it.multi_index[0], k._array.shape)
#                dim2_index = np.unravel_index(it.multi_index[1], i._array.shape)

                if k._array.size == 1 and i._array.size == 1:
                    dim1_index = 0
                    dim2_index = 0
                elif k._array.size == 1:
                    dim1_index = 0
                    dim2_index = np.unravel_index(it.multi_index[0], i._array.shape)
                elif i._array.size == 1:
                    dim1_index = np.unravel_index(it.multi_index[1], k._array.shape)
                    dim2_index = 0
                else:
                    dim1_index = np.unravel_index(it.multi_index[0], k._array.shape)
                    dim2_index = np.unravel_index(it.multi_index[1], i._array.shape)

                tmpFwd[kKey]._array[dim1_index] += h
                tmpFwd[iKey]._array[dim2_index] += h

                tmpBack[kKey]._array[dim1_index] -= h
                tmpBack[iKey]._array[dim2_index] -= h

                tmpFwdBack[kKey]._array[dim1_index] += h
                tmpFwdBack[iKey]._array[dim2_index] -= h

                tmpBkFwd[kKey]._array[dim1_index] -= h
                tmpBkFwd[iKey]._array[dim2_index] += h

                intermediate_result[it.multi_index] = (fn(*tmpFwd) - fn(*tmpFwdBack) - fn(*tmpBkFwd) + fn(*tmpBack)) / (4*(h**2))

                tmpFwd[kKey]._array[dim1_index] -= h
                tmpFwd[iKey]._array[dim2_index] -= h

                tmpBack[kKey]._array[dim1_index] += h
                tmpBack[iKey]._array[dim2_index] += h

                tmpFwdBack[kKey]._array[dim1_index] -= h
                tmpFwdBack[iKey]._array[dim2_index] += h

                tmpBkFwd[kKey]._array[dim1_index] += h
                tmpBkFwd[iKey]._array[dim2_index] -= h

                #print(intermediate_result)
                #print(it.multi_index)

                it.iternext()
#            print(intermediate_result)
            inner_result.append(intermediate_result)
#            print(inner_result)
        result.append(np.hstack(inner_result))
    result = np.vstack(result)

#    print(result)
    return result

def calc_fdiff_gradient(lf, lg, nbasis, *args):
    h = 10e-4

#    fn = stupid_fn
    fn = lg.lagrangian

    result = []

    tmpFwdArgs = cp.deepcopy(args)
    tmpBackArgs = cp.deepcopy(args)

    tmpFwd = list(tmpFwdArgs)
    tmpBack = list(tmpBackArgs)

    for iKey,i in enumerate(args): # iterating over D, B, Mu
        for kKey,k in enumerate(i):
#            print(iKey, kKey)

            intermediate_result = np.zeros(k._array.size)

            it = np.nditer(intermediate_result, flags=['multi_index'])
            while not it.finished:
                if k._array.size == 1:
                    dim2_index = 0
                else:
                    dim2_index = np.unravel_index(it.multi_index[0], k._array.shape)

                tmpFwd[iKey][kKey]._array[dim2_index] += h
                tmpBack[iKey][kKey]._array[dim2_index] -= h

                intermediate_result[it.multi_index] = (fn(*tmpFwd) - fn(*tmpBack)) / (2*h)

                tmpFwd[iKey][kKey]._array[dim2_index] -= h
                tmpBack[iKey][kKey]._array[dim2_index] += h

                it.iternext()
    #            print(intermediate_result)
            result.append(intermediate_result)
    #            print(inner_result)

    result = np.hstack(result)

#    print(result)
    return result

def _verify(block, block2, index = None):
    condition = (np.abs(block - block2) > 10e-4).any()
    if condition:
        print ("MISMATCH MISMATCH MISMATCH")
#        print (abs(block - block2) > 10e-4)
        it = np.nditer(block, flags = ['multi_index'])
        while not it.finished:
            if index is None:
                slice_condition = True
            else:
                if ((it.multi_index[0] >= index[0] and index[1] > it.multi_index[0])
                        and (it.multi_index[1] >= index[2] and index[3] > it.multi_index[1])):
                    slice_condition = True
                else:
                    slice_condition = False

            if np.abs(block[it.multi_index] - block2[it.multi_index]) > 10e-4 and slice_condition:
                print(it.multi_index)
                print("block 1: ", block[it.multi_index])
                print("block 2: ", block2[it.multi_index])
                print("--pair end--")
            it.iternext()

#        print (block, block2)
    else:
        print("success")
    return condition

def calc_gradient():
    lf, lg = _lg_init()

    dm = lf.create_one_body(7)
    wfn.compute_density_matrix(dm)
    D = [dm,dm]
    b = lf.create_one_body(7)
    B = [b,b]
    Mu = [0,0]
    results = lg.gradient(D, B, Mu)
    print(results)
    return results

def calc_hessian():
    lf, lg = _lg_init()

    dm = lf.create_one_body(7)
    wfn.compute_density_matrix(dm)
    D = [dm,dm]
    b = lf.create_one_body(7)
    B = [b,b]
    Mu = [0,0]
    results = lg.hessian(D, B, Mu)
    print(results)
    return results

def test_pauli_constraint():
    init = _lg_init()
    lf = init[0]
    lg = init[1]

    dm = lf.create_one_body(7)
    b = lf.create_one_body(7)

    dm._array = np.ones([7,7])
    b._array = np.eye(7)

    lg.S._array = np.eye(7)

    test = np.sum(-1*np.ones([7,7])-np.eye(7))
    #assert (abs(lg._pauli_constraint(dm, b) - test) < 10e-4) #FIXME: needs a new test case
#test_pauli_constraint()

def test_calc_dLdDD():
    init = _lg_init()
    lf = init[0]
    lg = init[1]

    dm = lf.create_one_body(7)
    b = lf.create_one_body(7)

    dm._array = np.ones([7,7])
    b._array = np.eye(7)

    lg.S._array = np.eye(7)

    test = np.zeros([7,7,7,7])
    for i in np.arange(7):
        test[i,:,:,i]=-2*np.eye(7)

    result = lg._calc_dLdDD(np.zeros([7,7,7,7]), b._array)
    #assert(abs(test - result) < 10e-4).all()
#test_calc_dLdDD()

def test_calc_dLdDB():
    init = _lg_init()
    lf = init[0]
    lg = init[1]

    dm = lf.create_one_body(7)
    b = lf.create_one_body(7)

    dm._array = np.ones([7,7])
    b._array = np.eye(7)

    lg.S._array = np.eye(7)

    test = np.zeros([7,7,7,7])
    for i in np.arange(7):
        for j in np.arange(7):
            if i != j:
                test[:,i,j,:]=-2*(np.eye(7))

    result = lg._calc_dLdDB(np.zeros([7,7,7,7]),dm._array,b._array)
    assert(abs(test - result) < 10e-4).all()
#test_calc_dLdDB()

def test_calc_dLdBB():
    init = _lg_init()
    lf = init[0]
    lg = init[1]

    dm = lf.create_one_body(7)
    b = lf.create_one_body(7)

    dm._array = np.ones([7,7])
    b._array = np.eye(7)

    lg.S._array = np.eye(7)

    test = np.zeros([7,7,7,7])
    for i in np.arange(7):
        test[:i,i,:] = 6*np.ones([7,7])

    result = lg._calc_dLdDB(np.zeros([7,7,7,7]),dm._array,b._array)
    #assert(abs(test - result)< 10e-4).all() #TODO: fixme
#test_calc_dLdBB()

def test_hessian_assembly():
    ee_rep = np.zeros([7,7,7,7])

    lf, lg, nbasis = _lg_init(ee_rep)

    lg.T._array = np.eye(nbasis)
    lg.V[0]._array = np.eye(nbasis)
    lg.V[1]._array = np.eye(nbasis)
    lg.S._array = np.eye(nbasis)

    dm = lf.create_one_body(7)
    dm._array = np.ones([7,7])
    D = [dm,dm]
    b = lf.create_one_body(7)
    b._array = np.eye(nbasis)
    B = [b,b]
    Mu = [0,0]
    results = lg.hessian(D, B, Mu)
    print(results)
    test = 0

#    dLdDD = dLdDD.reshape([dLdDD.shape[0]**2, -1])
#    dLdDB = dLdDB.reshape([dLdDB.shape[0]**2, -1])
#    dLdBB = dLdBB.reshape([dLdBB.shape[0]**2, -1])

    assert (np.abs(results - test) < 10e-4).all()
    return results

#test_hessian_assembly()

def test_gradient_assembly():
    ee_rep = np.zeros([7,7,7,7])

    lf, lg, nbasis = _lg_init(ee_rep)

    lg.T._array = np.eye(nbasis)
    lg.V[0]._array = np.eye(nbasis)
    lg.V[1]._array = np.eye(nbasis)
    lg.S._array = np.eye(nbasis)

    dm = lf.create_one_body(7)
    dm._array = np.ones([7,7])
    D = [dm,dm]
    b = lf.create_one_body(7)
    b._array = np.ones([7,7])*2
    B = [b,b]
    Mu = [1,1]
    results = lg.gradient(D, B, Mu)
    assert (np.abs(results - np.hstack([np.eye(7).ravel(),np.eye(7).ravel(), np.zeros([7,7]).ravel(), np.zeros([7,7]).ravel(), [-4.0, -4.0]])) < 10e-4).all()

def test_lagrangian():
    lf, lg, nbasis, wfn = _lg_init()

    dm = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm)
    #dm._array = np.ones([nbasis,nbasis])

    dm2 = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm2)
    #dm._array = np.ones([nbasis,nbasis])


    b = lf.create_one_body(nbasis)
    b._array = np.zeros([nbasis,nbasis])

    b2 = lf.create_one_body(nbasis)
    b2._array = np.zeros([nbasis,nbasis])

    mu = lf.create_one_body(1)
    mu._array = np.zeros(1)

    mu2 = lf.create_one_body(1)
    mu2._array = np.zeros(1)

    D = [dm,dm2]
    B = [b,b2]
    Mu = [mu,mu2]

    result = lg.lagrangian(D, B, Mu)

    enn = 9.2535672047 # nucleus-nucleus interaction
    result += enn

    assert result - (-74.9592923284) < 1e-4

    return result

test_lagrangian()

def test_exact_DM():
    lf, lg, nbasis, wfn = _lg_init()

    dm = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm)


    b = lf.create_one_body(nbasis)
    b._array = np.zeros([nbasis,nbasis])

    D = [dm,dm]
    B = [b,b]
    Mu = [0,0]
    x0 = [D, B, Mu]

    nw = newton.reallyStupidNewton()

    final = nw.minimize(x0, lg.hessian, lg.gradient,lg.rebuild, 10e-4)
    return final

#test_exact_DM()

def test_stupid_hess():
    lf, lg, nbasis, wfn = _lg_init()
    dm = lf.create_one_body(3)
    dm._array = np.zeros([2,2])

    dm2 = lf.create_one_body(3)
    dm2._array = np.zeros([2,2])

    dm3 = lf.create_one_body(3)
    dm3._array = np.zeros([2,2])

    dm4 = lf.create_one_body(3)
    dm4._array = np.zeros([2,2])

    result = test_calc_fdiff_hess(dm,dm2,dm3,dm4)
    print(result)
#test_stupid_hess()

def test_fdiff_gradient():
    #lf, lg, nbasis, wfn = _lg_init(np.ones([7,7,7,7]))
    lf, lg, nbasis, wfn = _lg_init()

#    lg.S._array = np.eye(nbasis)
    #lg.V[0]._array = np.eye(nbasis)
    #lg.V[1]._array = np.eye(nbasis)
    #lg.T._array = np.eye(nbasis)

    dm = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm)
    #dm._array = np.ones([nbasis,nbasis])

    dm2 = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm2)
    #dm._array = np.ones([nbasis,nbasis])


    b = lf.create_one_body(nbasis)
#    b._array = np.zeros([nbasis,nbasis])
    b._array = np.ones([nbasis, nbasis])
#    b._array = np.eye(nbasis)

    b2 = lf.create_one_body(nbasis)
#    b2._array = np.zeros([nbasis,nbasis])
    b2._array = np.ones([nbasis,nbasis])
#    b2._array = np.eye(nbasis)

    mu = lf.create_one_body(1)
    mu._array = 3*np.ones(1)
#    mu._array = np.zeros(1)

    mu2 = lf.create_one_body(1)
    mu2._array = 3*np.ones(1)
#    mu2._array = np.zeros(1)

    D = [dm,dm2]
    B = [b,b2]
    Mu = [mu,mu2]

    #test, test_block = calc_fdiff_gradient(lf, lg, nbasis, dm, b, mu)
#    test = calc_fdiff_gradient(lf, lg, nbasis, D, B, Mu)

    a_grad = lg.fdiff_ordered_lambda_gradient(D, B, Mu)
    a_grad2 = lg.sym_gradient(D, B, Mu)

    #a_grad_block_formatted = np.vstack([a_grad_block[0], a_grad_block[2], a_grad_block[1], a_grad_block[3]])

    condition = _verify(a_grad2, a_grad)
    assert not condition
#    print("success")
#test_fdiff_gradient()

def test_min_fdiff_hess():
#    step = np.arange(2401)
#    step = np.reshape(step, [7,7,7,7])
    step = np.zeros([2,2,2,2])
#    step = 2
    lf, lg, nbasis, wfn = _lg_init(step)
#    lf, lg, nbasis, wfn = _lg_init()

    full_nbasis=7
    nbasis=2

    lg.S._array = np.eye(nbasis)
    lg.V[0]._array = np.eye(nbasis)
    lg.V[1]._array = np.eye(nbasis)
    lg.T._array = np.eye(nbasis)

#    lg.S._array = lg.S._array[0:nbasis, 0:nbasis]
#    lg.S._array[0,0] = 5
#    print(lg.S._array)
#    lg.V[0]._array = lg.V[0]._array[0:nbasis, 0:nbasis]
#    lg.V[1]._array = lg.V[1]._array[0:nbasis, 0:nbasis]
#    lg.T._array = lg.T._array[0:nbasis, 0:nbasis]

    dm = lf.create_one_body(full_nbasis)
    wfn.compute_density_matrix(dm)
    dm._array = dm._array[0:nbasis,0:nbasis]
#    dm._array = np.eye(nbasis)

#    print(dm._array)

    dm2 = lf.create_one_body(full_nbasis)
    wfn.compute_density_matrix(dm2)
    dm2._array = dm2._array[0:nbasis,0:nbasis]
#    dm2._array = np.eye(nbasis)

    b = lf.create_one_body(nbasis)
#    b._array = np.zeros([nbasis,nbasis])
    b._array = np.ones([nbasis,nbasis])

    b2 = lf.create_one_body(nbasis)
#    b2._array = np.zeros([nbasis,nbasis])
    b2._array = np.ones([nbasis,nbasis])

    mu=1
    mua=1

    mu2 = lf.create_one_body(1)
    mu2._array = np.zeros(1)
    mu2a = lf.create_one_body(1)
    mu2a._array = np.zeros(1)

#    test_block2 = test_calc_fdiff_hess(lf, lg, dm, dm2, b, b2, mu2, mu2a)

    D = [dm,dm2]
    B = [b,b2]
    Mu = [mu,mua]
    Mu2 = [mu2, mu2a]
    anly_hess = lg.hessian(D, B, Mu)
    test_block2 = lg.fdiff_sym_hessian(D, B, Mu2)
#    anly_hess2 = lg.sym_hessian(D, B, Mu)
#
    print(anly_hess[0:4, 8:12])
    print(test_block2[0:4, 8:12])
#
#    condition = _verify(anly_hess, test_block2, [0,8,8,16])
    condition = _verify(anly_hess, test_block2)
    assert not condition

#test_min_fdiff_hess()

def test_fdiff_hess():
#    step = np.arange(2401)
#    step = np.reshape(step, [7,7,7,7])
#    step = np.ones([7,7,7,7])
#    lf, lg, nbasis, wfn = _lg_init(step)
    lf, lg, nbasis, wfn = _lg_init()
#
#    lg.S._array = np.eye(nbasis)
#    lg.V[0]._array = np.eye(nbasis)
#    lg.V[1]._array = np.eye(nbasis)
#    lg.T._array = np.eye(nbasis)

    dm = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm)
#    dm._array = np.eye(nbasis)

    dm2 = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm2)
#    dm2._array = np.eye(nbasis)

    b = lf.create_one_body(nbasis)
    b._array = np.zeros([nbasis,nbasis])
    b._array = np.eye(nbasis)

    b2 = lf.create_one_body(nbasis)
    b2._array = np.zeros([nbasis,nbasis])
    b._array = np.eye(nbasis)

    mu=1
    mua=1

    mu2 = lf.create_one_body(1)
    mu2._array = np.zeros(1)
    mu2a = lf.create_one_body(1)
    mu2a._array = np.zeros(1)

#    test_block2 = test_calc_fdiff_hess(lf, lg, dm, dm2, b, b2, mu2, mu2a)

    D = [dm,dm2]
    B = [b,b2]
    Mu = [mu,mua]
    Mu2 = [mu2, mu2a]

    anly_hess = lg.hessian(D, B, Mu)
    test_block2 = lg.fdiff_sym_hessian(D, B, Mu2)
#
#    print(test_block)

    condition = _verify(anly_hess, test_block2)
    assert not condition
    print(test_block2)
#test_fdiff_hess()

#def test_fdiff_hess_slow():
##    step = np.arange(2401)
##    step = np.reshape(step, [7,7,7,7])
##    step = np.ones([7,7,7,7])
##    lf, lg, nbasis, wfn = _lg_init(step)
#    lf, lg, nbasis, wfn = _lg_init()
##
##    lg.S._array = np.eye(nbasis)
##    lg.V[0]._array = np.eye(nbasis)
##    lg.V[1]._array = np.eye(nbasis)
##    lg.T._array = np.eye(nbasis)
#
#    dm = lf.create_one_body(nbasis)
#    wfn.compute_density_matrix(dm)
##    dm._array = np.eye(nbasis)
#
#    dm2 = lf.create_one_body(nbasis)
#    wfn.compute_density_matrix(dm2)
##    dm2._array = np.eye(nbasis)
#
#    b = lf.create_one_body(nbasis)
#    b._array = np.zeros([nbasis,nbasis])
#    b._array = np.eye(nbasis)
#
#    b2 = lf.create_one_body(nbasis)
#    b2._array = np.zeros([nbasis,nbasis])
#    b._array = np.eye(nbasis)
#
#    mu2 = lf.create_one_body(1)
#    mu2._array = np.zeros(1)
#    mu2a = lf.create_one_body(1)
#    mu2a._array = np.zeros(1)
#
##    test_block2 = test_calc_fdiff_hess(lf, lg, dm, dm2, b, b2, mu2, mu2a)
#
#    D = [dm,dm2]
#    B = [b,b2]
#    Mu2 = [mu2, mu2a]
#
#    anly_hess = lg.fdiff_slow_sym_hessian(lg.lagrangian, D, B, Mu2)
#    test_block2 = lg.fdiff_ordered_sym_hessian(lg.lagrangian, D, B, Mu2)
##
##    print(test_block)
#
#    condition = _verify(anly_hess, test_block2)
#    assert not condition
#    print(test_block2)
#test_fdiff_hess_slow()

def test_new_fdiff_hess():
    #    step = np.arange(2401)
#    step = np.reshape(step, [7,7,7,7])
#    step = np.ones([7,7,7,7])
#    lf, lg, nbasis, wfn = _lg_init(step)
    lf, lg, nbasis, wfn = _lg_init()
#
#    lg.S._array = np.eye(nbasis)
#    lg.V[0]._array = np.eye(nbasis)
#    lg.V[1]._array = np.eye(nbasis)
#    lg.T._array = np.eye(nbasis)

    dm = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm)
#    dm._array = np.eye(nbasis)

    dm2 = lf.create_one_body(nbasis)
    wfn.compute_density_matrix(dm2)
#    dm2._array = np.eye(nbasis)

    b = lf.create_one_body(nbasis)
    b._array = np.zeros([nbasis,nbasis])
    b._array = np.eye(nbasis)

    b2 = lf.create_one_body(nbasis)
    b2._array = np.zeros([nbasis,nbasis])
    b._array = np.eye(nbasis)

    mu=lf.create_one_body(1)
    mu._array = np.zeros(1)
    mua=lf.create_one_body(1)
    mua._array = np.zeros(1)

    mu2 = lf.create_one_body(1)
    mu2._array = np.zeros(1)
    mu2a = lf.create_one_body(1)
    mu2a._array = np.zeros(1)

#    test_block2 = test_calc_fdiff_hess(lf, lg, dm, dm2, b, b2, mu2, mu2a)

    D = [dm,dm2]
    B = [b,b2]
    Mu = [mu,mua]
    Mu2 = [mu2, mu2a]

    test_block1 = lg.fdiff_ordered_lambda_hessian(D, B, Mu)
    test_block2 = lg.fdiff_sym_hessian(D, B, Mu2)
#
#    print(test_block)

    condition = _verify(test_block1, test_block2)
    assert not condition
    print("assert ok", test_block2)
    return

#test_new_fdiff_hess()

#def test_min_B():
#    lf, lg, nbasis, wfn = _lg_init()
#    nwtInst = newt()
#    
#    dm = lf.create_one_body(nbasis)
#    wfn.compute_density_matrix(dm)
#    dm._array = dm._array*0.9
#    
#    b = lf.create_one_body(nbasis)
#    b._array = np.ones([nbasis,nbasis])
#
#    print("idempotency check", lg._pauli_constraint(dm, b))
#    print("occupancy", nwtInst._check_occ([[dm,dm]], lg.S))
#
#    for i in (1e-4,1e-5,1e-6,1e-7,1e-8):
#        conditionNumber = i
#        print("condition number: ", conditionNumber)
#        
#        dm_orig = cp.deepcopy(dm)
#        b_orig = cp.deepcopy(b)
#    
#        print(lg._minimize_B(dm_orig,b_orig, conditionNumber))
#    
#test_min_B()

def test_rebuild():
    lf,lg,nbasis,wfn = _lg_init()
    
    lg.nbasis = 2
    lg.inputShape = [2,2]
    lg.recalc_offsets()
    
    D = [0,0]
    D[0] = lf.create_one_body(2)
    D[0]._array = np.zeros([2,2]) 
    D[1] = lf.create_one_body(2)
    D[1]._array = np.zeros([2,2])
    dD0 = np.ones([2,2])
    dD1 = np.ones([2,2])*2
    
    B = [0,0]
    B[0] = lf.create_one_body(2)
    B[0]._array = np.zeros([2,2])
    B[1] = lf.create_one_body(2)
    B[1]._array = np.zeros([2,2])
    dB0 = np.ones([2,2])*3
    dB1 = np.ones([2,2])*4
    
    Mu = [0,0]
    Mu[0] = lf.create_one_body(1)
    Mu[0]._array = 0
    Mu[1] = lf.create_one_body(1)
    Mu[1]._array = 0
    dMu0 = 5
    dMu1 = 6
    
    dx = np.hstack((dD0.ravel(),dD1.ravel(), dB0.ravel(), dB1.ravel(), dMu0, dMu1))
    
    x,lin_x = lg.rebuild(dx, (D, B, Mu))
    
    counter = 1
    for i in (x[0],x[1]):
        for j in i:
            print(np.abs(j._array - np.ones([2,2])*counter))
            assert (np.abs(j._array - np.ones([2,2])*counter) < 10e-8).all()
            counter += 1
#test_rebuild()
     
