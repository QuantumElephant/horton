# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import sys

from numpy import array, allclose
from nose.plugins.attrib import attr

from horton import context


@attr('regression_check')
def test_regression():
    ref_result_dm_alpha = array([[  1.03044581e+00,  -3.95663809e-02,  -2.67585997e-07,
          3.83622472e-07,   2.83301314e-07,  -1.12441237e-01,
         -1.17673400e-06,   2.00598128e-06,   3.17439395e-07,
          2.14795764e-03,   8.34406230e-07,  -6.67832566e-07,
          2.98943704e-07,   8.17516951e-07,  -2.92075415e-02,
         -4.75672217e-03,  -2.92100171e-02,  -4.75658182e-03,
         -2.92082438e-02,  -4.75397515e-03],
       [ -3.95663809e-02,   1.56945587e-01,  -8.00639676e-07,
         -1.14277147e-06,  -5.78203766e-07,   1.64059848e-01,
          1.41486965e-06,  -4.38207551e-06,  -6.15999180e-07,
         -4.08277602e-03,  -1.63489105e-06,   1.27218306e-06,
         -6.31242215e-07,  -1.51223931e-06,   5.27683788e-02,
          1.98392675e-02,   5.27741541e-02,   1.98398120e-02,
          5.27715925e-02,   1.98352351e-02],
       [ -2.67585997e-07,  -8.00639676e-07,   2.06355717e-01,
          2.69084495e-06,  -1.27280071e-06,   1.90314231e-06,
          9.64084817e-02,  -6.77452592e-06,  -2.77181757e-07,
          9.32736842e-07,  -1.48631971e-06,  -3.14583795e-07,
          9.02746846e-03,  -2.59011158e-06,   1.20061349e-01,
          1.04331553e-01,  -6.00380314e-02,  -5.21547621e-02,
         -6.00351430e-02,  -5.21712245e-02],
       [  3.83622472e-07,  -1.14277147e-06,   2.69084495e-06,
          2.06357866e-01,  -1.76871364e-06,   7.28130560e-07,
         -6.08505131e-07,   9.64098902e-02,  -1.64026031e-06,
          4.93792080e-07,  -6.74378341e-07,   2.54355295e-06,
          3.69035084e-06,  -9.03071221e-03,  -5.18032212e-06,
          5.87654824e-06,   1.03983713e-01,   9.03441452e-02,
         -1.03981996e-01,  -9.03501926e-02],
       [  2.83301314e-07,  -5.78203766e-07,  -1.27280071e-06,
         -1.76871364e-06,   3.54696250e-01,  -9.52715719e-07,
         -1.19958173e-06,  -2.50824888e-06,   3.17262890e-01,
          4.89650566e-08,   1.00518509e-06,   1.70040727e-06,
         -2.63474432e-06,   1.44520676e-06,   2.64666542e-06,
          4.50972693e-07,   8.10449400e-06,  -4.64408866e-06,
         -5.14683826e-06,   2.93725404e-06],
       [ -1.12441237e-01,   1.64059848e-01,   1.90314231e-06,
          7.28130560e-07,  -9.52715719e-07,   1.76447794e-01,
          2.80098269e-06,  -3.74220235e-06,  -9.58180304e-07,
         -4.34576817e-03,  -1.73843829e-06,   1.35404671e-06,
         -5.51118553e-07,  -1.69564443e-06,   5.62697513e-02,
          2.07228877e-02,   5.62744688e-02,   2.07222291e-02,
          5.62697682e-02,   2.07156540e-02],
       [ -1.17673400e-06,   1.41486965e-06,   9.64084817e-02,
         -6.08505131e-07,  -1.19958173e-06,   2.80098269e-06,
          4.50416178e-02,  -4.03670009e-06,  -6.70590932e-07,
          3.88570730e-07,  -6.94416520e-07,  -1.46983354e-07,
          4.21759345e-03,  -1.12846356e-06,   5.60927438e-02,
          4.87434676e-02,  -2.80498329e-02,  -2.43670658e-02,
         -2.80466033e-02,  -2.43731234e-02],
       [  2.00598128e-06,  -4.38207551e-06,  -6.77452592e-06,
          9.64098902e-02,  -2.50824888e-06,  -3.74220235e-06,
         -4.03670009e-06,   4.50424653e-02,  -2.27076414e-06,
          3.31708115e-07,  -3.14974387e-07,   1.18831441e-06,
          1.37278690e-06,  -4.21912653e-03,  -8.40043229e-06,
         -1.80149171e-06,   4.85819657e-02,   4.22101074e-02,
         -4.85791048e-02,  -4.22098446e-02],
       [  3.17439395e-07,  -6.15999180e-07,  -2.77181757e-07,
         -1.64026031e-06,   3.17262890e-01,  -9.58180304e-07,
         -6.70590932e-07,  -2.27076414e-06,   2.83780112e-01,
          4.64146491e-08,   8.99096522e-07,   1.52094959e-06,
         -2.31900451e-06,   1.29522297e-06,   2.83462921e-06,
          8.26357004e-07,   6.93541185e-06,  -4.40962490e-06,
         -4.85873310e-06,   2.42252603e-06],
       [  2.14795764e-03,  -4.08277602e-03,   9.32736842e-07,
          4.93792080e-07,   4.89650566e-08,  -4.34576817e-03,
          3.88570730e-07,   3.31708115e-07,   4.64146491e-08,
          1.07435511e-04,   4.29848781e-08,  -3.34705454e-08,
          5.64980633e-08,   1.95028263e-08,  -1.38961870e-03,
         -5.15368194e-04,  -1.39033258e-03,  -5.15870866e-04,
         -1.39073226e-03,  -5.16156532e-04],
       [  8.34406230e-07,  -1.63489105e-06,  -1.48631971e-06,
         -6.74378341e-07,   1.00518509e-06,  -1.73843829e-06,
         -6.94416520e-07,  -3.14974387e-07,   8.99096522e-07,
          4.29848781e-08,   3.29632312e-11,  -1.46235313e-11,
         -6.50349145e-08,   2.95505440e-08,  -1.42100772e-06,
         -9.58049292e-07,  -4.63683386e-07,  -1.26167860e-07,
          2.15905639e-07,   4.64519401e-07],
       [ -6.67832566e-07,   1.27218306e-06,  -3.14583795e-07,
          2.54355295e-06,   1.70040727e-06,   1.35404671e-06,
         -1.46983354e-07,   1.18831441e-06,   1.52094959e-06,
         -3.34705454e-08,  -1.46235313e-11,   5.04137429e-11,
         -1.37353504e-08,  -1.11314428e-07,   2.50056107e-07,
          1.74698786e-09,   1.80647525e-06,   1.35381473e-06,
         -7.57006985e-07,  -8.73406637e-07],
       [  2.98943704e-07,  -6.31242215e-07,   9.02746846e-03,
          3.69035084e-06,  -2.63474432e-06,  -5.51118553e-07,
          4.21759345e-03,   1.37278690e-06,  -2.31900451e-06,
          5.64980633e-08,  -6.50349145e-08,  -1.37353504e-08,
          3.94925835e-04,  -2.69660852e-07,   5.25213521e-03,
          4.56412977e-03,  -2.62489378e-03,  -2.28013185e-03,
         -2.62836777e-03,  -2.28398038e-03],
       [  8.17516951e-07,  -1.51223931e-06,  -2.59011158e-06,
         -9.03071221e-03,   1.44520676e-06,  -1.69564443e-06,
         -1.12846356e-06,  -4.21912653e-03,   1.29522297e-06,
          1.95028263e-08,   2.95505440e-08,  -1.11314428e-07,
         -2.69660852e-07,   3.95205548e-04,  -1.74387473e-06,
         -1.70454782e-06,  -4.55038801e-03,  -3.95324771e-03,
          4.55068714e-03,   3.95436757e-03],
       [ -2.92075415e-02,   5.27683788e-02,   1.20061349e-01,
         -5.18032212e-06,   2.64666542e-06,   5.62697513e-02,
          5.60927438e-02,  -8.40043229e-06,   2.83462921e-06,
         -1.38961870e-03,  -1.42100772e-06,   2.50056107e-07,
          5.25213521e-03,  -1.74387473e-06,   8.78437956e-02,
          6.73686647e-02,  -1.69426348e-02,  -2.36806016e-02,
         -1.69354235e-02,  -2.36861720e-02],
       [ -4.75672217e-03,   1.98392675e-02,   1.04331553e-01,
          5.87654824e-06,   4.50972693e-07,   2.07228877e-02,
          4.87434676e-02,  -1.80149171e-06,   8.26357004e-07,
         -5.15368194e-04,  -9.58049292e-07,   1.74698786e-09,
          4.56412977e-03,  -1.70454782e-06,   6.73686647e-02,
          5.52570882e-02,  -2.36849353e-02,  -2.38589209e-02,
         -2.36884954e-02,  -2.38719033e-02],
       [ -2.92100171e-02,   5.27741541e-02,  -6.00380314e-02,
          1.03983713e-01,   8.10449400e-06,   5.62744688e-02,
         -2.80498329e-02,   4.85819657e-02,   6.93541185e-06,
         -1.39033258e-03,  -4.63683386e-07,   1.80647525e-06,
         -2.62489378e-03,  -4.55038801e-03,  -1.69426348e-02,
         -2.36849353e-02,   8.78597605e-02,   6.73668414e-02,
         -1.69370416e-02,  -2.36828166e-02],
       [ -4.75658182e-03,   1.98398120e-02,  -5.21547621e-02,
          9.03441452e-02,  -4.64408866e-06,   2.07222291e-02,
         -2.43670658e-02,   4.22101074e-02,  -4.40962490e-06,
         -5.15870866e-04,  -1.26167860e-07,   1.35381473e-06,
         -2.28013185e-03,  -3.95324771e-03,  -2.36806016e-02,
         -2.38589209e-02,   6.73668414e-02,   5.52433818e-02,
         -2.36831069e-02,  -2.38623267e-02],
       [ -2.92082438e-02,   5.27715925e-02,  -6.00351430e-02,
         -1.03981996e-01,  -5.14683826e-06,   5.62697682e-02,
         -2.80466033e-02,  -4.85791048e-02,  -4.85873310e-06,
         -1.39073226e-03,   2.15905639e-07,  -7.57006985e-07,
         -2.62836777e-03,   4.55068714e-03,  -1.69354235e-02,
         -2.36884954e-02,  -1.69370416e-02,  -2.36831069e-02,
          8.78522362e-02,   6.73694570e-02],
       [ -4.75397515e-03,   1.98352351e-02,  -5.21712245e-02,
         -9.03501926e-02,   2.93725404e-06,   2.07156540e-02,
         -2.43731234e-02,  -4.22098446e-02,   2.42252603e-06,
         -5.16156532e-04,   4.64519401e-07,  -8.73406637e-07,
         -2.28398038e-03,   3.95436757e-03,  -2.36861720e-02,
         -2.38719033e-02,  -2.36828166e-02,  -2.38623267e-02,
          6.73694570e-02,   5.52543994e-02]])
    ref_result_energy = -39.412999077525775
    ref_result_exp_alpha = array([[  9.93908969e-01,  -2.06375309e-01,   1.90959296e-06,
         -4.04566151e-06,   7.75992686e-07,  -1.56645877e-01,
         -7.77111245e-06,   2.19273734e-06,  -8.05095609e-07,
          9.60711898e-07,   1.41958565e-06,   4.66879978e-02,
          3.62437791e-06,   3.38119331e-07,   3.95180320e-02,
         -1.85722522e-06,   3.15726015e-06,   4.01338421e-02,
         -2.93080586e-06,  -2.66259267e-06],
       [  4.19872671e-02,   3.93932278e-01,  -3.49627984e-06,
          3.99935878e-06,  -1.50884953e-06,   2.62898388e-01,
          1.21622044e-05,  -1.03552748e-05,  -2.21701595e-05,
         -8.68113176e-06,   2.61863745e-07,   7.37489649e-01,
         -5.17731853e-04,   4.09214379e-05,  -1.92067321e+00,
         -2.97429135e-05,   3.21197414e-05,   2.76120973e-01,
         -1.17443535e-05,  -7.05186274e-06],
       [  1.12879112e-07,  -7.99896620e-06,  -2.10347839e-01,
          4.02628217e-01,   1.16535718e-05,   2.02488027e-05,
         -3.50315273e-01,  -2.80519343e-01,   8.94320581e-05,
          6.14404563e-01,  -5.84891982e-01,  -4.64291845e-05,
         -4.83806204e-01,   5.44143550e-01,   1.24165128e-04,
         -1.86060315e-06,   5.93952021e-06,   5.16065124e-06,
         -6.13250145e-02,  -6.58953798e-03],
       [  1.61281689e-07,  -1.48032574e-06,   4.02627570e-01,
          2.10354184e-01,  -1.30076517e-05,   2.97591069e-05,
         -2.80526297e-01,   3.50304840e-01,  -2.32036790e-05,
          5.84848269e-01,   6.14441781e-01,  -5.55304583e-05,
         -5.44181939e-01,  -4.83767511e-01,   1.13185922e-04,
          8.50330476e-06,  -6.53429966e-06,   7.35945370e-06,
         -6.60654327e-03,   6.13422786e-02],
       [ -1.08481854e-08,   8.14949164e-07,   2.00365172e-05,
         -9.93125065e-06,   5.95563778e-01,  -9.22515768e-07,
         -2.86628493e-06,  -1.68292022e-06,  -1.06110839e+00,
          6.15951559e-05,  -1.01556954e-04,  -2.69776117e-05,
         -7.57211997e-07,  -5.68849094e-06,   1.21022061e-06,
          4.92894399e-06,  -1.82031089e-06,  -1.75110835e-06,
         -3.10538163e-06,   2.01238881e-06],
       [ -2.60779258e-02,   4.19246626e-01,  -2.72183715e-06,
          1.16413153e-05,  -2.17289610e-06,   1.85133731e+00,
          1.32666482e-04,  -4.83080220e-05,   3.23413982e-05,
          2.55443400e-05,   2.20871528e-05,  -1.16011236e+00,
          9.59260622e-04,  -6.12888622e-05,   3.55522830e+00,
          7.40517645e-05,  -7.55403021e-05,  -7.87610209e-01,
          4.09116807e-05,   2.30111047e-05],
       [ -6.00009728e-08,   8.16114508e-07,  -9.82772282e-02,
          1.88104236e-01,   4.42884768e-06,   3.09121990e-05,
         -9.86894577e-01,  -7.90325048e-01,  -1.79983330e-04,
         -1.26080702e+00,   1.20023291e+00,   8.35100906e-05,
          7.25395522e-01,  -8.15799814e-01,  -1.74034723e-04,
          1.57745747e-05,  -3.49068865e-05,  -3.51720475e-05,
         -5.26681756e-01,  -5.66796570e-02],
       [ -1.10966838e-07,  -1.04401401e-05,   1.88114812e-01,
          9.82612942e-02,  -8.90173095e-06,   8.24147893e-05,
         -7.90320601e-01,   9.86902576e-01,   4.59058445e-05,
         -1.20013068e+00,  -1.26086827e+00,   1.77450026e-04,
          8.15969095e-01,   7.25270425e-01,  -1.32183244e-04,
         -5.06347459e-06,  -8.16340969e-06,   1.27990763e-05,
         -5.66637087e-02,   5.26669868e-01],
       [  2.36499178e-09,   4.76699885e-07,   1.69303860e-05,
         -7.26199586e-06,   5.32710188e-01,   2.97846885e-06,
          6.95909362e-07,   2.69592447e-06,   1.09401421e+00,
         -6.64176650e-05,   9.94903919e-05,   2.39524092e-05,
         -6.07128144e-06,  -1.08182899e-06,  -3.20504648e-06,
         -1.51286879e-06,   3.19122960e-06,   9.52780603e-07,
         -1.71041300e-06,   2.48668224e-06],
       [  8.90985991e-06,  -1.03651064e-02,   6.72207173e-08,
          2.14581311e-06,   9.65222378e-08,   1.20833644e-02,
          7.14761402e-08,   3.45673668e-06,   2.92220568e-06,
         -3.85508979e-07,  -1.19026089e-05,  -1.83273811e-01,
          2.53806377e-05,  -1.46769515e-05,  -3.42807828e-02,
         -7.81168705e-05,   6.26118443e-05,   1.05118847e+00,
         -5.68117926e-05,  -3.58871823e-05],
       [ -2.17500843e-08,  -4.14782004e-06,   1.99370352e-07,
         -3.58744690e-06,   1.68772346e-06,   1.89725747e-06,
         -1.11818203e-06,   3.17945492e-06,  -3.73627255e-06,
         -6.59578315e-06,  -7.25293973e-06,  -6.90356709e-06,
         -1.19943065e-05,  -2.26799579e-05,   5.48183130e-06,
         -7.75177188e-01,   6.31743867e-01,  -9.63244115e-05,
         -5.19734168e-05,   7.13731332e-06],
       [ -1.33001345e-09,   3.22962533e-06,   5.28361568e-06,
          1.97901171e-06,   2.85498380e-06,  -3.34350238e-06,
         -1.86227924e-06,  -1.91993696e-06,  -3.91368310e-09,
          1.92519159e-06,   1.08595133e-05,   2.23189172e-05,
          4.23160857e-07,   1.80886224e-05,   6.08115247e-06,
         -6.31743873e-01,  -7.75177191e-01,   3.38670253e-06,
          3.24600517e-05,  -1.12354855e-05],
       [  3.27193644e-09,  -1.86324553e-06,  -9.19514180e-03,
          1.76174670e-02,  -3.82081755e-06,   1.61789763e-06,
          1.64826832e-02,   1.31926241e-02,   1.31100617e-05,
          8.04462062e-02,  -7.65526027e-02,   1.02133902e-05,
          1.37150300e-01,  -1.54239907e-01,  -3.95489186e-05,
          2.07247233e-05,  -6.48104038e-05,  -6.80588826e-05,
         -1.07735773e+00,  -1.15942166e-01],
       [  8.54977808e-09,  -3.90258248e-06,  -1.76174221e-02,
         -9.21042460e-03,   2.86573952e-06,   7.30708005e-06,
         -1.32079841e-02,   1.64799945e-02,   3.08782899e-06,
         -7.65929634e-02,  -8.04389655e-02,  -2.71093472e-05,
         -1.54303327e-01,  -1.37157154e-01,   2.70477241e-05,
          1.83189408e-06,   1.42214079e-05,  -3.36440502e-05,
          1.15929829e-01,  -1.07734659e+00],
       [ -1.53803784e-03,   1.34113382e-01,  -1.22397910e-01,
          2.34251541e-01,   1.22844010e-05,  -1.03100666e-01,
          9.63847254e-02,   7.71420256e-02,   3.05068939e-05,
          2.89990793e-01,  -2.75940006e-01,   6.27274444e-01,
          5.89580589e-01,  -6.63159379e-01,   3.11951876e-01,
         -3.65848803e-05,   6.15357768e-05,   3.76180036e-01,
          6.82678184e-01,   7.34073688e-02],
       [  5.54892420e-03,   4.97676914e-02,  -1.06341401e-01,
          2.03570608e-01,   7.66119881e-06,  -9.59867354e-01,
          1.26943909e+00,   1.01671518e+00,   5.13168436e-05,
          3.85814838e-01,  -3.67429257e-01,  -1.72279377e-01,
         -1.02945100e+00,   1.15758489e+00,  -1.09915666e+00,
         -1.63296027e-05,   2.10360376e-05,   6.33971470e-02,
          4.93636182e-03,   5.63128087e-04],
       [ -1.53799721e-03,   1.34133960e-01,   2.64085740e-01,
         -1.11445022e-02,   4.35399327e-06,  -1.03109275e-01,
          1.86627877e-02,  -1.22042893e-01,  -4.52107289e-05,
          9.40500247e-02,   3.89076566e-01,   6.27170700e-01,
          2.79487261e-01,   8.42182162e-01,   3.12031873e-01,
         -1.97254095e-05,   6.42185480e-05,   3.76126275e-01,
         -2.77731524e-01,  -6.28027146e-01],
       [  5.54890797e-03,   4.97742203e-02,   2.29437699e-01,
         -9.66808206e-03,  -1.57462654e-05,  -9.59875568e-01,
          2.45599276e-01,  -1.60775073e+00,  -3.27118686e-05,
          1.25132741e-01,   5.17892780e-01,  -1.72226482e-01,
         -4.88202913e-01,  -1.47005179e+00,  -1.09933457e+00,
         -2.48299174e-06,  -2.90276276e-05,   6.33695974e-02,
         -2.03224404e-03,  -4.50054406e-03],
       [ -1.53794854e-03,   1.34126003e-01,  -1.41684650e-01,
         -2.23126754e-01,  -7.77963406e-06,  -1.03073454e-01,
         -1.15006538e-01,   4.48642236e-02,  -2.83801542e-05,
         -3.83989179e-01,  -1.13082609e-01,   6.27130994e-01,
         -8.69019666e-01,  -1.79012345e-01,   3.12310950e-01,
         -1.77372362e-06,  -8.95835911e-06,   3.76168992e-01,
         -4.05021863e-01,   5.54532175e-01],
       [  5.54878697e-03,   4.97613562e-02,  -1.23103095e-01,
         -1.93889291e-01,   5.77199375e-06,  -9.59716814e-01,
         -1.51528235e+00,   5.91153740e-01,  -8.22588152e-06,
         -5.11006182e-01,  -1.50543426e-01,  -1.71943425e-01,
          1.51690114e+00,   3.12484546e-01,  -1.09972977e+00,
         -2.45030301e-05,   1.64196836e-05,   6.33663066e-02,
         -2.90638609e-03,   3.96185353e-03]])
    ref_result_exp_beta = array([[  9.94135071e-01,  -1.99441151e-01,   2.15073058e-06,
         -4.27402644e-06,  -6.87426499e-07,   1.62541195e-01,
         -7.04493454e-06,  -1.96887744e-06,   1.34715384e-06,
          1.58127495e-06,   1.49493824e-06,   5.45697410e-02,
          6.80742007e-07,  -1.02284950e-06,   4.11051913e-02,
          1.14856309e-06,   2.55028729e-06,  -3.41681457e-02,
          3.51144800e-06,   2.85884143e-06],
       [  4.02170779e-02,   3.74495995e-01,  -3.87265587e-06,
          3.84764138e-06,   2.09424998e-06,  -2.54193884e-01,
          1.01927081e-05,   8.90781468e-06,  -5.98188332e-06,
         -1.59266952e-06,   2.81759514e-05,   6.36781151e-01,
          3.90406250e-04,  -3.06881435e-05,  -1.95644496e+00,
          3.55583197e-05,   3.88013704e-05,  -3.09086247e-01,
          2.26483213e-05,   1.04650223e-05],
       [  1.00790560e-07,  -9.87448487e-06,  -2.03690374e-01,
          3.81003772e-01,  -7.50531005e-06,  -2.00967492e-05,
         -3.63189885e-01,   2.54775644e-01,   6.02535368e-01,
         -5.79452932e-01,   2.58313881e-04,  -6.76007711e-05,
          5.22637719e-01,  -5.49728783e-01,   9.18088254e-05,
          2.35207482e-06,   6.11396727e-06,  -2.01739328e-06,
          5.97713089e-02,   1.06626393e-02],
       [  1.47242056e-07,  -2.65370722e-06,   3.81003991e-01,
          2.03696002e-01,   1.08255181e-05,  -2.74329188e-05,
         -2.54784840e-01,  -3.63179083e-01,   5.79406440e-01,
          6.02576384e-01,  -8.99380487e-05,  -6.56248280e-05,
          5.49769011e-01,   5.22598265e-01,   7.85174037e-05,
         -8.38692141e-06,  -6.58984077e-06,  -6.98467663e-06,
          1.06802231e-02,  -5.97868808e-02],
       [ -2.79113464e-09,   1.14106550e-06,   1.80396870e-05,
         -8.57414398e-06,  -5.45714577e-01,   1.95396364e-06,
         -3.45051731e-06,   2.28612981e-06,  -1.61721905e-04,
          3.12829199e-04,   1.08758573e+00,  -3.84080955e-05,
          2.11617913e-06,   4.63650988e-06,   2.16498206e-06,
         -4.36772645e-06,  -2.59831491e-06,   2.39110990e-06,
          4.09524007e-06,  -1.90372056e-06],
       [ -2.46075357e-02,   3.77502145e-01,  -2.92264536e-06,
          1.06521430e-05,  -1.23123736e-06,  -1.91245376e+00,
          1.23827905e-04,   5.05478256e-05,   2.20134056e-05,
          2.16060525e-05,  -4.28073611e-05,  -1.02980138e+00,
         -7.16866150e-04,   4.35795337e-05,   3.56427697e+00,
         -7.94537155e-05,  -8.37265191e-05,   8.02859160e-01,
         -6.53962434e-05,  -3.03968279e-05],
       [ -5.42637686e-08,   1.05950968e-06,  -9.11662468e-02,
          1.70517053e-01,   1.33354113e-06,  -3.29145316e-05,
         -1.06416658e+00,   7.46567312e-01,  -1.22755447e+00,
          1.18051349e+00,  -5.27256871e-04,   1.17471187e-04,
         -7.66467195e-01,   8.06137644e-01,  -1.23518247e-04,
         -2.08713442e-05,  -4.20914895e-05,   5.02704371e-05,
          5.21943328e-01,   9.31997937e-02],
       [ -1.04067368e-07,  -1.21675111e-05,   1.70530670e-01,
          9.11491022e-02,   6.19289592e-06,  -7.85244782e-05,
         -7.46561226e-01,  -1.06417369e+00,  -1.18040823e+00,
         -1.22762111e+00,   1.82028361e-04,   1.93091659e-04,
         -8.06309898e-01,  -7.66344729e-01,  -8.22440923e-05,
          5.14250455e-06,  -1.32221144e-05,  -1.20908549e-05,
          9.31848330e-02,  -5.21933154e-01],
       [ -6.48224762e-10,   5.90317598e-07,   1.64835284e-05,
         -6.45445920e-06,  -5.82871496e-01,  -2.47721191e-06,
         -9.99777321e-07,   1.27168079e-06,   1.55280483e-04,
         -3.13026047e-04,  -1.06813292e+00,   3.39117979e-05,
          5.30486164e-06,   2.05830645e-06,  -4.32703817e-06,
          1.27671318e-06,   3.42233344e-06,  -1.30678831e-06,
          1.79384929e-06,  -2.14950247e-06],
       [ -5.54680435e-04,  -2.54355475e-02,   1.67257164e-08,
          1.82076181e-06,  -4.53971610e-07,   7.17121424e-03,
         -7.89564800e-07,  -3.39634408e-06,  -8.47071110e-07,
         -1.10919761e-05,  -3.69051251e-06,  -1.64486367e-01,
         -3.23076158e-05,   1.39306200e-05,  -5.19882630e-03,
          7.89142659e-05,   6.82391071e-05,  -1.05462486e+00,
          8.69360049e-05,   4.52698146e-05],
       [ -4.27562482e-08,  -5.25871028e-06,   6.52194040e-08,
         -3.90711680e-06,  -1.57609959e-06,  -8.34124657e-07,
         -8.08109316e-07,  -3.36448635e-06,  -7.27084410e-06,
         -7.16360285e-06,   3.82587271e-06,  -5.60201782e-06,
          1.00830428e-05,   2.28989288e-05,   8.15273490e-06,
          7.69930303e-01,   6.38127976e-01,   9.98613128e-05,
          6.61901306e-05,  -5.31742770e-06],
       [  1.65593067e-08,   4.02235464e-06,   5.62221762e-06,
          1.98881339e-06,  -2.31779604e-06,   2.71303915e-06,
         -2.01797224e-06,   1.57729792e-06,   2.10102936e-06,
          1.10479525e-05,  -4.41643364e-07,   2.18170617e-05,
          4.20769677e-07,  -1.72647831e-05,   4.72796142e-06,
          6.38127983e-01,  -7.69930305e-01,  -5.57640837e-06,
         -3.80252913e-05,   1.48268593e-05],
       [  2.44884897e-09,  -2.10930430e-06,  -9.13667582e-03,
          1.71081741e-02,   4.36454476e-06,  -1.62800301e-06,
          1.47036331e-02,  -1.03093279e-02,   8.17758773e-02,
         -7.86118684e-02,   3.32811453e-05,   1.16141917e-05,
         -1.36988008e-01,   1.44075556e-01,  -2.87521340e-05,
         -2.88500841e-05,  -7.72749189e-05,   1.01444853e-04,
          1.06792399e+00,   1.90693594e-01],
       [  4.64947987e-09,  -4.49651525e-06,  -1.71081583e-02,
         -9.15303617e-03,  -2.80076281e-06,  -6.50917368e-06,
         -1.03237347e-02,  -1.47015945e-02,  -7.86542353e-02,
         -8.17684428e-02,   1.23500038e-05,  -2.88314644e-05,
          1.44137433e-01,   1.36997558e-01,   1.57147218e-05,
         -2.91342802e-06,   2.30340785e-05,   3.29627739e-05,
         -1.90680887e-01,   1.06791399e+00],
       [ -1.54419236e-03,   1.45092934e-01,  -1.30390010e-01,
          2.43866912e-01,  -1.34448871e-05,   9.02869457e-02,
          8.98276053e-02,  -6.29747660e-02,   3.04742021e-01,
         -2.92938998e-01,   1.50909817e-04,   6.48448395e-01,
         -6.05863482e-01,   6.37354862e-01,   2.87367145e-01,
          3.94269130e-05,   6.86107590e-05,  -3.58761913e-01,
         -6.70299382e-01,  -1.19630879e-01],
       [  5.23718645e-03,   7.07551545e-02,  -1.20804302e-01,
          2.25989903e-01,  -1.15572323e-05,   9.77068919e-01,
          1.34570918e+00,  -9.44196208e-01,   3.47046165e-01,
         -3.33902522e-01,   1.47590344e-04,  -2.12338859e-01,
          1.06333348e+00,  -1.11830459e+00,  -1.07500079e+00,
          1.94879674e-05,   2.43261364e-05,  -7.40284400e-02,
         -8.49870136e-03,  -1.54932209e-03],
       [ -1.54415350e-03,   1.45115354e-01,   2.76407339e-01,
         -9.02855744e-03,  -4.22515978e-06,   9.02944552e-02,
          9.67395109e-03,   1.09283408e-01,   1.01400257e-01,
          4.10341641e-01,  -8.27989526e-05,   6.48330988e-01,
         -2.48941339e-01,  -8.43424121e-01,   2.87426842e-01,
          1.44482128e-05,   6.38193057e-05,  -3.58676240e-01,
          2.31518116e-01,   6.40427220e-01],
       [  5.23717190e-03,   7.07645011e-02,   2.56080368e-01,
         -8.34939605e-03,   2.37376683e-05,   9.77071795e-01,
          1.44666167e-01,   1.63753551e+00,   1.15479743e-01,
          4.67564709e-01,  -1.26881512e-04,  -2.12250840e-01,
          4.37115391e-01,   1.47984667e+00,  -1.07513648e+00,
          4.77747050e-06,  -2.53956377e-05,  -7.40030333e-02,
          2.96578791e-03,   8.08611303e-03],
       [ -1.54410831e-03,   1.45106431e-01,  -1.46013621e-01,
         -2.34853605e-01,   8.88021921e-06,   9.02618485e-02,
         -9.94590052e-02,  -4.62682490e-02,  -4.06076805e-01,
         -1.17355484e-01,  -4.12921949e-06,   6.48283509e-01,
          8.54878147e-01,   2.06043021e-01,   2.87621347e-01,
         -1.32178109e-06,  -1.74400087e-05,  -3.58716364e-01,
          4.38881108e-01,  -5.20699195e-01],
       [  5.23705883e-03,   7.07496923e-02,  -1.35284212e-01,
         -2.17623438e-01,  -5.00902257e-06,   9.76925525e-01,
         -1.49060337e+00,  -6.93461931e-01,  -4.62592347e-01,
         -1.33735983e-01,  -3.81589845e-05,  -2.11944803e-01,
         -1.49996424e+00,  -3.61536434e-01,  -1.07539535e+00,
          2.52284840e-05,   1.72902506e-05,  -7.40003681e-02,
          5.54579935e-03,  -6.55915447e-03]])
    ref_result_dm_beta = array([[  1.02808132e+00,  -3.47087144e-02,   3.07461830e-09,
          6.24469141e-07,  -2.30240494e-07,  -9.97526813e-02,
         -1.19012460e-06,   2.30043835e-06,  -1.18284713e-07,
          4.52146799e-03,   1.00631464e-06,  -7.85757264e-07,
          3.30345279e-07,   9.03737796e-07,  -3.04739596e-02,
         -8.90624141e-03,  -3.04764366e-02,  -8.90630784e-03,
         -3.04745553e-02,  -8.90341403e-03],
       [ -3.47087144e-02,   1.41864690e-01,  -1.43911164e-06,
         -1.67962681e-06,   4.27030162e-07,   1.40383417e-01,
          1.40374526e-06,  -4.87056864e-06,   2.20906259e-07,
         -9.54781970e-03,  -1.97110099e-06,   1.50700766e-06,
         -6.88618275e-07,  -1.65270385e-06,   5.42760655e-02,
          2.67094797e-02,   5.42819147e-02,   2.67106182e-02,
          5.42793419e-02,   2.67057782e-02],
       [  3.07461830e-09,  -1.43911164e-06,   1.86653681e-01,
          2.09996765e-06,  -6.94130483e-06,   9.23700956e-07,
          8.35373412e-02,  -7.30412170e-06,  -5.81671593e-06,
          9.41416700e-07,  -1.50188198e-06,  -3.87485893e-07,
          8.37933324e-03,  -2.57410039e-06,   1.19471978e-01,
          1.10708974e-01,  -5.97428653e-02,  -5.53429523e-02,
         -5.97399768e-02,  -5.53599542e-02],
       [  6.24469141e-07,  -1.67962681e-06,   2.09996765e-06,
          1.86656141e-01,   5.12667190e-06,   5.08727281e-08,
         -1.06180435e-06,   8.35395877e-02,   4.96554137e-06,
          4.44673003e-07,  -7.71001279e-07,   2.54716745e-06,
          3.75673744e-06,  -8.38271491e-03,  -4.78439159e-06,
          6.13160130e-06,   1.03472839e-01,   9.58667112e-02,
         -1.03470905e-01,  -9.58730307e-02],
       [ -2.30240494e-07,   4.27030162e-07,  -6.94130483e-06,
          5.12667190e-06,   4.00247918e-10,   4.30599957e-07,
         -3.10664779e-06,   2.29478089e-06,   3.53372518e-10,
         -2.90320242e-08,   2.86782764e-11,   8.89589295e-11,
         -3.11513196e-07,  -2.30151540e-07,  -4.27761121e-06,
         -4.03623507e-06,   5.22927431e-06,   4.77191579e-06,
         -4.54821646e-07,  -4.93849555e-07],
       [ -9.97526813e-02,   1.40383417e-01,   9.23700956e-07,
          5.08727281e-08,   4.30599957e-07,   1.43113412e-01,
          2.48412146e-06,  -4.11816321e-06,   2.22693971e-07,
         -9.58832531e-03,  -1.98416423e-06,   1.51804480e-06,
         -5.87385318e-07,  -1.74505778e-06,   5.48138708e-02,
          2.65841032e-02,   5.48184506e-02,   2.65840342e-02,
          5.48139101e-02,   2.65773612e-02],
       [ -1.19012460e-06,   1.40374526e-06,   8.35373412e-02,
         -1.06180435e-06,  -3.10664779e-06,   2.48412146e-06,
          3.73873548e-02,  -4.16489495e-06,  -2.60333639e-06,
          2.82026681e-07,  -6.72191735e-07,  -1.73425600e-07,
          3.75019238e-03,  -1.06217665e-06,   5.34707901e-02,
          4.95484772e-02,  -2.67383895e-02,  -2.47695234e-02,
         -2.67348776e-02,  -2.47750766e-02],
       [  2.30043835e-06,  -4.87056864e-06,  -7.30412170e-06,
          8.35395877e-02,   2.29478089e-06,  -4.11816321e-06,
         -4.16489495e-06,   3.73888733e-02,   2.22262189e-06,
          4.78359019e-07,  -3.44944279e-07,   1.13998001e-06,
          1.31128684e-06,  -3.75175719e-03,  -9.01101104e-06,
         -2.92332414e-06,   4.63112195e-02,   4.29076513e-02,
         -4.63082624e-02,  -4.29071458e-02],
       [ -1.18284713e-07,   2.20906259e-07,  -5.81671593e-06,
          4.96554137e-06,   3.53372518e-10,   2.22693971e-07,
         -2.60333639e-06,   2.22262189e-06,   3.13715069e-10,
         -1.50227250e-08,   2.31905214e-11,   8.22105690e-11,
         -2.61029937e-07,  -2.22927578e-07,  -3.63768416e-06,
         -3.40816825e-06,   4.70008794e-06,   4.31675870e-06,
         -8.05326287e-07,  -7.83567481e-07],
       [  4.52146799e-03,  -9.54781970e-03,   9.41416700e-07,
          4.44673003e-07,  -2.90320242e-08,  -9.58832531e-03,
          2.82026681e-07,   4.78359019e-07,  -1.50227250e-08,
          6.47274820e-04,   1.33774789e-07,  -1.02316266e-07,
          8.46470240e-08,   9.74170347e-08,  -3.68921985e-03,
         -1.80219126e-03,  -3.69024379e-03,  -1.80284936e-03,
         -3.69043509e-03,  -1.80286021e-03],
       [  1.00631464e-06,  -1.97110099e-06,  -1.50188198e-06,
         -7.71001279e-07,   2.86782764e-11,  -1.98416423e-06,
         -6.72191735e-07,  -3.44944279e-07,   2.31905214e-11,
          1.33774789e-07,   4.29260474e-11,  -2.85568120e-11,
         -6.74294659e-08,   3.46698427e-08,  -1.72427078e-06,
         -1.26316590e-06,  -7.09743430e-07,  -3.23023542e-07,
          1.45078170e-07,   4.69187704e-07],
       [ -7.85757264e-07,   1.50700766e-06,  -3.87485893e-07,
          2.54716745e-06,   8.89589295e-11,   1.51804480e-06,
         -1.73425600e-07,   1.13998001e-06,   8.22105690e-11,
         -1.02316266e-07,  -2.85568120e-11,   5.17436967e-11,
         -1.73518969e-08,  -1.14406539e-07,   3.35514445e-07,
          5.49527171e-08,   2.11973316e-06,   1.70784876e-06,
         -7.04343751e-07,  -9.08730797e-07],
       [  3.30345279e-07,  -6.88618275e-07,   8.37933324e-03,
          3.75673744e-06,  -3.11513196e-07,  -5.87385318e-07,
          3.75019238e-03,   1.31128684e-06,  -2.61029937e-07,
          8.46470240e-08,  -6.74294659e-08,  -1.73518969e-08,
          3.76168524e-04,  -2.80030928e-07,   5.36314295e-03,
          4.96987469e-03,  -2.68021255e-03,  -2.48271526e-03,
         -2.68414340e-03,  -2.48724069e-03],
       [  9.03737796e-07,  -1.65270385e-06,  -2.57410039e-06,
         -8.38271491e-03,  -2.30151540e-07,  -1.74505778e-06,
         -1.06217665e-06,  -3.75175719e-03,  -2.22927578e-07,
          9.74170347e-08,   3.46698427e-08,  -1.14406539e-07,
         -2.80030928e-07,   3.76467225e-04,  -2.04215052e-06,
         -2.07275730e-06,  -4.64683446e-03,  -4.30495894e-03,
          4.64699534e-03,   4.30606044e-03],
       [ -3.04739596e-02,   5.42760655e-02,   1.19471978e-01,
         -4.78439159e-06,  -4.27761121e-06,   5.48138708e-02,
          5.34707901e-02,  -9.01101104e-06,  -3.63768416e-06,
         -3.68921985e-03,  -1.72427078e-06,   3.35514445e-07,
          5.36314295e-03,  -2.04215052e-06,   9.75269612e-02,
          8.11211025e-02,  -1.71849242e-02,  -2.51671173e-02,
         -1.71780026e-02,  -2.51742488e-02],
       [ -8.90624141e-03,   2.67094797e-02,   1.10708974e-01,
          6.13160130e-06,  -4.03623507e-06,   2.65841032e-02,
          4.95484772e-02,  -2.92332414e-06,  -3.40816825e-06,
         -1.80219126e-03,  -1.26316590e-06,   5.49527171e-08,
          4.96987469e-03,  -2.07275730e-06,   8.11211025e-02,
          7.06988124e-02,  -2.51719822e-02,  -2.77881001e-02,
         -2.51765246e-02,  -2.78044442e-02],
       [ -3.04764366e-02,   5.42819147e-02,  -5.97428653e-02,
          1.03472839e-01,   5.22927431e-06,   5.48184506e-02,
         -2.67383895e-02,   4.63112195e-02,   4.70008794e-06,
         -3.69024379e-03,  -7.09743430e-07,   2.11973316e-06,
         -2.68021255e-03,  -4.64683446e-03,  -1.71849242e-02,
         -2.51719822e-02,   9.75433739e-02,   8.11187875e-02,
         -1.71792909e-02,  -2.51699395e-02],
       [ -8.90630784e-03,   2.67106182e-02,  -5.53429523e-02,
          9.58667112e-02,   4.77191579e-06,   2.65840342e-02,
         -2.47695234e-02,   4.29076513e-02,   4.31675870e-06,
         -1.80284936e-03,  -3.23023542e-07,   1.70784876e-06,
         -2.48271526e-03,  -4.30495894e-03,  -2.51671173e-02,
         -2.77881001e-02,   8.11187875e-02,   7.06818866e-02,
         -2.51700347e-02,  -2.77926045e-02],
       [ -3.04745553e-02,   5.42793419e-02,  -5.97399768e-02,
         -1.03470905e-01,  -4.54821646e-07,   5.48139101e-02,
         -2.67348776e-02,  -4.63082624e-02,  -8.05326287e-07,
         -3.69043509e-03,   1.45078170e-07,  -7.04343751e-07,
         -2.68414340e-03,   4.64699534e-03,  -1.71780026e-02,
         -2.51765246e-02,  -1.71792909e-02,  -2.51700347e-02,
          9.75344454e-02,   8.11211183e-02],
       [ -8.90341403e-03,   2.67057782e-02,  -5.53599542e-02,
         -9.58730307e-02,  -4.93849555e-07,   2.65773612e-02,
         -2.47750766e-02,  -4.29071458e-02,  -7.83567481e-07,
         -1.80286021e-03,   4.69187704e-07,  -9.08730797e-07,
         -2.48724069e-03,   4.30606044e-03,  -2.51742488e-02,
         -2.78044442e-02,  -2.51699395e-02,  -2.77926045e-02,
          8.11211183e-02,   7.06947015e-02]])

    thresholds = {'ref_result_exp_alpha': 1e-08, 'ref_result_dm_beta': 1e-08, 'ref_result_dm_alpha': 1e-08, 'ref_result_energy': 1e-08, 'ref_result_exp_beta': 1e-08}

    test_path = context.get_fn("examples/hf_dft/uks_methyl_lda.py")

    l = {}
    m = locals()
    with open(test_path) as fh:
        exec fh in l

    for k,v in thresholds.items():
        var_name = k.split("ref_")[1]
        assert allclose(l[var_name], m[k], v), m[k] - l[var_name]

if __name__ == "__main__":
    test_regression()
