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
    ref_result_dm_alpha = array([[  1.03160963e+00,  -4.82021015e-02,  -1.35898060e-06,
         -1.29120814e-06,   4.11472624e-07,  -1.05596879e-01,
         -8.13439192e-07,  -6.98049056e-07,   2.64411648e-07,
          1.77863514e-03,   3.95735356e-08,  -4.45376600e-07,
         -1.85543584e-07,   7.38517974e-08,  -2.90718174e-02,
         -4.66156768e-03,  -2.90734801e-02,  -4.65856816e-03,
         -2.90753521e-02,  -4.65807629e-03],
       [ -4.82021015e-02,   1.63497373e-01,   1.19576748e-06,
          2.92110147e-06,  -9.40976128e-07,   1.67146098e-01,
          6.97909294e-07,   1.79727535e-06,  -6.93149086e-07,
         -2.93856195e-03,  -1.47085170e-07,   9.90748518e-07,
          3.97215556e-07,  -2.51230386e-07,   5.55467351e-02,
          1.81748836e-02,   5.55522398e-02,   1.81702466e-02,
          5.55552970e-02,   1.81690024e-02],
       [ -1.35898060e-06,   1.19576748e-06,   2.00943845e-01,
         -3.91263367e-06,   9.97659624e-07,   5.49311348e-06,
          9.13680086e-02,   1.06099681e-05,  -9.30406732e-07,
          7.01656761e-07,   7.96220950e-07,  -7.98762104e-07,
          1.05534570e-02,   4.93754234e-06,   1.23446753e-01,
          1.03291057e-01,  -6.17323034e-02,  -5.16545834e-02,
         -6.17314941e-02,  -5.16301046e-02],
       [ -1.29120814e-06,   2.92110147e-06,  -3.91263367e-06,
          2.00940713e-01,   4.10248296e-06,   2.57088065e-06,
         -1.38338725e-05,   9.13652479e-02,   1.61383748e-06,
         -1.12142750e-06,  -5.13085230e-07,  -6.22015380e-07,
         -1.33621478e-06,  -1.05556582e-02,   7.58416082e-06,
          6.00805644e-06,   1.06912978e-01,   8.94426262e-02,
         -1.06924444e-01,  -8.94474392e-02],
       [  4.11472624e-07,  -9.40976128e-07,   9.97659624e-07,
          4.10248296e-06,   3.65778139e-01,  -1.29207739e-06,
          2.40419507e-06,   3.15326799e-06,   3.16397854e-01,
         -7.38821898e-07,   6.57040902e-07,  -1.50492640e-06,
          1.61502555e-06,  -1.55849117e-06,   8.96632099e-07,
         -4.02442913e-07,   6.23210069e-07,  -2.35333091e-06,
          3.96202245e-06,   2.01391016e-06],
       [ -1.05596879e-01,   1.67146098e-01,   5.49311348e-06,
          2.57088065e-06,  -1.29207739e-06,   1.73993846e-01,
          2.66366020e-06,   1.64724790e-06,  -9.91094739e-07,
         -3.05464177e-03,  -1.50140590e-07,   1.02132996e-06,
          6.31236254e-07,  -2.33652204e-07,   5.74917220e-02,
          1.85440859e-02,   5.74932339e-02,   1.85358094e-02,
          5.74968804e-02,   1.85349220e-02],
       [ -8.13439192e-07,   6.97909294e-07,   9.13680086e-02,
         -1.38338725e-05,   2.40419507e-06,   2.66366020e-06,
          4.15445078e-02,  -6.56878576e-07,   1.26430307e-06,
          3.16193332e-07,   3.62071225e-07,  -3.63162278e-07,
          4.79859616e-03,   2.87831907e-06,   5.61305812e-02,
          4.69658650e-02,  -2.80756822e-02,  -2.34923902e-02,
         -2.80624857e-02,  -2.34705278e-02],
       [ -6.98049056e-07,   1.79727535e-06,   1.06099681e-05,
          9.13652479e-02,   3.15326799e-06,   1.64724790e-06,
         -6.56878576e-07,   4.15426448e-02,   1.84773153e-06,
         -5.18263632e-07,  -2.33242504e-07,  -2.82874380e-07,
          4.31091443e-08,  -4.79952649e-03,   1.12184123e-05,
          9.15224782e-06,   4.86083571e-02,   4.06653194e-02,
         -4.86208645e-02,  -4.06737714e-02],
       [  2.64411648e-07,  -6.93149086e-07,  -9.30406732e-07,
          1.61383748e-06,   3.16397854e-01,  -9.91094739e-07,
          1.26430307e-06,   1.84773153e-06,   2.73683939e-01,
         -6.41297822e-07,   5.68337681e-07,  -1.30174650e-06,
          1.30281981e-06,  -1.24649838e-06,  -2.84497176e-07,
         -1.25664763e-06,   1.02331922e-07,  -2.42246013e-06,
          5.04942385e-06,   3.07750168e-06],
       [  1.77863514e-03,  -2.93856195e-03,   7.01656761e-07,
         -1.12142750e-06,  -7.38821898e-07,  -3.05464177e-03,
          3.16193332e-07,  -5.18263632e-07,  -6.41297822e-07,
          5.36332525e-05,   2.64442422e-09,  -1.79407766e-08,
          3.08289494e-08,   6.06667562e-08,  -1.00928877e-03,
         -3.25662919e-04,  -1.01062345e-03,  -3.26611988e-04,
         -1.00954217e-03,  -3.25638173e-04],
       [  3.95735356e-08,  -1.47085170e-07,   7.96220950e-07,
         -5.13085230e-07,   6.57040902e-07,  -1.50140590e-07,
          3.62071225e-07,  -2.33242504e-07,   5.68337681e-07,
          2.64442422e-09,   5.77757213e-12,  -5.17074013e-12,
          4.18224669e-08,   2.69694121e-08,   4.39199650e-07,
          3.92906904e-07,  -5.67518611e-07,  -4.49405699e-07,
         -2.15077100e-08,   7.47143263e-09],
       [ -4.45376600e-07,   9.90748518e-07,  -7.98762104e-07,
         -6.22015380e-07,  -1.50492640e-06,   1.02132996e-06,
         -3.63162278e-07,  -2.82874380e-07,  -1.30174650e-06,
         -1.79407766e-08,  -5.17074013e-12,   1.73191558e-11,
         -4.19514334e-08,   3.26613448e-08,  -1.52229880e-07,
         -3.00581259e-07,   2.52978861e-07,   3.84672940e-08,
          9.14935437e-07,   5.92116325e-07],
       [ -1.85543584e-07,   3.97215556e-07,   1.05534570e-02,
         -1.33621478e-06,   1.61502555e-06,   6.31236254e-07,
          4.79859616e-03,   4.31091443e-08,   1.30281981e-06,
          3.08289494e-08,   4.18224669e-08,  -4.19514334e-08,
          5.54261599e-04,   3.18709365e-07,   6.48346736e-03,
          5.42482494e-03,  -3.24263342e-03,  -2.71333564e-03,
         -3.24138757e-03,  -2.71104335e-03],
       [  7.38517974e-08,  -2.51230386e-07,   4.93754234e-06,
         -1.05556582e-02,  -1.55849117e-06,  -2.33652204e-07,
          2.87831907e-06,  -4.79952649e-03,  -1.24649838e-06,
          6.06667562e-08,   2.69694121e-08,   3.26613448e-08,
          3.18709365e-07,   5.54501588e-04,   2.47568299e-06,
          2.10589546e-06,  -5.61775444e-03,  -4.69975637e-03,
          5.61538340e-03,   4.69755520e-03],
       [ -2.90718174e-02,   5.55467351e-02,   1.23446753e-01,
          7.58416082e-06,   8.96632099e-07,   5.74917220e-02,
          5.61305812e-02,   1.12184123e-05,  -2.84497176e-07,
         -1.00928877e-03,   4.39199650e-07,  -1.52229880e-07,
          6.48346736e-03,   2.47568299e-06,   9.48670119e-02,
          6.96210283e-02,  -1.88878474e-02,  -2.55647086e-02,
         -1.88958666e-02,  -2.55580983e-02],
       [ -4.66156768e-03,   1.81748836e-02,   1.03291057e-01,
          6.00805644e-06,  -4.02442913e-07,   1.85440859e-02,
          4.69658650e-02,   9.15224782e-06,  -1.25664763e-06,
         -3.25662919e-04,   3.92906904e-07,  -3.00581259e-07,
          5.42482494e-03,   2.10589546e-06,   6.96210283e-02,
          5.51153654e-02,  -2.55616354e-02,  -2.45282465e-02,
         -2.55690690e-02,  -2.45226524e-02],
       [ -2.90734801e-02,   5.55522398e-02,  -6.17323034e-02,
          1.06912978e-01,   6.23210069e-07,   5.74932339e-02,
         -2.80756822e-02,   4.86083571e-02,   1.02331922e-07,
         -1.01062345e-03,  -5.67518611e-07,   2.52978861e-07,
         -3.24263342e-03,  -5.61775444e-03,  -1.88878474e-02,
         -2.55616354e-02,   9.48808263e-02,   6.96214762e-02,
         -1.88908614e-02,  -2.55651695e-02],
       [ -4.65856816e-03,   1.81702466e-02,  -5.16545834e-02,
          8.94426262e-02,  -2.35333091e-06,   1.85358094e-02,
         -2.34923902e-02,   4.06653194e-02,  -2.42246013e-06,
         -3.26611988e-04,  -4.49405699e-07,   3.84672940e-08,
         -2.71333564e-03,  -4.69975637e-03,  -2.55647086e-02,
         -2.45282465e-02,   6.96214762e-02,   5.51096819e-02,
         -2.55601801e-02,  -2.45230179e-02],
       [ -2.90753521e-02,   5.55552970e-02,  -6.17314941e-02,
         -1.06924444e-01,   3.96202245e-06,   5.74968804e-02,
         -2.80624857e-02,  -4.86208645e-02,   5.04942385e-06,
         -1.00954217e-03,  -2.15077100e-08,   9.14935437e-07,
         -3.24138757e-03,   5.61538340e-03,  -1.88958666e-02,
         -2.55690690e-02,  -1.88908614e-02,  -2.55601801e-02,
          9.48993151e-02,   6.96247015e-02],
       [ -4.65807629e-03,   1.81690024e-02,  -5.16301046e-02,
         -8.94474392e-02,   2.01391016e-06,   1.85349220e-02,
         -2.34705278e-02,  -4.06737714e-02,   3.07750168e-06,
         -3.25638173e-04,   7.47143263e-09,   5.92116325e-07,
         -2.71104335e-03,   4.69755520e-03,  -2.55580983e-02,
         -2.45226524e-02,  -2.55651695e-02,  -2.45230179e-02,
          6.96247015e-02,   5.51034762e-02]])
    ref_result_energy = -39.829627168151077
    ref_result_exp_alpha = array([[  9.95043945e-01,   2.03708558e-01,   9.88948164e-07,
          4.20728059e-06,   5.31947316e-07,  -1.52592652e-01,
         -9.03572488e-06,   7.06261462e-06,  -5.91620581e-07,
         -4.24719656e-08,   6.26498743e-07,   4.93256961e-02,
          4.94200409e-06,  -3.64166018e-07,   3.63188231e-02,
          1.91802958e-06,  -1.03016046e-06,  -4.11057092e-02,
         -1.29446478e-05,   1.35666214e-07],
       [  3.40433266e-02,  -4.02912436e-01,  -2.16595499e-06,
         -4.19119460e-06,  -1.33437130e-06,   2.39003502e-01,
          2.07078206e-05,  -6.52007639e-06,   1.58862746e-05,
          4.54849539e-05,   5.00019524e-06,   6.72365127e-01,
         -6.19739829e-04,   2.81545883e-04,  -1.94676273e+00,
          9.39807145e-06,  -2.60683445e-05,  -2.69444331e-01,
         -4.66931218e-05,  -3.96090139e-07],
       [  1.41023861e-07,   2.15054939e-06,  -1.08264343e-01,
         -4.34997340e-01,   2.22742287e-06,   3.91585015e-05,
         -3.24606872e-01,   2.74627496e-01,   2.13573656e-05,
          3.57987865e-01,  -7.37236537e-01,  -2.77143363e-06,
         -6.11291563e-01,  -4.80243965e-01,   1.34176863e-04,
          5.03996750e-06,   5.61414654e-06,  -7.42051300e-06,
         -5.77134311e-02,  -1.65763353e-02],
       [ -8.44126148e-08,  -6.04363581e-06,  -4.34991846e-01,
          1.08271959e-01,  -1.64793417e-06,  -1.27938496e-05,
          2.74606115e-01,   3.24599848e-01,   9.16975115e-05,
         -7.37208728e-01,  -3.57993185e-01,   1.08411407e-04,
          4.80278602e-01,  -6.11307245e-01,  -2.21239404e-04,
         -9.37013782e-06,   1.64361408e-06,   1.00084062e-05,
         -1.65649258e-02,   5.77526925e-02],
       [  2.01533131e-08,   3.24807736e-07,  -1.08490046e-05,
          3.50696797e-06,   6.04795964e-01,  -2.32905607e-06,
          3.70208450e-06,  -5.15998847e-06,   1.05587361e+00,
          8.80395944e-05,   9.14055595e-05,  -7.99833717e-06,
         -1.71576180e-05,  -9.50130034e-06,   5.30594689e-06,
         -4.38840417e-06,   4.57106363e-06,  -1.53994243e-06,
          2.69386782e-06,  -2.77017583e-06],
       [ -2.08341158e-02,  -4.16605109e-01,  -3.53458756e-06,
         -1.38117110e-05,  -1.83693427e-06,   1.88773109e+00,
          1.48824931e-04,  -6.94112096e-05,  -1.93578427e-05,
         -4.26273117e-05,   1.38853867e-05,  -1.04431463e+00,
          1.15349314e-03,  -5.25438270e-04,   3.57262201e+00,
         -2.79420865e-05,   5.45829051e-05,   7.86124736e-01,
          1.69273020e-04,   2.65976723e-06],
       [ -5.21888812e-08,   5.84441357e-07,  -4.92011038e-02,
         -1.97797305e-01,   4.23742358e-06,   9.96778435e-05,
         -1.00178759e+00,   8.47584269e-01,  -4.26880098e-05,
         -7.34471905e-01,   1.51251414e+00,   3.98601538e-05,
          8.88192757e-01,   6.97830125e-01,  -1.89604059e-04,
          1.23517523e-05,  -1.39450889e-05,   1.38692671e-04,
         -5.08316315e-01,  -1.45722718e-01],
       [  8.54509784e-08,  -3.90562820e-06,  -1.97792111e-01,
          4.92030867e-02,   1.38014251e-06,  -4.13505613e-05,
          8.47617379e-01,   1.00180496e+00,  -1.92399221e-04,
          1.51242910e+00,   7.34478785e-01,  -2.53153255e-04,
         -6.97928708e-01,   8.88219181e-01,   2.98701946e-04,
          2.98440696e-05,   1.50030519e-06,   4.78691728e-05,
         -1.45743538e-01,   5.08283275e-01],
       [ -1.50140574e-08,  -1.00811027e-08,  -4.23005515e-06,
          5.87226664e-06,   5.23148087e-01,   5.12774916e-06,
         -7.28866791e-06,  -1.42863165e-08,  -1.09861882e+00,
         -9.03112511e-05,  -9.55183796e-05,   3.88073158e-06,
          1.64433950e-05,   1.23720025e-05,  -7.51720203e-06,
          4.36704932e-06,  -1.64207444e-06,   1.15098690e-06,
          8.91381089e-07,   6.63624765e-07],
       [  2.89380597e-04,   7.31775369e-03,   1.96651203e-06,
         -2.06532788e-06,  -1.20972225e-06,   1.48411259e-02,
          2.73201686e-06,  -5.44669909e-06,  -2.04598443e-06,
         -1.78417235e-06,  -8.85980166e-06,  -1.83959733e-01,
          3.66547303e-05,   2.82225676e-06,  -3.07572686e-02,
          4.33833472e-05,  -4.84161984e-05,  -1.05116790e+00,
         -2.63195732e-04,   2.18974836e-05],
       [ -3.43839696e-08,   3.62718111e-07,   6.78723042e-07,
         -2.01195824e-06,   1.08824677e-06,  -8.43838331e-07,
         -1.61185182e-06,   2.80374810e-07,  -2.32570821e-06,
          2.93396828e-06,  -3.90399762e-06,   3.97516342e-06,
         -6.60915394e-06,   7.44106532e-06,   5.32476981e-06,
         -8.88208081e-01,  -4.59441403e-01,  -1.63496108e-05,
         -2.83499485e-05,   2.28052023e-05],
       [  5.48579708e-08,  -2.45426495e-06,   1.76498174e-06,
          1.39693549e-06,  -2.47933745e-06,   1.42161297e-06,
         -2.49691788e-07,  -3.26001099e-06,  -4.33459029e-06,
         -1.79774627e-06,   2.34869125e-06,   2.71348581e-06,
         -3.96124472e-06,   5.94863643e-06,  -4.15413184e-06,
         -4.59441400e-01,   8.88208081e-01,  -6.02142725e-05,
         -3.28008870e-05,   8.47160299e-06],
       [  6.16510511e-08,  -7.12566225e-07,  -5.68352800e-03,
         -2.28464253e-02,   2.70091776e-06,   2.96827822e-06,
          1.32774438e-02,  -1.12384296e-02,   8.61363455e-06,
          5.03177013e-02,  -1.03596050e-01,  -1.37677870e-05,
          1.54156408e-01,   1.21106900e-01,  -4.62929370e-05,
          3.50446717e-05,  -1.72591071e-05,   2.65300879e-04,
         -1.04303884e+00,  -2.99053389e-01],
       [ -3.84372873e-08,   5.57051015e-07,   2.28480935e-02,
         -5.69790625e-03,  -2.13399145e-06,  -3.02705553e-06,
          1.12490299e-02,   1.32849193e-02,  -1.44112864e-05,
          1.03622987e-01,   5.03074142e-02,   1.34993821e-05,
          1.21179669e-01,  -1.54166062e-01,  -4.79837962e-05,
         -3.95234247e-05,   2.20654769e-06,  -9.43809916e-05,
          2.99061499e-01,  -1.04302425e+00],
       [ -9.75429439e-04,  -1.37942320e-01,  -6.65305001e-02,
         -2.67229612e-01,   1.90301353e-06,  -1.00042858e-01,
          8.18580411e-02,  -6.92143586e-02,   3.40483829e-05,
          1.90787186e-01,  -3.92732002e-01,   6.36393732e-01,
          6.86949545e-01,   5.39655506e-01,   2.91789068e-01,
         -7.18778152e-07,  -3.13843127e-06,  -3.76540783e-01,
          6.54075986e-01,   1.87630853e-01],
       [  4.47311333e-03,  -4.47281763e-02,  -5.56678678e-02,
         -2.23597440e-01,  -3.67996350e-07,  -9.72145931e-01,
          1.27107798e+00,  -1.07549983e+00,  -1.73442457e-05,
          1.95055716e-01,  -4.01847210e-01,  -2.09008732e-01,
         -1.20724212e+00,  -9.48100913e-01,  -1.08215653e+00,
         -3.06925313e-07,  -7.79929268e-06,  -6.35525129e-02,
          1.06087578e-02,   2.98309108e-03],
       [ -9.75282186e-04,  -1.37960032e-01,  -1.98179564e-01,
          1.91237466e-01,  -3.56968040e-06,  -9.99943021e-02,
         -1.00844010e-01,  -3.62467069e-02,   2.42258193e-05,
         -4.35496622e-01,   3.11875671e-02,   6.36287628e-01,
         -8.10692123e-01,   3.25057901e-01,   2.92192199e-01,
          1.86712123e-05,  -2.32676994e-05,  -3.76403362e-01,
         -1.64746531e-01,  -6.60407688e-01],
       [  4.47295779e-03,  -4.47200712e-02,  -1.65791137e-01,
          1.60009605e-01,  -7.79757439e-06,  -9.71934382e-01,
         -1.56713630e+00,  -5.63082337e-01,   5.89742084e-05,
         -4.45456127e-01,   3.19543381e-02,  -2.08709920e-01,
          1.42433852e+00,  -5.71000854e-01,  -1.08286637e+00,
         -1.58403871e-05,  -8.73023106e-06,  -6.34988632e-02,
         -2.68232281e-03,  -1.06207089e-02],
       [ -9.75318460e-04,  -1.37968911e-01,   2.64732557e-01,
          7.60236775e-02,   1.09227230e-05,  -1.00042988e-01,
          1.90528263e-02,   1.05507963e-01,  -5.57030839e-05,
          2.44795011e-01,   3.61581654e-01,   6.36399537e-01,
          1.23876129e-01,  -8.64665291e-01,   2.91764915e-01,
          4.16520665e-05,  -2.17272400e-05,  -3.76280108e-01,
         -4.89659847e-01,   4.72789693e-01],
       [  4.47306098e-03,  -4.47180763e-02,   2.21454565e-01,
          6.35736368e-02,   6.92849175e-06,  -9.71966064e-01,
          2.95764523e-01,   1.63865653e+00,  -3.60222617e-05,
          2.50347517e-01,   3.69832223e-01,  -2.09139075e-01,
         -2.18048018e-01,   1.51948384e+00,  -1.08216348e+00,
          1.24313488e-05,  -7.29359462e-06,  -6.35044044e-02,
         -7.92498906e-03,   7.62748357e-03]])
    ref_result_exp_beta = array([[  9.95305322e-01,   1.94672063e-01,  -9.72679060e-07,
         -4.00330983e-06,   5.89569742e-07,  -1.60149163e-01,
          1.51971671e-05,  -1.43938662e-05,  -5.65871688e-07,
          8.10084460e-07,   5.49509622e-07,   5.98308742e-02,
          4.08552106e-07,  -1.68421524e-06,   3.89620735e-02,
         -1.61456224e-05,   6.19398406e-06,   3.25650022e-02,
         -2.19675856e-05,  -9.87040066e-07],
       [  3.22911509e-02,  -3.80261896e-01,   9.67143083e-07,
          8.13370256e-06,  -5.21484539e-07,   2.37219435e-01,
         -2.19681136e-05,   1.61972511e-05,  -4.31151540e-05,
         -5.76069529e-08,  -2.57472992e-05,   5.89200945e-01,
          6.02769350e-04,  -6.80604452e-05,  -1.97133014e+00,
         -5.10316511e-05,   1.93891284e-05,   3.16172723e-01,
         -1.75043770e-04,  -1.16219574e-05],
       [  3.21076478e-07,  -3.03442611e-06,   3.48686171e-02,
          4.26713385e-01,  -2.92100733e-06,   6.83736187e-05,
          2.83423277e-01,  -3.16396838e-01,  -3.29718510e-01,
         -7.34145624e-01,   6.34607846e-05,  -2.25185055e-05,
          5.76266425e-01,   5.60584831e-01,   1.62348743e-04,
          1.17731372e-05,  -1.30428261e-05,  -3.15136356e-05,
         -5.69469696e-02,  -1.79654233e-02],
       [ -6.66570082e-08,  -3.12503541e-06,   4.26720386e-01,
         -3.48837953e-02,   3.93376225e-05,  -1.82776970e-05,
         -3.16386176e-01,  -2.83421358e-01,   7.34126744e-01,
         -3.29721311e-01,  -2.91448381e-04,   1.22176345e-04,
         -5.60597529e-01,   5.76274000e-01,  -1.71539746e-04,
          9.59787108e-06,   7.07204411e-07,  -1.01490139e-05,
         -1.79531164e-02,   5.69827207e-02],
       [  3.94247862e-07,  -2.80589887e-06,  -4.73460408e-05,
          7.47746552e-06,  -5.03979500e-01,  -3.86251058e-06,
          5.97794774e-06,   1.49650746e-05,  -4.48063007e-04,
          1.36490165e-04,  -1.10754308e+00,   2.54330662e-07,
         -6.42570517e-05,   1.07180260e-04,   1.41305065e-05,
          7.38158318e-06,  -2.65392002e-05,  -3.82085512e-06,
          5.46041568e-06,  -2.06537703e-06],
       [ -1.96574609e-02,  -3.57256241e-01,   4.59227232e-06,
          7.04037611e-06,  -1.38923817e-05,   1.95053230e+00,
         -2.40552188e-04,   1.67499083e-04,   4.75992928e-05,
          2.43517207e-05,   4.84439948e-05,  -9.81137623e-01,
         -1.13028434e-03,   1.25850195e-04,   3.55909409e+00,
          2.05666033e-04,  -8.01330745e-05,  -8.04586383e-01,
          4.64134686e-04,   2.96409904e-05],
       [ -1.90429451e-07,   2.24499624e-06,   1.48380105e-02,
          1.81889727e-01,  -1.16451422e-05,   1.65000725e-04,
          8.91724130e-01,  -9.95533254e-01,   6.76697387e-01,
          1.50664376e+00,  -1.22764230e-04,   7.42677718e-05,
         -8.23122091e-01,  -8.00770606e-01,  -2.18168151e-04,
          9.11521942e-06,  -1.69452726e-05,  -3.04905094e-04,
         -5.04629476e-01,  -1.58971918e-01],
       [  1.11738692e-07,  -5.39405143e-06,   1.81885811e-01,
         -1.48366486e-02,   2.03898012e-05,  -4.92580288e-05,
         -9.95539386e-01,  -8.91728925e-01,  -1.50659771e+00,
          6.76698501e-01,   5.30480500e-04,  -2.68645444e-04,
          8.00842661e-01,  -8.23136523e-01,   2.20401899e-04,
         -5.80511448e-05,  -8.23098826e-05,  -9.70965533e-05,
         -1.58993558e-01,   5.04600490e-01],
       [ -1.72061985e-07,  -8.10255109e-07,  -1.18526166e-05,
         -7.15211171e-07,  -6.23051515e-01,  -4.88540507e-06,
         -6.58296459e-05,  -5.74773469e-05,   4.43430059e-04,
         -1.40044972e-04,   1.04520516e+00,  -2.35033186e-06,
          4.37213175e-05,  -8.69959804e-05,  -1.44100692e-05,
         -4.33447651e-06,   1.33981334e-05,   1.28509375e-06,
         -7.27152897e-08,  -3.86210474e-07],
       [ -6.51641969e-04,   3.12064762e-02,  -3.20949029e-06,
          5.30287968e-06,  -1.93782465e-06,  -1.24424779e-02,
          4.69222477e-06,   1.55067840e-07,  -2.80332485e-06,
         -1.34746255e-05,  -2.79714231e-06,  -1.56924739e-01,
         -3.56972086e-05,  -1.13910881e-07,   3.25544322e-03,
         -3.64160091e-04,   1.29760266e-04,   1.05558017e+00,
         -6.40051089e-04,  -7.40780414e-06],
       [ -9.44951713e-08,  -1.78920909e-06,  -3.67900213e-06,
          2.22694462e-06,  -9.53938142e-06,   2.05067540e-06,
          2.60422003e-06,  -1.67009021e-07,  -8.16191388e-06,
         -1.11339576e-06,  -1.68741254e-05,   5.13978342e-07,
          5.42637751e-06,  -2.27551283e-06,  -6.60472834e-07,
          2.74136801e-01,   9.61690693e-01,  -2.35393414e-05,
         -8.31817465e-05,   1.33246665e-04],
       [ -7.53442479e-07,   8.15581032e-06,  -2.68586786e-06,
         -2.76692227e-06,   5.20866441e-06,  -1.20305569e-05,
         -1.39339532e-06,   6.93703885e-06,  -2.90757319e-06,
          1.26119241e-06,   1.06077668e-05,   1.92587235e-05,
         -2.37583792e-06,  -1.37601384e-05,   3.41276982e-05,
          9.61690638e-01,  -2.74136782e-01,   3.67854268e-04,
          1.79950914e-05,   5.29918720e-05],
       [ -8.63160899e-08,   1.58708265e-06,   1.89302782e-03,
          2.31752796e-02,  -3.36819939e-06,  -1.44327751e-07,
         -9.79051996e-03,   1.09329333e-02,  -4.80621581e-02,
         -1.06996738e-01,  -1.45009267e-06,  -7.44874663e-06,
         -1.36116316e-01,  -1.32403459e-01,  -4.65900333e-05,
          2.09669584e-05,  -5.04637842e-05,  -6.37469136e-04,
         -1.03578548e+00,  -3.26328572e-01],
       [ -1.79788246e-07,   5.78901889e-07,  -2.31811543e-02,
          1.90738796e-03,   7.19545862e-06,  -3.08822532e-06,
         -1.09431740e-02,  -9.79607623e-03,  -1.07023328e-01,
          4.80531784e-02,   7.11521413e-05,   1.62233543e-05,
         -1.32478220e-01,   1.36117652e-01,  -3.10415770e-05,
          9.35247786e-05,   1.45235135e-04,   1.88907402e-04,
          3.26336053e-01,  -1.03577100e+00],
       [ -1.02000408e-03,  -1.53383234e-01,   2.35432362e-02,
          2.87825447e-01,   5.04204056e-06,  -8.15093203e-02,
         -6.27622903e-02,   7.00178757e-02,  -1.86962240e-01,
         -4.16131215e-01,   7.14795773e-06,   6.55593922e-01,
         -6.19382111e-01,  -6.02550100e-01,   2.75935493e-01,
         -1.67295333e-04,   8.32788666e-05,   3.53613473e-01,
          6.44637432e-01,   2.03221994e-01],
       [  4.22024359e-03,  -7.22810696e-02,   2.08580678e-02,
          2.54993845e-01,   1.91994087e-05,  -9.91853172e-01,
         -1.11812204e+00,   1.24838705e+00,  -1.66397913e-01,
         -3.70601479e-01,   5.40566962e-05,  -2.26810701e-01,
          1.09705946e+00,   1.06695410e+00,  -1.05810618e+00,
          8.05404629e-06,  -1.24376072e-06,   7.72681604e-02,
          1.28519496e-02,   4.00174476e-03],
       [ -1.01991220e-03,  -1.53399720e-01,   2.37507045e-01,
         -1.64302828e-01,  -2.72099102e-05,  -8.14483465e-02,
          9.19912977e-02,   1.93137615e-02,   4.53833382e-01,
          4.61963912e-02,  -2.43392881e-04,   6.55427242e-01,
          8.31439312e-01,  -2.35169096e-01,   2.76312204e-01,
         -8.66321515e-05,   1.11541776e-04,   3.53161995e-01,
         -1.46692655e-01,  -6.60149859e-01],
       [  4.21996763e-03,  -7.22698890e-02,   2.10380489e-01,
         -1.45562763e-01,  -7.06834431e-05,  -9.91504141e-01,
          1.64048275e+00,   3.44129403e-01,   4.04103066e-01,
          4.11535314e-02,  -7.19160727e-06,  -2.26408120e-01,
         -1.47213336e+00,   4.16353998e-01,  -1.05876579e+00,
          8.50593345e-06,   1.19263992e-05,   7.72345172e-02,
         -2.96857252e-03,  -1.31054290e-02],
       [ -1.01986334e-03,  -1.53411334e-01,  -2.61066441e-01,
         -1.23556863e-01,   1.86847145e-05,  -8.14984714e-02,
         -2.93005475e-02,  -8.93682538e-02,  -2.66950353e-01,
          3.69975519e-01,   2.47227007e-04,   6.55566845e-01,
         -2.12052166e-01,   8.37619408e-01,   2.75999831e-01,
         -1.77169412e-04,  -4.68693667e-05,   3.52943775e-01,
         -4.98632110e-01,   4.56908735e-01],
       [  4.22007814e-03,  -7.22704427e-02,  -2.31235454e-01,
         -1.09414448e-01,   7.91637673e-05,  -9.91557792e-01,
         -5.21933754e-01,  -1.59274828e+00,  -2.37664659e-01,
          3.29375338e-01,  -9.00773417e-05,  -2.26878846e-01,
          3.75919858e-01,  -1.48335276e+00,  -1.05826011e+00,
         -1.18566098e-05,  -2.23890635e-06,   7.72309017e-02,
         -9.97850631e-03,   9.08524781e-03]])
    ref_result_dm_beta = array([[  1.02852990e+00,  -4.18868067e-02,  -2.01363177e-06,
         -9.51366769e-07,  -1.50063242e-07,  -8.91129898e-02,
         -4.94833262e-07,  -1.05505332e-06,  -3.24535575e-07,
          5.42644629e-03,  -4.42286250e-07,   8.37829345e-07,
          1.28419481e-07,  -5.13440992e-08,  -3.08758203e-02,
         -9.87171632e-03,  -3.08773360e-02,  -9.86839719e-03,
         -3.08792270e-02,  -9.86810423e-03],
       [ -4.18868067e-02,   1.45641804e-01,   4.66886465e-06,
          1.31522888e-06,   1.06887833e-06,   1.35216173e-01,
          6.33635864e-07,   2.10819630e-06,   2.95408575e-07,
         -1.18876749e-02,   6.77136586e-07,  -3.12572830e-06,
         -4.15958987e-07,  -2.32697631e-07,   5.82952197e-02,
          2.76241073e-02,   5.82980205e-02,   2.76167753e-02,
          5.83022889e-02,   2.76168466e-02],
       [ -2.01363177e-06,   4.66886465e-06,   1.83300114e-01,
         -6.23737429e-06,   1.53836341e-06,   4.24375887e-06,
          7.81321791e-02,   1.10995573e-05,  -7.19353282e-07,
          2.05624270e-06,   8.17511470e-07,  -1.27435799e-06,
          9.95520661e-03,   5.61061733e-06,   1.23640344e-01,
          1.09536786e-01,  -6.18282013e-02,  -5.47776717e-02,
         -6.18259255e-02,  -5.47512476e-02],
       [ -9.51366769e-07,   1.31522888e-06,  -6.23737429e-06,
          1.83307145e-01,  -2.04645737e-05,   2.83921563e-06,
         -1.33330893e-05,   7.81319551e-02,  -5.03293521e-06,
         -1.65075261e-06,  -1.64758152e-06,  -1.05405933e-06,
         -6.45592419e-07,  -9.95840638e-03,   6.41315254e-06,
          5.63791293e-06,   1.07081069e-01,   9.48516363e-02,
         -1.07091758e-01,  -9.48558603e-02],
       [ -1.50063242e-07,   1.06887833e-06,   1.53836341e-06,
         -2.04645737e-05,   2.30545300e-09,   9.84410572e-07,
          6.56913382e-07,  -8.72259462e-06,   5.57977019e-10,
         -8.67413222e-08,   1.95726637e-10,   8.40340181e-11,
          8.35793537e-08,   1.11180841e-06,   1.46214808e-06,
          1.12068985e-06,  -1.20475399e-05,  -1.08464034e-05,
          1.18629064e-05,   1.03328892e-05],
       [ -8.91129898e-02,   1.35216173e-01,   4.24375887e-06,
          2.83921563e-06,   9.84410572e-07,   1.28018458e-01,
          5.50759592e-07,   2.65708250e-06,   2.86056340e-07,
         -1.11358994e-02,   6.40880592e-07,  -2.89896359e-06,
         -3.93356443e-07,  -2.96569146e-07,   5.48193068e-02,
          2.57418001e-02,   5.48229962e-02,   2.57358669e-02,
          5.48251344e-02,   2.57342722e-02],
       [ -4.94833262e-07,   6.33635864e-07,   7.81321791e-02,
         -1.33330893e-05,   6.56913382e-07,   5.50759592e-07,
          3.33040574e-02,   1.81401330e-07,  -3.06335984e-07,
          9.87235889e-07,   3.48556159e-07,  -5.43108110e-07,
          4.24343429e-03,   2.97144322e-06,   5.27014944e-02,
          4.66900992e-02,  -2.63612193e-02,  -2.33549141e-02,
         -2.63477775e-02,  -2.33326036e-02],
       [ -1.05505332e-06,   2.10819630e-06,   1.10995573e-05,
          7.81319551e-02,  -8.72259462e-06,   2.65708250e-06,
          1.81401330e-07,   3.33025894e-02,  -2.14526489e-06,
         -8.30148828e-07,  -7.02188616e-07,  -4.49406022e-07,
          4.72036882e-07,  -4.24462280e-03,   1.26358246e-05,
          1.09185193e-05,   4.56377011e-02,   4.04252873e-02,
         -4.56502937e-02,  -4.04347212e-02],
       [ -3.24535575e-07,   2.95408575e-07,  -7.19353282e-07,
         -5.03293521e-06,   5.57977019e-10,   2.86056340e-07,
         -3.06335984e-07,  -2.14526489e-06,   1.41666052e-10,
         -2.45524250e-08,   4.34596883e-11,   2.76163975e-11,
         -3.90617173e-08,   2.73399966e-07,  -3.63930054e-07,
         -3.73640081e-07,  -2.57576735e-06,  -2.33278210e-06,
          3.30465690e-06,   2.87580763e-06],
       [  5.42644629e-03,  -1.18876749e-02,   2.05624270e-06,
         -1.65075261e-06,  -8.67413222e-08,  -1.11358994e-02,
          9.87235889e-07,  -8.30148828e-07,  -2.45524250e-08,
          9.74268804e-04,  -5.57338559e-08,   2.55001507e-07,
          1.66417227e-07,   1.02615752e-07,  -4.78443446e-03,
         -2.25710231e-03,  -4.78803275e-03,  -2.25948543e-03,
         -4.78658057e-03,  -2.25789425e-03],
       [ -4.42286250e-07,   6.77136586e-07,   8.17511470e-07,
         -1.64758152e-06,   1.95726637e-10,   6.40880592e-07,
          3.48556159e-07,  -7.02188616e-07,   4.34596883e-11,
         -5.57338559e-08,   2.16626149e-11,  -1.07266438e-11,
          4.43994731e-08,   8.95298710e-08,   8.25787199e-07,
          6.17333731e-07,  -9.63687628e-07,  -9.67940384e-07,
          9.61322449e-07,   7.37263866e-07],
       [  8.37829345e-07,  -3.12572830e-06,  -1.27435799e-06,
         -1.05405933e-06,   8.40340181e-11,  -2.89896359e-06,
         -5.43108110e-07,  -4.49406022e-07,   2.76163975e-11,
          2.55001507e-07,  -1.07266438e-11,   8.20068782e-11,
         -6.91954847e-08,   5.72300066e-08,  -2.10983149e-06,
         -1.35426520e-06,  -1.43623642e-06,  -7.57196326e-07,
         -2.04781806e-07,   3.33496988e-07],
       [  1.28419481e-07,  -4.15958987e-07,   9.95520661e-03,
         -6.45592419e-07,   8.35793537e-08,  -3.93356443e-07,
          4.24343429e-03,   4.72036882e-07,  -3.90617173e-08,
          1.66417227e-07,   4.43994731e-08,  -6.91954847e-08,
          5.40676909e-04,   3.21387992e-07,   6.71475813e-03,
          5.94892177e-03,  -3.35839732e-03,  -2.97531418e-03,
         -3.35791524e-03,  -2.97356152e-03],
       [ -5.13440992e-08,  -2.32697631e-07,   5.61061733e-06,
         -9.95840638e-03,   1.11180841e-06,  -2.96569146e-07,
          2.97144322e-06,  -4.24462280e-03,   2.73399966e-07,
          1.02615752e-07,   8.95298710e-08,   5.72300066e-08,
          3.21387992e-07,   5.41003927e-04,   3.14501075e-06,
          2.81369902e-06,  -5.81916323e-03,  -5.15454793e-03,
          5.81606252e-03,   5.15156669e-03],
       [ -3.08758203e-02,   5.82952197e-02,   1.23640344e-01,
          6.41315254e-06,   1.46214808e-06,   5.48193068e-02,
          5.27014944e-02,   1.26358246e-05,  -3.63930054e-07,
         -4.78443446e-03,   8.25787199e-07,  -2.10983149e-06,
          6.71475813e-03,   3.14501075e-06,   1.06925220e-01,
          8.49671754e-02,  -1.81688607e-02,  -2.58629353e-02,
         -1.81773923e-02,  -2.58555258e-02],
       [ -9.87171632e-03,   2.76241073e-02,   1.09536786e-01,
          5.63791293e-06,   1.12068985e-06,   2.57418001e-02,
          4.66900992e-02,   1.09185193e-05,  -3.73640081e-07,
         -2.25710231e-03,   6.17333731e-07,  -1.35426520e-06,
          5.94892177e-03,   2.81369902e-06,   8.49671754e-02,
          7.06992748e-02,  -2.58586715e-02,  -2.74879129e-02,
         -2.58671502e-02,  -2.74815424e-02],
       [ -3.08773360e-02,   5.82980205e-02,  -6.18282013e-02,
          1.07081069e-01,  -1.20475399e-05,   5.48229962e-02,
         -2.63612193e-02,   4.56377011e-02,  -2.57576735e-06,
         -4.78803275e-03,  -9.63687628e-07,  -1.43623642e-06,
         -3.35839732e-03,  -5.81916323e-03,  -1.81688607e-02,
         -2.58586715e-02,   1.06937514e-01,   8.49650857e-02,
         -1.81700805e-02,  -2.58609836e-02],
       [ -9.86839719e-03,   2.76167753e-02,  -5.47776717e-02,
          9.48516363e-02,  -1.08464034e-05,   2.57358669e-02,
         -2.33549141e-02,   4.04252873e-02,  -2.33278210e-06,
         -2.25948543e-03,  -9.67940384e-07,  -7.57196326e-07,
         -2.97531418e-03,  -5.15454793e-03,  -2.58629353e-02,
         -2.74879129e-02,   8.49650857e-02,   7.06892002e-02,
         -2.58552876e-02,  -2.74799704e-02],
       [ -3.08792270e-02,   5.83022889e-02,  -6.18259255e-02,
         -1.07091758e-01,   1.18629064e-05,   5.48251344e-02,
         -2.63477775e-02,  -4.56502937e-02,   3.30465690e-06,
         -4.78658057e-03,   9.61322449e-07,  -2.04781806e-07,
         -3.35791524e-03,   5.81606252e-03,  -1.81773923e-02,
         -2.58671502e-02,  -1.81700805e-02,  -2.58552876e-02,
          1.06958068e-01,   8.49695266e-02],
       [ -9.86810423e-03,   2.76168466e-02,  -5.47512476e-02,
         -9.48558603e-02,   1.03328892e-05,   2.57342722e-02,
         -2.33326036e-02,  -4.04347212e-02,   2.87580763e-06,
         -2.25789425e-03,   7.37263866e-07,   3.33496988e-07,
         -2.97356152e-03,   5.15156669e-03,  -2.58555258e-02,
         -2.74815424e-02,  -2.58609836e-02,  -2.74799704e-02,
          8.49695266e-02,   7.06821837e-02]])

    thresholds = {'ref_result_exp_alpha': 1e-08, 'ref_result_dm_beta': 1e-08, 'ref_result_dm_alpha': 1e-08, 'ref_result_energy': 1e-08, 'ref_result_exp_beta': 1e-08}

    test_path = context.get_fn("examples/hf_dft/uks_methyl_hybgga.py")

    l = {}
    m = locals()
    with open(test_path) as fh:
        exec fh in l

    for k,v in thresholds.items():
        var_name = k.split("ref_")[1]
        assert allclose(l[var_name], m[k], v), m[k] - l[var_name]

if __name__ == "__main__":
    test_regression()
