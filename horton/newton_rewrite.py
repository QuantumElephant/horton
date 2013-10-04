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

import scipy.optimize as op
import numpy as np

def solve(lg, x0):
#        jac = op.nonlin.BroydenFirst()
#        op.nonlin.KrylovJacobian
#        x_star= op.newton_krylov(lg.grad_wrap, x0, verbose=True, method='bicgstab', f_tol = 1e-8, callback = lg.callback_system)
#        x_star= op.newton_krylov(lg.grad_wrap, x0, verbose=True, method='lgmres', f_tol = 1e-8, callback = lg.callback_system, inner_M=jac)
    x_star = op.newton_krylov(lg.grad_wrap, x0, verbose=True, method='lgmres', f_tol = 1e-7, callback = lg.callback_system)
#         x_star= op.newton_krylov(lg.grad_wrap, x0, verbose=True, method='lgmres', rdiff = 1e-10, f_tol = 1e-6, callback = lg.callback_system)
    
#         x_star= op.broyden2(lg.grad_wrap, x0, verbose=True, f_tol = 1e-5, callback = lg.callback_system)
    return x_star, lg.nIter
    
