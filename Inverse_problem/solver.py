import numpy as np
import matplotlib.pyplot as plt
import time

from linear_models.linear_models import linear
import utils.utils as utl



# IP SOLVER
def grad_f(x, b, op):
  # input: x- wavelet coeff.
  #        L- Lipchitz const.
  # out: grad_f(.) = At.A(.) - At(b)
  grad = op.At(op.A(x)) - op.At(b)
  return grad

class IP_solver:
    def __init__(self, reg_params, params):
        self.params = params
        self.reg_params = reg_params

    def run_solver(self):
        return self.pnp_CSALSA()

    def ISTA(self):
        """
            params - b, mode, oper, L, max_iters, lamda
            reg_params - lvl, wave 
        """
        import regularization.l1_shinkage as l1

        params = self.params
        FISTA_para = self.reg_params
        lin_op = linear(params.mode, params.oper)
        prox_op = l1.prox_FISTA(FISTA_para, params.lamda)
        wav_op = l1.wavelet_op(FISTA_para.lvl, FISTA_para.wave)
        xk = utl.wav_full_like(wav_op.wave_trasf(params.b), val=0)
        L = params.L

        func_vals = []

        for i in range(params.max_iter):
            xk = prox_op.prox(utl.wav_grad_step(xk, params.b, L, lin_op, wav_op), params.lamda/L)

            if i>10:
                func_val = lin_op.function_value(wav_op.inverse_wave_trasf(xk), params.b) + \
                    prox_op.function_value(xk)
                func_vals.append(func_val)

        out = wav_op.inverse_wave_trasf(xk)
        return np.real(out), func_vals

    def FISTA(self):
        """
            params - b, mode, oper, L, max_iters, lamda \\
            reg_params - lvl, wave 
        """
        import regularization.l1_shinkage as l1

        params = self.params
        FISTA_para = self.reg_params
        lin_op = linear(params.mode, params.oper)
        prox_op = l1.prox_FISTA(FISTA_para, params.lamda)
        wav_op = l1.wavelet_op(FISTA_para.lvl, FISTA_para.wave)
        xk_minus_1 = utl.wav_full_like(wav_op.wave_trasf(params.b), val=0)
        xk = yk = xk_minus_1
        tk = 1
        L = params.L

        func_vals = []
        for i in range(params.max_iter):

            xk = prox_op.prox(utl.wav_grad_step(yk, params.b, L, lin_op, wav_op), params.lamda/L)
            tk_plus_1 = (1 + np.sqrt(1 + 4 * (tk**2)))/2
            
            yk = utl.wav_add(1 + (tk - 1)/tk_plus_1, xk, -(tk - 1)/tk_plus_1, xk_minus_1)

            if i>10:
                func_val = lin_op.function_value(wav_op.inverse_wave_trasf(xk), params.b) + \
                    prox_op.function_value(xk)
                func_vals.append(func_val)

            # update next iteration variables
            xk_minus_1 = xk
            tk = tk_plus_1

        out = wav_op.inverse_wave_trasf(yk)
        return np.real(out), func_vals

    def FISTA_FGP(self):
        """
            params - b, mode, oper, L, max_iters \\
            reg_params - lamda, iters, l, u, proj_p 
        """
        from regularization.TV import prox_TV

        params = self.params
        TV_para = self.reg_params
        xk_minus_1 = np.zeros(params.b.shape)
        xk = yk = np.zeros(params.b.shape)
        tk = 1
        lin_op = linear(params.mode, params.oper)
        prox_op = prox_TV(TV_para)
        L = params.L

        func_vals = []
        for i in range(params.max_iter):
            xk = prox_op.prox( yk - (1/L)*grad_f(yk, params.b, lin_op) )
            tk_plus_1 = (1 + np.sqrt(1 + 4 * (tk**2)))/2

            yk = xk + ((tk - 1) / tk_plus_1) * (xk - xk_minus_1)

            if i>10:
                func_val = lin_op.function_value(xk, params.b) + prox_op.function_value(xk)
                func_vals.append(func_val)

            xk_minus_1 = xk
            tk = tk_plus_1

        return yk, func_vals

    def MFISTA(self):
        """
            params -image, b, max_iters,\
                    mode, oper,\
                    L\
            TV_para-proj_p, iters, \
                    lamda, l, u
        """
        import regularization.TV as tv
        
        params = self.params
        TV_para = self.reg_params
        lin_op = linear(params.mode, params.oper)
        prox_op = tv.prox_TV(TV_para)

        L = params.L
        xk_minus_1 = yk = np.random.normal(scale=0.1, size=params.b.shape)
        tk = 1

        for _ in range(params.max_iter):
            zk = prox_op.prox( yk - (2 / L) * grad_f(yk, params.b, lin_op) )
            tk_plus_1 = (1 + np.sqrt(1 + 4 * (tk**2)))/2
            xk = tv.check_F_val(zk, xk_minus_1, params.b, params.lamda, TV_para.proj_p, 
                                lin_op)
            yk_plus_1 = xk + (tk / tk_plus_1)*(zk - xk) + ((tk - 1) / tk_plus_1) * (xk - xk_minus_1)

            yk = yk_plus_1
            xk_minus_1 = xk
            tk = tk_plus_1

        return yk
