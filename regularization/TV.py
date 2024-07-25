import numpy as np
import utils.utils as utl

# TOTAL VARIANCE
def check_F_val(zk, xk_minus_1, b, lamda, proj_p, op):
    TV_op = L_image_grad()
    Ax = op.A(xk_minus_1)
    Az = op.A(zk)
    tv_x = TV(xk_minus_1, TV_op, proj_p)
    tv_z = TV(zk, TV_op, proj_p)
    Fx = 0.5 * np.linalg.norm(Ax - b)**2 + lamda * tv_x
    Fz = 0.5 * np.linalg.norm(Az - b)**2 + lamda * tv_z
    if Fx>Fz:
        return zk
    else:
        return xk_minus_1

def TV(x, TV_op, proj_p):
    shape = x.shape
    p, q = TV_op.L_T(x)
    if proj_p == "isotropic":
        mag_grad = np.sum(np.sqrt(p[:shape[0]-1, :shape[1]-1]**2 + q[:shape[0]-1, :shape[1]-1]**2))
        abs_val = np.sum(np.abs(p[:, shape[1]-1])) + np.sum(np.abs(q[shape[0]-1, :]))
        return mag_grad + abs_val

    elif proj_p == "l1-term":
        abs_val = np.sum(np.abs(p)) + np.sum(np.abs(q))
        return abs_val
    
class L_image_grad:
    def __init__(self):
        pass
        
    def L_T(self, image):
        shape = image.shape
        p = image[0:shape[0]-1] - image[1:shape[0]]
        q = image[:, 0:shape[1]-1] - image[:, 1:shape[1]]
        return (p, q)

    def L(self, p_q):
        p, q = p_q
        out_shape = (q.shape[0], p.shape[1])
        x = np.zeros(out_shape)
        x[0:out_shape[0]-1] += p
        x[1:out_shape[0]] -= p
        x[:, 0:out_shape[1]-1] += q
        x[:, 1:out_shape[1]] -= q
        return x
    
class prox_TV:
    def __init__(self, tv_params):
        self.params = tv_params

    def prox(self, x):
        return self.FGP(x, self.params)
        
    def GP(self, x, params, ops):
        img_shape = x.shape
        pk = np.zeros((img_shape[0]-1, img_shape[1]), float)
        qk = np.zeros((img_shape[0], img_shape[1]-1), float)
        
        for i in range(params.iters):
            Pc = utl.TV_proj_c(x - (params.lamda) * (ops.L((pk,qk))), params)
            temp_p, temp_q = ops.L_T(Pc)
            temp_p, temp_q = pk + (1 / (8 * params.lamda)) * temp_p , qk + (1 / (8 * params.lamda)) * temp_q
            pk, qk = utl.TV_proj_p((temp_p, temp_q), params, ops)

        x_star = utl.TV_proj_c(x - params.lamda * ops.L((pk,qk)), params)

        return x_star

    def FGP(self, x, params):
        """
        reg_params - lamda, iters, l, u, proj_p 
        """
        ops = L_image_grad()
        img_shape = x.shape
        pk = rk = np.zeros((img_shape[0]-1, img_shape[1]), float)
        qk = sk = np.zeros((img_shape[0], img_shape[1]-1), float)
        tk = 1
        
        for i in range(params.iters):
            Pc = utl.TV_proj_c(x - (params.lamda) * (ops.L((rk,sk))), params.l, params.u)
            temp_p, temp_q = ops.L_T(Pc)
            temp_p, temp_q = pk + (1 / (8 * params.lamda)) * temp_p , qk + (1 / (8 * params.lamda)) * temp_q
            pk, qk = utl.TV_proj_p((temp_p, temp_q), params.proj_p)

            tk_plus_1 = (1 + np.sqrt(1 + 4 * (tk**2))) / 2

            if i != 0:
                rk = pk + (tk - 1) / tk_plus_1 * (pk - pk_minus_1)
                sk = qk + ((tk - 1) / tk_plus_1) * (qk - qk_minus_1)
            else:
                rk, sk = pk, qk
            pk_minus_1 = pk
            qk_minus_1 = sk

        x_star = utl.TV_proj_c(x-params.lamda*ops.L((pk,qk)), params.l, params.u)
        
        return x_star

    def function_value(self, x):
        TV_op = L_image_grad()
        shape = x.shape
        p, q = TV_op.L_T(x)
        if self.params.proj_p == "isotropic":
            mag_grad = np.sum(np.sqrt(p[:shape[0]-1, :shape[1]-1]**2 + q[:shape[0]-1, :shape[1]-1]**2))
            abs_val = np.sum(np.abs(p[:, shape[1]-1])) + np.sum(np.abs(q[shape[0]-1, :]))
            return self.params.lamda * (mag_grad + abs_val)

        elif self.params.proj_p == "l1-term":
            abs_val = np.sum(np.abs(p)) + np.sum(np.abs(q))
            return self.params.lamda * abs_val
