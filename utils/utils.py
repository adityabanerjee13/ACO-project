import numpy as np
import random
from regularization.l1_shinkage import wavelet_op
    
def TV_proj_p(p_q, proj_p):
    p = p_q[0]
    q = p_q[1]
        
    if proj_p == "isotropic":
        m, n = (q.shape[0], p.shape[1])
        mag_grad = np.sqrt(p[:m-1, :n-1]**2 + q[:m-1, :n-1]**2)
        p[:m-1, :n-1] = p[:m-1, :n-1]/np.maximum(1, mag_grad, dtype=float)
        p[:, n-1] = p[:, n-1] /np.maximum(1, np.abs(p[:, n-1]))
        q[:m-1, :n-1] = q[:m-1, :n-1]/np.maximum(1, mag_grad)
        q[m-1, :] = q[m-1, :] /np.maximum(1, np.abs(q[m-1, :]))
    
    elif proj_p == "l1-term":
        p = p / np.maximum(1, np.abs(p))
        q = q / np.maximum(1, np.abs(q))

    return (p,q)

def TV_proj_c(image, l, u):
    image[np.where(image<=l)] = l
    image[np.where(image>=u)] = u
    return image

def TV_innerprod(p_q, x):
    p_, q_ = grad(x)
    p, q = p_q
    return np.sum(p_*p) + np.sum(q_*q)

def TV_add(a, x, b, y):
    ax = a*x[0], a*x[1]
    by = b*y[0], b*y[1]
    return (ax[0]+by[0], ax[1]+by[1])

def grad(x, b, op):
    return op.At(op.A(x)) - 1 * op.At(b)

def grad_step(x, b, L, op):
    # input: x- wavelet coeff.
    #        L- Lipchitz const.
    # out: prox_{ lamda/L * ||.|| }( I - L^{-1} * grad_f )(x)
    #      grad_f(.) = At.A(.) - At(b)
    g = grad(x, b, op)
    grad_step = x - (1 / L) * g
    return grad_step

def wav_grad(x, b, op, wav_op):
    return wav_op.wave_trasf(op.At(op.A(wav_op.inverse_wave_trasf(x)) - b))

def wav_grad_step(x, b, L, op, wav_op):
    # input: x- wavelet coeff.
    #        L- Lipchitz const.
    # out: prox_{ lamda/L * ||.|| }( I - L^{-1} * grad_f )(x)
    #      grad_f(.) = At.A(.) - At(b)
    g = wav_grad(x, b, op, wav_op)
    grad_step = wav_add(1, x, -1*(1/L), g)
    return grad_step

def wav_add(a, x, b, y):
    # input: a:number, x:wave_coeff,
    #        b:number, y:wave_coeff
    # Outputs: a * x + b * y
    out = []
    count=0
    lvl = len(x)
    for i in range(lvl):
        if count==0:
            count=1
            out.append(a*x[0]+b*y[0])
        else:
            temp=[]
            for j in range(3):
                temp.append(a*x[i][j]+b*y[i][j])
            out.append(tuple(temp))
    return tuple(out)

def wav_innerproduct(x, y):
    out = 0
    count=0
    lvl = len(x)
    for i in range(lvl):
        if count==0:
            count=1
            out += np.sum(x[i]*y[i])
        else:
            for j in range(3):
                out += np.sum(x[i][j]*y[i][j])
    return out

def wav_l1_proj(x):
    count=0
    for cof in x:
        if count==0:
            count=1
            cof = cof / np.maximum(np.abs(cof), 1)
        else:
            for c in cof:
                c = c / np.maximum(np.abs(c), 1)
    return x

def wav_l1(x, normal = False):
    out = 0
    count=0
    lvl = len(x)
    for i in range(lvl):
        if count==0:
            count=1
            if not normal:
                out += np.sum(np.abs(x[i]))
            else:
                out += np.sum(np.abs(x[i]))/((x[i].shape[0])**2)
        else:
            for j in range(3):
                if not normal:
                    out += np.sum(np.abs(x[i][j]))
                else:   
                    out += np.sum(np.abs(x[i][j]))/((x[i][j].shape[0])**2)
    return out

def wav_l2(x):
    out = 0
    count=0
    lvl = len(x)
    for i in range(lvl):
        if count==0:
            count=1
            out += np.linalg.norm(x[i])**2
        else:
            for j in range(3):
                out += np.linalg.norm((x[i][j]))**2
    return out

def wav_full_like(x, val=0):
    out = []
    count=0
    for cof in x:
        if count==0:
            count=1
            out.append(np.full_like(cof, val))
        else:
            temp=[]
            for c in cof:
                temp.append(np.full_like(c, val))
            out.append(tuple(temp))
    return tuple(out)    

def wav_normal_like(x, std):
    out = []
    count=0
    for cof in x:
        if count==0:
            count=1
            out.append(np.random.normal(scale=std, size=cof.shape))
        else:
            temp=[]
            for c in cof:
                temp.append(np.random.normal(scale=std, size=c.shape))
            out.append(tuple(temp))
    return tuple(out)

def gaussian_kernel(size, sigma=1):
    # Create a grid of coordinates.
    x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    y = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    xv, yv = np.meshgrid(x, y)
    # Calculate the Gaussian function.
    kernel = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
    # Normalize the kernel.
    kernel = kernel / np.sum(kernel)
    return kernel

def random_mask(height, width, percent_of_holes):
    percentage = 100 - percent_of_holes
    num_pixels = height * width
    num_ones = int(num_pixels * percentage / 100)
    indices = np.random.choice(num_pixels, num_ones, replace=False)
    matrix = np.zeros((height, width), dtype=int)
    matrix.flat[indices] = 1
    return matrix

def star_mask(pixels=128, no_angles=22):   
    angles=np.linspace(0, 180, no_angles+1)[:no_angles]
    mask = np.zeros((pixels,pixels))
    for angle in angles:
        i_max = int(abs((pixels//2)*np.sqrt(2)*np.sin(angle/180*np.pi)))
        j_max = int(abs((pixels//2)*np.sqrt(2)*np.cos(angle/180*np.pi)))
        if i_max>j_max:
            o_max = abs((pixels//2)/np.sin(angle/180*np.pi))
        else:
            o_max = abs((pixels//2)/np.cos(angle/180*np.pi))

        d=0
        while d<int(o_max)+3:
            i_p = (pixels//2)+int(np.sin(angle/180*np.pi)*d)
            j_p = (pixels//2)+int(np.cos(angle/180*np.pi)*d)
            if pixels%2==0:
                i_n = (pixels//2)-int(np.sin(angle/180*np.pi)*d)-1
                j_n = (pixels//2)-int(np.cos(angle/180*np.pi)*d)-1
            else:
                i_n = (pixels//2)-int(np.sin(angle/180*np.pi)*d)
                j_n = (pixels//2)-int(np.cos(angle/180*np.pi)*d)
            
            if i_p>=pixels or j_p>=pixels or i_p<0 or j_p<0:
                pass
            else:
                if mask[i_p,j_p]==1:
                    pass
                else: 
                    mask[i_p, j_p] = 1
            if i_n>=pixels or j_n>=pixels or i_n<0 or j_n<0:
                pass
            else:
                if mask[i_n,j_n]==1:
                    pass
                else: 
                    mask[i_n, j_n] = 1
            if angle<45 or angle>135:
                d+=abs(1/np.cos(angle/180*np.pi))*0.85
            else:
                d+=abs(1/np.sin(angle/180*np.pi))*0.85
    return mask

def random_star_mask(samples_per_line, pixels=128, no_angles=22):   

    if samples_per_line%2!=0:
        samples_per_line+=1
    
    angles=np.linspace(0, 180, no_angles+1)[:no_angles]
    mask = np.zeros((pixels,pixels))
    for angle in angles:
        i_max = int(abs((pixels//2)*np.sqrt(2)*np.sin(angle/180*np.pi)))
        j_max = int(abs((pixels//2)*np.sqrt(2)*np.cos(angle/180*np.pi)))
        if i_max>j_max:
            o_max = abs((pixels//2)/np.sin(angle/180*np.pi))
        else:
            o_max = abs((pixels//2)/np.cos(angle/180*np.pi))

        if angle<45 or angle>135:
            d_arr = np.arange(0, int(o_max)+3, abs(1/np.cos(angle/180*np.pi))*0.85)
        else:
            d_arr = np.arange(0, int(o_max)+3, abs(1/np.sin(angle/180*np.pi))*0.85)
            
        if samples_per_line//2 >= len(d_arr):
            raise Exception(f"Excess value for variable samples_per_line.")

        d = random.sample(list(d_arr), samples_per_line//2)
        d_count = 0
        count = 0
        while count < samples_per_line:
            if d_count >= samples_per_line//2: 
                dist = random.sample(list(d_arr), 1)[0]
            else: dist = d[d_count]
            i_p = (pixels//2)+int(np.sin(angle/180*np.pi)*dist)
            j_p = (pixels//2)+int(np.cos(angle/180*np.pi)*dist)
            if pixels%2==0:
                i_n = (pixels//2)-int(np.sin(angle/180*np.pi)*dist)-1
                j_n = (pixels//2)-int(np.cos(angle/180*np.pi)*dist)-1
            else:
                i_n = (pixels//2)-int(np.sin(angle/180*np.pi)*dist)
                j_n = (pixels//2)-int(np.cos(angle/180*np.pi)*dist)
            
            if i_p>=pixels or j_p>=pixels or i_p<0 or j_p<0:
                pass
            else:
                if mask[i_p,j_p]==1:
                    pass
                else:
                    mask[i_p, j_p] = 1
                    count+=1
            if i_n>=pixels or j_n>=pixels or i_n<0 or j_n<0:
                pass
            else:
                if mask[i_n,j_n]==1:
                    pass
                else:
                    mask[i_n, j_n] = 1
                    count+=1
            d_count+=1
    if not np.array_equal(mask, np.flipud(np.fliplr(mask))):
        raise Exception(f"Error creating mask try different config.")
    return mask

def random_Rademacher_matrix(measurements, n):
    return np.random.choice([-1, 1], size=(measurements, n), p=[0.5, 0.5])

def random_Fourier_matrix(rows, cols, m):
    D = np.random.choice([-1, 1], size= (rows, cols), p= [0.5, 0.5])
    S = random_mask(rows, cols, 1 - m/(rows * cols))
    return D, S

def psnr(x, image, peak=1):
    mse = np.mean((x-image)**2)
    return round(10*np.log(peak**2/mse),2)

class reg_params:
    def __init__(self, image=None, lamda=0.007, iters=10, l=0, u=1, proj_p="l1-term"):
        self.image = image
        self.lamda = lamda
        self.iters = iters
        self.l = l
        self.u = u
        self.proj_p = proj_p          # "isotropic" , "l1-term"

class parameters:
    def __init__(self, b, image, mode, oper, radius=0, L = None, tol=None, n=None, max_iter=None, lamda=8e-2):
        self.lamda = lamda
        self.mode = mode
        self.oper = oper
        self.L = L
        self.max_iter = max_iter
        self.radius = 0
        self.n = n
        self.tol = tol
        self.b = b
        self.image = image

class SIP_parameters:
    def __init__(self, b, image, mode, oper, gamma, lamda, sigma, L, max_iters):
        self.mode = mode
        self.oper = oper
        self.L = L
        self.max_iters = max_iters
        self.b = b
        self.image = image
        self.gamma = gamma
        self.lamda = lamda
        self.sigma = sigma
