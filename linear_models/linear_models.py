import numpy as np
from scipy.signal import convolve2d

import pywt

def Downsample(img, block_size):
    return img[::block_size, ::block_size]

def Upsample(img, block_size):
    return np.kron(img, np.ones((block_size,block_size)))

def Upsample_transpose(img, block_size):
    rows, cols = img.shape
    reduced_rows = rows // block_size
    reduced_cols = cols // block_size
    block_shape = (reduced_rows, block_size, reduced_cols, block_size)
    blocks = img[:reduced_rows*block_size, :reduced_cols*block_size].reshape(block_shape)
    reduced_image = np.sum(blocks, axis=(1, 3))
    return reduced_image

def Downsample_transpose(image, block_size):
    temp = np.zeros((block_size, block_size), float)
    temp[0,0] = 1
    return np.kron(image, temp)

# LINEAR OPERATOR MODEL
class wavelet_op:
    def __init__(self, lvl = 0, wave='haar'):
        self.lvl = lvl
        self.wave = wave

    def wave_trasf(self, img):
        coeffs = pywt.wavedec2(img, self.wave, level=self.lvl)
        return coeffs
    def inverse_wave_trasf(self, coeffs):
        img = pywt.waverec2(coeffs, self.wave)
        return img
    def shrink(self, coff, t):
        out = []
        count=0
        for cof in coff:
            if count==0:
                count=1
                out.append(cof)
                # out.append(pywt.threshold(cof, t, 'soft'))
            else:
                temp=[]
                for c in cof:
                    temp.append(pywt.threshold(c, t, 'soft'))
                out.append(tuple(temp))
        return tuple(out)
    
class linear_op:
    def __init__(self, mode=None, oper=None):
        # Mode:- "blur" - input:  A = kernel
        #                         At = kernel.lr.ud
        #        "supr" - input:  A = kron-factor
        #                         At = downsample
        #        "inpnt" - input: A = mask
        #                         At = mask
        #        "cs-R"  - input: A = rademacher            
        #                         At = rademacher           
        #        "cs-f"  - input: A = randomFourier
        #                         At = randomFourier
        #        "I"     - input: A = identity
        #                         At = identity
        self.mode = mode
        if mode == "blur":
            self.operator = oper
            self.size = oper.shape[0]
        elif mode == "supr":
            self.operator = oper["operator"]
            self.res = oper["res"]
            self.size = oper["operator"].shape[0]
        elif mode == "inpnt":
            self.operator = oper
        elif mode == "cs-R":
            m = oper.shape[0]
            self.I_plus_AAt_inv = np.linalg.inv(np.eye(m) + oper @ oper.T)
            n = int(np.sqrt(oper.shape[1]))
            self.img_shape = n
            self.operator = oper
        elif mode == "cs-sf":
            self.S = oper["S"] 
            self.m = oper["m"]
            self.n = self.S.shape[0]**2

        elif mode == "I":
            pass

    def A(self, img):
        if self.mode == "blur":
            img = np.pad(img, self.size//2, mode = 'reflect')
            out = convolve2d(img, self.operator, 'valid')
            return out
        
        elif self.mode == "supr":
            img = np.pad(img, self.size//2, mode = 'wrap')
            img = convolve2d(img, self.operator, 'valid')
            img = Downsample(img, self.res)
            out = Upsample(img, self.res)
            return out
        
        elif self.mode == "inpnt":
            return img*self.operator
        
        elif self.mode == "cs-R":
            out = self.operator @ img.flatten()
            return out
        
        elif self.mode == "cs-sf":
            # y = M * fft2(x);    % Normalize by sqrt(N) to ensure orthonormality
            # y = S * F;
            out = self.S * np.fft.fft2(img)
            return out
        
        elif self.mode == "I":
            return img
        
    def At(self, inp):
        if self.mode == "blur":
            inp = np.pad(inp, self.size//2, mode = 'reflect')
            out = convolve2d(inp, np.flipud(np.fliplr(self.operator)), 'valid')
            return out
        
        elif self.mode == "supr":
            out = Upsample_transpose(inp, self.res)
            out = Downsample_transpose(out, self.res)
            out = np.pad(out, self.size//2, mode = 'wrap')
            out = convolve2d(out, np.flipud(np.fliplr(self.operator)), 'valid')
            return out
        
        elif self.mode == "inpnt":
            return inp*self.operator
        
        elif self.mode == "cs-R":
            out = self.operator.T @ inp
            out = out.reshape(self.img_shape, self.img_shape)
            return out
        
        elif self.mode == "cs-sf":
            # y = ifft2(S * x);
            # y = F.T * S.T;
            out = np.fft.ifft2(self.S * inp)
            return np.real(out)
        
        elif self.mode == "I":
            return inp

    def leastsq_A(self, x, gamma=1):
        if self.mode == "cs-R":
            B = self.oper.I_plus_AAt_inv
            y = self.oper.A(x)
            y = (B @ y)
            y = self.oper.At(y)
            out = x - y
            return out
        elif self.mode == "cs-sfd":
            # y = D.*x;
            # y = fft2(y);
            # y = M.*y;
            # y = ifft2(y);
            # y = D.*y;
            # y = y/2;
            # out = x - y;
            y = self.D * x
            y = np.fft.fft2(y)
            y = self.S * y
            y = np.fft.ifft2(y)
            y = (self.D * y) / 2
            out = x - y
            return np.real(out)
        
        elif self.mode == "cs-sf":
            # y = fft2(x);
            # y = M.*y;
            # y = ifft2(y);
            # y = y/2;
            # out = x - y;
            y = np.fft.fft2(x)
            y = self.S * y
            y = np.fft.ifft2(y) / (2 * gamma)
            out = x - y
            return np.real(out)/gamma


class linear:
    def __init__(self, mode, oper, lvl = 0, wave='haar'):
        self.oper = linear_op(mode=mode, oper=oper)

    def A(self, x):
        return self.oper.A(x)

    def At(self, x):
        return self.oper.At(x)
    
    def leastsq_A(self, x, gamma=1):
        return self.oper.leastsq_A(x, gamma)
    
    def function_value(self, x, b):
        return 1/2 * np.linalg.norm(self.A(x) - b)