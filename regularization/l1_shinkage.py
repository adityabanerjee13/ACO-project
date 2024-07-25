import numpy as np
import pywt


class wavelet_op:
    def __init__(self, lvl = 3, wave='haar'):
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
    
    def wav_l1_proj(self, x):
        count=0
        for cof in x:
            if count==0:
                count=1
                cof = cof / np.maximum(np.abs(cof), 1)
            else:
                for c in cof:
                    c = c / np.maximum(np.abs(c), 1)
        return x

    def wav_l1(self, x, normal = False):
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

    
# FISTA
class prox_FISTA:
    def __init__(self, l1_params, lamda):
        self.lamda = lamda
        self.wave_op = wavelet_op(lvl=l1_params.lvl, wave=l1_params.wave)

    def prox(self, x, t):
        return self.wave_op.shrink(x, t)
    
    def function_value(self, x):
        return self.lamda * self.wave_op.wav_l1(x)