from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import scipy.stats as stats
import math
from scipy.stats import gamma
from LFHSIC.kernels import kernel_midwidth_rbf


class IndpTest(object):
    """
    An abstract class for independence test with Fourier feature. 
    The test requires a paired dataset specified by giving X, Y (torch tensors float64) such that X.shape[0] = Y.shape[0] = n. 
    """
    def __init__(self, X, Y, alpha=0.05):
        """ 
        X: Torch tensor of size n x dx, float64
        Y: Torch tensor of size n x dy, float64
        alpha: significance level of the test
        """
        self.X = X
        self.Y = Y
        self.alpha = alpha

    @abstractmethod
    def perform_test(self):
        """
        Perform the independence test and return values computed in a
        dictionary:
        {
            alpha: 0.05, 
            thresh: 1.0, 
            test_stat: 2.3, 
            h0_rejected: True, 
        }

        All values in the returned dictionary. 
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self):
        """
        Compute the test statistic. 
        Return a scalar value.
        """
        raise NotImplementedError()

class IndpTest_naive_rff(IndpTest):
    """
    Independence test with fixed midwidth parameters.
    Gaussian classes are implemented!
    
    This test runs in O(n D^2 (dx+dy)) time
    n: the sample size
    D: the number of frequency samplings
    dx,dy: the dimension of x,y

    H0: x and y are independence 
    H1: x and y are not independence

    """

    def __init__(self, X, Y, alpha=0.05, n_permutation=100, kernel_type="Gaussian", null_gamma = True):
        """
        alpha: significance level 
        n_permutation: The number of times to simulate from the null distribution
            by permutations. Must be a positive integer. Default: 100. 
        type: "Gaussian"
        null_gamma: if using gamma approximate. Default: True.
        """
        super(IndpTest_naive_rff, self).__init__(X, Y, alpha)
        self.n_permutation = n_permutation
        self.kernel_type = kernel_type
        self.null_gamma = null_gamma
        
        if self.kernel_type not in ["Gaussian", "Laplace"]:
            raise NotImplementedError()
    
    def perform_test(self, rff_num):
        """
        Perform the independence test and return values computed in a dictionary.
        rff_num (D): the number of frequency samplings
        """
        if self.kernel_type == "Gaussian":
            ### generate frequency ###
            dx = self.X.shape[1]
            dy = self.Y.shape[1]
            unit_rff_freqx_fix, unit_rff_freqy_fix = self.freq_gen_gaussian(dx, dy, rff_num = rff_num)
            
            wx, wy = self.midwidth_rbf(self.X, self.Y)
            if self.null_gamma == True:
                fX, fY = self.feat_gen_gaussian(self.X, self.Y, wx, wy)
                rfx, rfy = self.rff_generate(fX, fY, unit_rff_freqx_fix, unit_rff_freqy_fix)
                rfxc = rfx - torch.mean(rfx,0)
                rfyc = rfy - torch.mean(rfy,0)
                testStat = self.compute_stat(rfx, rfy, rfxc, rfyc)
                thresh = self.cal_thresh_gamma(rfx, rfy, rfxc, rfyc)
            else:
                fX, fY = self.feat_gen_gaussian(self.X, self.Y, wx, wy)
                rfx, rfy = self.rff_generate(fX, fY, unit_rff_freqx_fix, unit_rff_freqy_fix)
                rfxc = rfx - torch.mean(rfx,0)
                rfyc = rfy - torch.mean(rfy,0)
                testStat = self.compute_stat(rfx, rfy, rfxc, rfyc)
                thresh = self.cal_thresh_pm(rfx, rfy, rfxc, rfyc)  
        
        h0_rejected = (testStat>thresh)
        
        results_all = {}
        results_all["alpha"] = self.alpha
        results_all["thresh"] = thresh
        results_all["test_stat"] = testStat
        results_all["h0_rejected"] = h0_rejected

        return results_all
    
    def cal_thresh_pm(self, X, Y, wx, wy):
        ind = []
        for _ in range(self.n_permutation):
            p = np.random.permutation(len(X))
            Xp = X[p]
            K, L = self.cal_kernels(Xp, Y, wx, wy)
            Kc = K - torch.mean(K,0)
            Lc = L - torch.mean(L,1)
            s_p = self.compute_stat(K, L, Kc, Lc)
            ind.append(s_p)
        sort_statistic = np.sort(ind)
        ls = len(sort_statistic)
        thresh_p = sort_statistic[int((1-self.alpha)*ls)+1]
        return thresh_p
    
    def cal_thresh_gamma(self, rfx, rfy, rfxc, rfyc):
        """
        Compute the test thresh and parameter (of gamma distribution). 
        """
        n = len(rfx)
        
        vm_xx = rfxc.T @ rfxc
        vm_yy = rfyc.T @ rfyc
        cxx_norm = torch.sum(vm_xx**2) / n / n
        cyy_norm = torch.sum(vm_yy**2) / n / n
        varHSIC = cxx_norm * cyy_norm

        varHSIC = varHSIC * 2 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

        mHSIC = (torch.sum(rfxc**2) * torch.sum(rfyc**2)) / n / (n-1) / (n-1)

        al = (mHSIC**2 / varHSIC).detach().numpy()
        bet = (varHSIC*n / mHSIC).detach().numpy()

        thresh = gamma.ppf(1-0.05, al, scale=bet)

        return thresh
    
    def feat_gen_gaussian(self, X, Y, wx, wy):
        
        fX = X/wx
        fY = Y/wy
        
        return fX, fY
        
    def freq_gen_gaussian(self, dx, dy, rff_num = 500):
        
        unit_rff_freqx = torch.randn(int(rff_num / 2), dx, dtype = torch.float64)
        unit_rff_freqy = torch.randn(int(rff_num / 2), dy, dtype = torch.float64)

        return unit_rff_freqx, unit_rff_freqy
    
    def rff_generate(self, fX, fY, unit_rff_freqx, unit_rff_freqy):
        Dx = len(unit_rff_freqx)*2
        Dy = len(unit_rff_freqy)*2

        rff_freqx = unit_rff_freqx
        rff_freqy = unit_rff_freqy

        xdotw = fX@rff_freqx.T
        ydotw = fY@rff_freqy.T

        rfx = math.sqrt(2./Dx)*torch.cat((torch.cos(xdotw),torch.sin(xdotw)), 1)
        rfy = math.sqrt(2./Dy)*torch.cat((torch.cos(ydotw),torch.sin(ydotw)), 1)

        return rfx, rfy
    
    def compute_stat(self, rfx, rfy, rfxc, rfyc):
        """
        Compute the test statistic. 
        """
        n = len(rfx)

        testStat = torch.sum((rfyc.T @ rfxc)**2) / n

        return testStat
    
    def cal_thresh_pm(self, rfx, rfy, rfxc, rfyc):
        ind = []
        for _ in range(self.n_permutation):
            p = np.random.permutation(len(rfx))
            rfxp = rfx[p]
            rfxcp = rfxc[p]
            s_p = self.compute_stat(rfxp, rfy, rfxcp, rfyc)
            ind.append(s_p)
        sort_statistic = np.sort(ind)
        ls = len(sort_statistic)
        thresh_p = sort_statistic[int((1-self.alpha)*ls)+1]
        return thresh_p
        
    def midwidth_rbf(self, X, Y, max_num = 1000):
        """
        Calculate midwidth of Gaussian kernels 
        (also return maxwidth that can be used to limit the range in learning kernels)
        
        Return 
        wx_mid, wy_mid, wx_max, wy_max
        """
        wx_mid, wy_mid, _, _ = kernel_midwidth_rbf(X[:max_num], Y[:max_num])
        
        return wx_mid, wy_mid