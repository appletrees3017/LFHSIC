import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import scipy.stats as stats
from scipy.stats import gamma
from LFHSIC.fhsic_naive import IndpTest
import math

import cupy as cp   #针对GPU的numpy
import cupyx.scipy 
from LFHSIC.kernels import kernel_midwidth_rbf

class IndpTest_LFGaussian(IndpTest):

    """
    Independence test with learnable Fourier feature (Gaussian: global scale).
    This test runs in O(T*n*D^2*(dx+dy)) time.
    T: the number to perform gradient descent
    D: the number of frequency samplings
    n: the sample size
    dx,dy: the dimension of x,y
    
    H0: x and y are independence 
    H1: x and y are not independence

    """

    #初始化#
    def __init__(self, X, Y, device, alpha=0.05, n_permutation=100, null_gamma = True, split_ratio = 0.5):
        """
        alpha: significance level 
        n_permutation: The number of times to simulate from the null distribution
            by permutations. Must be a positive integer.
        null_gamma: if null_gamma == 'False', use the permutation method for the test step.    
        split_ratio: split ratio of samples (Train/all)
        """
        super(IndpTest_LFGaussian, self).__init__(X, Y, alpha)
        self.n_permutation = n_permutation
        self.null_gamma = null_gamma
        self.split_ratio = split_ratio
        self.device = device
    #数据分割#
    def split_samples(self):
        """
        split datasets into train/test datasets
        """
        n = len(self.X)
        
        #索引重排生成
        p = np.random.permutation(n)  # 生成随机排列的索引permutation
        tr_size = int(n*self.split_ratio)  #训练集大小#
        ind_train = p[:tr_size] #训练集索引--行的索引#
        ind_test = p[tr_size:]  #测试集索引--行的索引#
        
        Xtr = self.X[ind_train,:] #ind_train提取对应行的数据
        Ytr = self.Y[ind_train,:]
        Xte = self.X[ind_test,:]  #ind_test提取对应行的数据
        Yte = self.Y[ind_test,:]
        
        if len(Xtr.size())==1:
            Xtr = Xtr.reshape(-1,1)  #处理1维数据--转换成​列向量​
        if len(Ytr.size())==1:
            Ytr = Ytr.reshape(-1,1)
        if len(Xte.size())==1:
            Xte = Xte.reshape(-1,1)
        if len(Yte.size())==1:
            Yte = Yte.reshape(-1,1)
        
        return Xtr, Ytr, Xte, Yte

###算法主流程↓###
    def perform_test(self, rff_num, lr = 0.05, iter_steps=100, if_grid_search = False, debug = -1):
        """
        Perform the independence test and return values computed in a dictionary.
        if_grid_search: if use grid_search for widths to init before perform optimize.
        debug: if >0: then print details of the optimization trace. 
        """
        
        """ split the datasets """
        Xtr, Ytr, Xte, Yte = self.split_samples()
        
        """ generate frequency """
        dx = self.X.shape[1]
        dy = self.Y.shape[1]
        unit_rff_freqx_fix, unit_rff_freqy_fix = self.freq_gen(dx, dy, rff_num = rff_num)
        
        #选择配置初始化核宽#
        if if_grid_search:
            wx_mid, wy_mid, wx_max, wy_max = self.midwidth_rbf(Xtr, Ytr)
            wx_init, wy_init = self.grid_search_init(Xtr, Ytr, wx_mid, wy_mid, unit_rff_freqx_fix, unit_rff_freqy_fix)
        else:
            wx_init, wy_init, wx_max, wy_max = self.midwidth_rbf(Xtr, Ytr)
        
        #将模型或数据迁移到GPU中进行加加速计算#
        Xtr = Xtr.to(self.device)    
        Ytr = Ytr.to(self.device)
        unit_rff_freqx_fix = unit_rff_freqx_fix.to(self.device)
        unit_rff_freqy_fix = unit_rff_freqy_fix.to(self.device)    
        #####################################
        
        #核宽优化#
        wx, wy, path = self.search_width_fix(Xtr, Ytr, wx_init, wy_init, wx_max, wy_max, unit_rff_freqx_fix, unit_rff_freqy_fix, \
                                             lr = lr, iter_steps = iter_steps, debug = debug, limit_max = False) 
        
        #选择配置计算检验统计量核阈值#
        if self.null_gamma == True:
            if self.device.type == "cuda":  #如果设备是CUDA（GPU），将变量unit_rff_freqfix从GPU显存移到CPU内存#
                unit_rff_freqx_fix = unit_rff_freqx_fix.cpu() #设置：后续用CPU计算#
                unit_rff_freqy_fix = unit_rff_freqy_fix.cpu()
            fX, fY = self.feat_gen(Xte, Yte, wx, wy)
            rfx, rfy = self.rff_generate(fX, fY, unit_rff_freqx_fix, unit_rff_freqy_fix)
            rfxc = rfx - torch.mean(rfx,0)
            rfyc = rfy - torch.mean(rfy,0)
            testStat, _ = self.J_maxpower_term(rfx, rfy, rfxc, rfyc)
            thresh, _, _ = self.cal_thresh(rfx, rfy, rfxc, rfyc)
        else:
            if self.device.type == "cuda":
                unit_rff_freqx_fix = unit_rff_freqx_fix.cpu()
                unit_rff_freqy_fix = unit_rff_freqy_fix.cpu()
            fX, fY = self.feat_gen(Xte, Yte, wx, wy)
            rfx, rfy = self.rff_generate(fX, fY, unit_rff_freqx_fix, unit_rff_freqy_fix)
            rfxc = rfx - torch.mean(rfx,0)
            rfyc = rfy - torch.mean(rfy,0)
            testStat, _ = self.J_maxpower_term(rfx, rfy, rfxc, rfyc)
            thresh, _, _ = self.cal_thresh_pm(rfx, rfy, rfxc, rfyc)  
        
        h0_rejected = (testStat>thresh)

        #输出结果result_all的组成
        results_all = {}
        results_all["alpha"] = self.alpha
        results_all["thresh"] = thresh
        results_all["test_stat"] = testStat
        results_all["h0_rejected"] = h0_rejected
        
        return results_all
###算法主流程↑###
    
    
    #随机频率矩阵生成#    
    def freq_gen(self, dx, dy, rff_num = 500):
        
        unit_rff_freqx = torch.randn(int(rff_num / 2), dx, dtype = torch.float64)
        unit_rff_freqy = torch.randn(int(rff_num / 2), dy, dtype = torch.float64)

        return unit_rff_freqx, unit_rff_freqy
    
    
    #核宽优化####################################
    def search_width_fix(self, X, Y, wx_init, wy_init, wx_max, wy_max, unit_rff_freqx_fix, unit_rff_freqy_fix, lr = 0.05, delta_estimate_grad=1e-6, \
                     iter_steps = 100, limit_max = True, debug = -1):
        
        n = len(X)
        
        wx_log_init = torch.log(wx_init)
        wy_log_init = torch.log(wy_init)

        use_gpu = False
        if X.device.type == "cuda":
            use_gpu = True
            device = X.device

        if use_gpu:
            wx_log = torch.tensor([wx_log_init],requires_grad=True, device = device)
            wy_log = torch.tensor([wy_log_init],requires_grad=True, device = device)
        else:
            wx_log = torch.tensor([wx_log_init],requires_grad=True)
            wy_log = torch.tensor([wy_log_init],requires_grad=True)

        optimizer = optim.Adam([wx_log,wy_log],lr=lr)  #优化器--Adam优化器
        delta = delta_estimate_grad   #δ--梯度估计的扰动
        path = np.zeros((iter_steps,3))  #iter_steps*3矩阵 记录每步的J wx wy 

        for st in range(iter_steps):
            
            optimizer.zero_grad()  # 清除上一轮的梯度，避免累积

            wx = torch.exp(wx_log) # 转换为实值  指数正值
            wy = torch.exp(wy_log)

            fX, fY = self.feat_gen(X, Y, wx, wy)
            rfx, rfy = self.rff_generate(fX, fY, unit_rff_freqx_fix, unit_rff_freqy_fix)
            
            rfxc = rfx - torch.mean(rfx,0)
            rfyc = rfy - torch.mean(rfy,0)

            testStat, sigma_estimate_reg = self.J_maxpower_term(rfx, rfy, rfxc, rfyc)  #生成检验统计量

            """--------------calculate thresh 选择配置--------------"""
            al, bet = self.cal_thresh_param(rfx, rfy, rfxc, rfyc)  #提取gamma分布的参数
            if use_gpu:
                r0, al_detach, bet_detach, grad_al, grad_bet = self.cal_thresh_gamma_gpu(al, bet, if_grad = True)
            else:
                r0, al_detach, bet_detach, grad_al, grad_bet = self.cal_thresh_gamma(al, bet, if_grad = True)
            
            """--------------calculate power criterion ----------------"""
            J0 = (testStat-r0)/sigma_estimate_reg
            J0_value = -J0.detach()
            sigma_estimate_reg_detach = sigma_estimate_reg.detach()
            
            """--------------calculate the grad of power criterion ----------------"""
            J = J0 - (grad_al*al+ grad_bet*bet)/sigma_estimate_reg_detach
            
            """----add """
            J = J/math.sqrt(n)

            (-J).backward()

            if debug > 0:
                if st%debug==0:
                    print(J0_value.item(), testStat.item(),r0.item(),wx.item(),wy.item())
            
            #记录每步的参数#
            path[st,0] = (J0_value).item()
            path[st,1] = wx.item()
            path[st,2] = wy.item()

            if limit_max == True:
                if wx.item() > 2*wx_max or wy.item() > 2*wy_max:
                    return wx_init, wy_init, path

            optimizer.step()

        return wx.item(), wy.item(), path
    #核宽初始化#
    #grid网络搜索#
    def grid_search_init(self, X, Y, wx_mid, wy_mid, unit_rff_freqx_fix, unit_rff_freqy_fix):
        """
        Using grid_search (log_scale) to init the widths (just the same as nfsic)
        使用grid_search(对数尺度--在对数空间搜索--对数指数互逆操作)初始化宽度(与nfsic相同)
        """
        n_gwidth_cand = 5
        gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand) 

        width_pair = []
        J_pair = []
        
        for facx in gwidth_factors:
            for facy in gwidth_factors:
                wx = facx*wx_mid
                wy = facy*wy_mid

                fX, fY = self.feat_gen(X, Y, wx, wy)
                rfx, rfy = self.rff_generate(fX, fY, unit_rff_freqx_fix, unit_rff_freqy_fix)
                rfxc = rfx - torch.mean(rfx,0)
                rfyc = rfy - torch.mean(rfy,0)

                thresh, al, bet = self.cal_thresh(rfx, rfy, rfxc, rfyc)
                testStat, sigma_estimate_reg = self.J_maxpower_term(rfx, rfy, rfxc, rfyc)

                width_pair.append((wx,wy)) #使用列表保存元组
                J_pair.append((testStat - thresh)/sigma_estimate_reg)

        J_array = np.array(J_pair)  #将 J_pair 从Python列表转为NumPy数组，便于高效操作
        indm = np.argmax(J_array)   #最大值点

        return width_pair[indm]
    #固定核宽--基于欧氏平方距离的中值估计核宽#
    def midwidth_rbf(self, X, Y, max_num = 1000):
        """
        Calculate midwidth of Gaussian kernels 
        (also return maxwidth that can be used to limit the range in learning kernels)
        
        Return 
        wx_mid, wy_mid, wx_max, wy_max
        """
        wx_mid, wy_mid, wx_max, wy_max = kernel_midwidth_rbf(X[:max_num], Y[:max_num])
        
        return wx_mid, wy_mid, wx_max, wy_max
    
    ############################################
    
    
    
    
    #计算检验统计量
    
    #gamma近似 计算检验统计量#####
    
    def J_maxpower_term(self, rfx, rfy, rfxc, rfyc):
        """
        Compute the terms for power criterion. 
        """
        n = len(rfx)

        testStat = torch.sum((rfyc.T @ rfxc)**2) / n
        
        Dxy = rfx.T@rfy
        Dxyy = Dxy@rfy.T
        A = torch.sum(rfx*(Dxyy.T),1).reshape(-1,1)
        fxs = torch.sum(rfx,0).reshape(-1,1)
        fys = torch.sum(rfy,0).reshape(-1,1)
        B = rfx@fxs
        C = rfy@fys
        D = B*C
        
        h_i = 1/2*((n**2)*A+n*torch.sum(A)+torch.sum(C)*B+torch.sum(B)*C-n*(D+rfx@(rfx.T@C)+rfy@(rfy.T@B))-torch.sum(D))/(n**3)
        #tourch.sum逐元素求和#
        var_estimate = 16*(torch.sum((h_i)**2)/n-((testStat)/n)**2)
        sigma_estimate_reg = torch.sqrt(var_estimate)
        
        return testStat, sigma_estimate_reg
    #置换实验计算检验统计量#
    
    def compute_stat(self, rfx, rfy, rfxc, rfyc):
        """
        Compute the test statistic. 
        """
        n = len(rfx)

        testStat = torch.sum((rfyc.T @ rfxc)**2) / n

        return testStat
    
    ###################
    
    
    
    #计算阈值######
    #gamma分布计算阈值---gamma分位数#
    def cal_thresh(self, rfx, rfy, rfxc, rfyc):
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

        return thresh, al, bet
    #置换检验计算gamma分布#
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
    
    ###############
    
    
    #标准化数据--高斯特征核#
    def feat_gen(self, X, Y, wx, wy):
        
        fX = X/wx
        fY = Y/wy
        
        return fX, fY
    #RFF傅里叶随机特征生成#
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
    
    
    #gamma分布参数提取  在梯度下降优化过程中，实时计算当前核宽度对应的Gamma分布参数#
    def cal_thresh_param(self, rfx, rfy, rfxc, rfyc):
        """
        Compute the parameter (of gamma distribution). 
        """
        n = len(rfx)
        
        vm_xx = rfxc.T @ rfxc
        vm_yy = rfyc.T @ rfyc
        cxx_norm = torch.sum(vm_xx**2) / n / n
        cyy_norm = torch.sum(vm_yy**2) / n / n
        varHSIC = cxx_norm * cyy_norm

        varHSIC = varHSIC * 2 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

        mHSIC = (torch.sum(rfxc**2) * torch.sum(rfyc**2)) / n / (n-1) / (n-1)

        al = (mHSIC**2 / varHSIC)
        bet = (varHSIC*n / mHSIC)

        return al, bet
    #用于计算阈值对核宽度参数 (wx, wy) 的梯度，驱动优化过程  梯度优化搜索中运用#
    def cal_thresh_gamma(self, al, bet, if_grad = False, delta = 1e-6):
        """
        Compute the thresh and parameter (of gamma distribution). 
        if_grad: if need to obtain the gradient of thresh.
        """
        al = al.detach().numpy()
        bet = bet.detach().numpy()

        thresh = gamma.ppf(1-self.alpha, al, scale=bet)

        if if_grad == True:
            thresh_al = gamma.ppf(1-self.alpha, al+delta, scale=bet) 
            thresh_bet = gamma.ppf(1-self.alpha, al, scale=bet+delta) 
            grad_al = (thresh_al - thresh)/delta
            grad_bet = (thresh_bet - thresh)/delta

            return thresh, al, bet, grad_al, grad_bet

        return thresh, al, bet
    #使用GPU加速计算---急急国王#
    def cal_thresh_gamma_gpu(self, al, bet, if_grad = False, delta = 1e-2):
        """
        For GPU (cupyx is needed)
        Compute the thresh and parameter (of gamma distribution). 
        if_grad: if need to obtain the gradient of thresh.
        """
        al = al.detach()
        bet = bet.detach()

        thresh = torch.tensor(cupyx.scipy.special.gammaincinv(al,1-self.alpha))*bet

        if if_grad == True:
            thresh_al = torch.tensor(cupyx.scipy.special.gammaincinv(al+delta,1-self.alpha))*bet
            # thresh_bet = torch.tensor(cupyx.scipy.special.gammaincinv(al,1-self.alpha))*(bet+delta)
            grad_al = (thresh_al - thresh)/delta
            # grad_bet = (thresh_bet - thresh)/delta
            grad_bet = thresh/bet

            return thresh, al, bet, grad_al, grad_bet

        return thresh, al, bet
    
    
    