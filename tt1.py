# 示例运行代码
#报错from LFHSIC.fhsic_naive import IndpTest_naive_rff
# 改为相对导入（需确保文件在正确位置）
from .fhsic_naive import IndpTest_naive_rff  # 注意开头的点
from .lfhsic_g import IndpTest_LFGaussian

# 准备数据
#X = torch.randn(1000, 5)  # 1000个5维样本
#Y = torch.randn(1000, 3)  # 1000个3维样本

# 基础RFF测试
#test_naive = IndpTest_naive_rff(X, Y)
#result = test_naive.perform_test(rff_num=100)

# 可学习版本测试(Gaussian)
#test_lf = IndpTest_LFGaussian(X, Y, device='cpu')
#result = test_lf.perform_test(rff_num=100, lr=0.05, iter_steps=50)