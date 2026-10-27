# 测试目录

这些测试默认从仓库根目录执行，例如：

```bash
python tests/test_train_args.py
python tests/test_prior_consistency.py
python tests/test_e0e4.py
```

测试文件会自动把仓库根目录加入 Python 搜索路径，不再依赖本机绝对路径。

`test_training_core.py`虽使用合成输入并在CPU上执行，但需要单独准备`pretrained/pvt_v2_b2.pth`以核对真实骨干加载和共享初始化；该权重不随Git发布。其他测试的依赖见各自文件，不把“免GPU”等同于“不需要Python依赖或外部资产”。
