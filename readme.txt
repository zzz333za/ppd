目录结构：
01_training:   模型训练相关
02_inference:  模型推断相关
03_submission: 提交文件相关

使用步骤：将官方数据集放入上级目录data中，手动解压，运行train中expand.py,运行lgb.py。训练部分结束。运行inference中的程序，结束。
依赖环境：windows10 ，python 3.6 ，常规环境sklearn，pandas最新，lightgbm==2.1.2.
expand只把数据及扩展30倍输出，时间由电脑io速度而定。
lgb部分只是用了1/3数据集，比较快速，全程应该两小时以内，不过据说使用linux系统内存占用会很大。
rulelast读取完成后几分钟结束。

