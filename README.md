# 项目架构
1. models：存储各种自定义块、自定义网络、以及结合这些网络的主体网络 以及自定义的数据集读取
2. models\dataset.py：存储各种自定义的dataset（继承自Dataset）用于读取训练数据和预处理
3. utils.py：存储一些有用的函数，例如动图制作，中间处理等
4. loss.py：存储一些自定义的损失函数，作为单独的层来使用 被放在model中
5. train.py 训练模型
6. pretrained_model：预训练好的模型
7. checkpoint：保存的模型checkpoint
8. test 测试模型 例如放入自己的数据集 让预训练好的模型来工作
9. docs 存储项目文档文件 和docs/images 存储图片 用于放入解释
10. configs 存储超参数 含yaml文件
11. images: 一些模型的输出结果