# Twitter-Sentiment-Analysis

**代码已经全部修改完毕，torch版本的可以得到93%左右的acc，tf版本的可以得到95+%左右的acc**

# 文件

- model：存放tf训练出来的model，textcnn.ckpt
- static：存放图片等静态文件
- data.py：生成torch和tf所使用的数据
- model_tf.py：textcnn的tensorflow代码
- model_torch.py：textcnn的pytorch代码
- plot_model.py：绘制模型图的代码
- twitter_training.csv：训练数据
- twitter_validation.csv：验证数据
- utils.py：设置gpu模式代码和pytoch模型所需的one_hot函数

# 依赖

tensorflow 2.10.0

pytorch 1.12.1+cu113

pandas 1.3.0

numpy 1.23.5

# 架构图

![textcnn架构图](.//static/textcnn架构图.png)

# 代码绘制的模型图

![textcnn架构图](.//static/model.png)

# 问题

torch版本的代码是可以运行的，但是还有一点点bug没有解决，就是在构建conv的时候用列表生成式的方式还有点问题没解决，但是程序是可以运行的。（本人已经不想解决了哈哈哈，就直接用最后上传的那种方式玩吧）

# 备注

model文件夹里面只有用tf的save_weights保存的模型，torch训练的模型我没有保存
