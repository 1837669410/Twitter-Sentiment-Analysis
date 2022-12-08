# Twitter-Sentiment-Analysis

**代码还没有修改完毕，大致结果能实现了95%左右的val_acc，但是datalaod代码比较混乱，有时间会修改后在更新**

# 依赖

tensorflow 2.10.0

pytorch 1.12.1+cu113

pandas 1.3.0

numpy 1.23.5

# 架构图

![textcnn架构图](.//static/textcnn架构图.png)

# 问题

torch版本的代码是可以运行的，但是还有一点点bug没有解决，就是在构建conv的时候用列表生成式的方式还有点问题没解决，但是程序是可以运行的。（本人已经不想解决了哈哈哈，就直接用最后上传的那种方式玩吧）

# 备注

model文件夹里面只有用tf的save_weights保存的模型，torch训练的模型我没有保存
