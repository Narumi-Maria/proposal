+ 打开config文件可以查看参数设置

+ 运行代码时请将数据集放在data文件夹内，并增加一个存放权重的文件夹ckpt，如下所示：

  ckpt

  data

  ​	images

  ​	mask

  ​	test_box.npy

  ​	test_data.csv

  ​	test_depth_feats.npy

  ​	train_box.npy

  ​	train_data.csv

  ​	train_depth_feats.npy

+ 修改了以下部分：

  ​	增加了depth_mask函数用于实现partial depth map

  ​	修改Bottleneck为_Bottleneck， 将geometry feature g<sub>i</sub>的维度从2048降为1024

  ​	增加了PlanA，将PlanB中的权值做了归一化后注释掉，方便以后测试使用

+ 最好结果：

  |                          | loss+1*loss2 | loss1+10*loss2 | loss1+100*loss2 | loss1+1000*loss2 |
  | ------------------------ | ------------ | -------------- | --------------- | ---------------- |
  | global f1                | 0.770        | 0.764          | 0.745           | 0.719            |
  | local f1                 | 0.765        | 0.781          | 0.787           | 0.797            |
  | global balanced accuracy | 0.833        | 0.827          | 0.815           | 0.787            |
  | local balanced accuracy  | 0.828        | 0.839          | 0.843           | 0.845            |
  
  