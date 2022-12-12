# AntCritic
## 简介
AntCritic 是一个应用于论点论据挖掘的融合超文本标记信息的多任务统一框架，它有以下特点：
 
* 同时建模元素分类和元素关系；
* **融合超文本标记信息**；

### 效果示例：

<p align="center"><img src="https://user-images.githubusercontent.com/113573331/206978762-8d77c5df-4b33-460a-8a96-af016e6d3eba.png"></img>
</p>
<p align="center">
<img src = "https://user-images.githubusercontent.com/113573331/206979157-f0506103-6d19-45eb-a573-24415f7ac6ad.png"></img>
</p>

## 快速上手
### 环境配置

安装依赖包: `pip install -r requirements.txt` 

### Checkpoints
使用`git lfs clone https://github.com/ContentTech/AntCritic.git` 克隆仓库，模型在下面对应的目录下：

预训练模型：
* 词级别: pretrained_model/paraphrase-xlm-r-multilingual-v1 
* 字级别: pretrained_model/FinBERT_L-12_H-768_A-12_pytorch

使用antcritic数据集finetune的预训练模型:
* 词级别: checkpoints/char/models-9.pt
* 字级别: checkpoints/word/models-12.pt

使用antcritic数据集训练的论文中Figure4所示模型:
* checkpoints/GRU/models-7.pt


### 数据下载

* 训练集: [antcritic/train.1.csv](https://tianchi.aliyun.com/dataset/142920)
* 测试集: [antcritic/test.1.csv](https://tianchi.aliyun.com/dataset/142920)
* 验证集: [antcritic/dev.1.csv](https://tianchi.aliyun.com/dataset/142920)
* 详情:


  | Dataset                 | Domain    | Unit | Relation?| Modal | Lang|
 |---------------------------|-----------|---------|---------|---------|---------|
  | antcritic                | Financial Comments | Segment    | Yes | Text&HTML| Chinese|


  | Dataset                | #Docs       | #Sents     | #Claims     | #Tokens    | 
  |------------------------|------------|-------------|------------|------------|
  | train                | 9994 | 214585     | 88311 | 11436977    |
  | dev                 | 9994 | 214585     | 88311 | 11436977    |
  | test                 | 9994 | 214585     | 88311 | 11436977    |

 

  <details>
  <summary><b>例子</b></summary>
  #### Example 1
  
  | 数据集字段 | 字段描述                                                                                                                                                                                                                                                                                                                                                                                                        | 字段类型                | 例子                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  |-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | srcs  | 原始文章                                                                                                                                                                                                                                                                                                                                                                                                        | string              | （1）一波行情，往大了说，都是时代的礼物，比如12年重工业见顶后，内需服务消费，代表品种消费股投资，我常说4点，社交属性、成瘾性（复购率）、产品生命力、金融属性；综合而言，为何酒是最好的品种，没有之一呢，比如社交属性，举个小例子，酒庄上最容易明白的就是关系，你说客户让我来参加一个酒局，你来不来？你肯定得来啊，这是给客户来表诚意来了，我来了，又喝多了，出丑了。我把我的面子全给放下了，这某种情况来说，就是给客户的诚意的抵押物呀。比如从低度酒到高度酒啊，可能是一个我们从熟人社会向生人社会变迁的这么一个过程，要迅速的找到这个四十度、五十度甚至六十度的东西呢，把我们的情绪顶起来。\n往后，19年缩表减税搞好资本市场，就类似美股1980，结束了漂亮50，为啥说是19年而不是20年或21年呢，觉得是疫情再放水，导致了这个周期被延后，严格点说消费周期是19年结束了。往后就制造业起来，光伏、锂电为代表的能源革命，既然也是时代礼物，那么通常三波走势，第一波先来个2-3倍，比如价格从10块干到40，然后回撤50%，然后再来一波3倍以上行情，从20干到70，然后再回落个下，然后上涨到80-100，这么完整一轮行情就结束了，龙头品种10倍。现在锂电处于第二波主升浪的末端，很多标的完成了50%回撤后的3倍以上行情了，所以我给的建议是有格局的选手，认为基本面不断刷新大家认知的，可以坚定持股，哪怕是顶部也是阶段性的或是走势复合型的，不用在意一时波动，喜欢拥抱波动的觉得不妨可以减仓。\n赣锋锂业上修业绩预告，预计上半年净利润13亿元-16亿元，同比增长730.75%-922.46%，此前预计盈利8亿元-12亿元。\n这波跟容百一起，也算给大家账户增值助力不少。\n（2）说说车载摄像头光学设备，负面的觉得摄像头这东西一直觉得没啥利润，也没啥技术含量，你说占了个认证优势吧，一般车企都要认证几家的摄像头，也不是就它一家，再说了，你汽车摄像头再多也比不上手机吧。乐观的觉得，一个汽车摄像头相当于3个手机摄像头，相关企业给自己带来的增量是明显的，另外摄像头不至于新能源车，2500w台车，每车10个，得有2.5个摄像头，跟手机也差不多了。不过甭管乐观还是负面，一个道理总是错不了的，智能万联时代，信息汲取，他需要一个入口，视觉，靠摄像头光学是最重要的来源。长期拥抱光学资产，从上游的芯片到下游的模组企业。\n（3）风电这个风电，能源行业就看运营商对折旧的容忍度，要让运营商相信可以把成本降下来，快速实现平价，但风电成本曲线与光伏不同，与产品规模效应和大型化相关，我们之前缺乏大型海工平台，然后陆地跟近海优势风力资源可开发资源不多。\n无非就是成本上，就是随着超大风机12MW以上以及漂浮式技术的出现，海上风电度电成本快速下降。\n暂时列入观察窗口！看装机能否上去，成本能否下来！\n                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  | sents | 分隔句子(依据标点+html标签), 从0开始标号；                                                                                                                                                                                                                                                                                                                                                                                  | json string         | "{""0"": ""（1）"", ""1"": ""一波行情，往大了说，都是时代的礼物，比如12年重工业见顶后，内需服务消费，代表品种消费股投资，我常说4点，社交属性、成瘾性（复购率）、产品生命力、金融属性；"", ""2"": ""综合而言，为何酒是最好的品种，没有之一呢，比如社交属性，举个小例子，酒庄上最容易明白的就是关系，你说客户让我来参加一个酒局，你来不来？"", ""3"": ""你肯定得来啊，这是给客户来表诚意来了，我来了，又喝多了，出丑了。"", ""4"": ""我把我的面子全给放下了，这某种情况来说，就是给客户的诚意的抵押物呀。"", ""5"": ""比如从低度酒到高度酒啊，可能是一个我们从熟人社会向生人社会变迁的这么一个过程，要迅速的找到这个四十度、五十度甚至六十度的东西呢，把我们的情绪顶起来。"", ""6"": ""往后，19年缩表减税搞好资本市场，就类似美股1980，结束了漂亮50，为啥说是19年而不是20年或21年呢，觉得是疫情再放水，导致了这个周期被延后，严格点说消费周期是19年结束了。"", ""7"": ""往后就制造业起来，光伏、锂电为代表的能源革命，既然也是时代礼物，那么通常三波走势，第一波先来个2-3倍，比如价格从10块干到40，然后回撤50%，然后再来一波3倍以上行情，从20干到70，然后再回落个下，然后上涨到80-100，这么完整一轮行情就结束了，龙头品种10倍。"", ""8"": ""现在锂电处于第二波主升浪的末端，很多标的完成了50%回撤后的3倍以上行情了，所以我给的建议是有格局的选手，认为基本面不断刷新大家认知的，可以坚定持股，哪怕是顶部也是阶段性的或是走势复合型的，不用在意一时波动，喜欢拥抱波动的觉得不妨可以减仓。"", ""9"": ""赣锋锂业上修业绩预告，预计上半年净利润13亿元-16亿元，同比增长730.75%-922.46%，此前预计盈利8亿元-12亿元。"", ""10"": ""这波跟容百一起，也算给大家账户增值助力不少。"", ""11"": ""（2）"", ""12"": ""说说车载摄像头光学设备，负面的觉得摄像头这东西一直觉得没啥利润，也没啥技术含量，你说占了个认证优势吧，一般车企都要认证几家的摄像头，也不是就它一家，再说了，你汽车摄像头再多也比不上手机吧。"", ""13"": ""乐观的觉得，一个汽车摄像头相当于3个手机摄像头，相关企业给自己带来的增量是明显的，另外摄像头不至于新能源车，2500w台车，每车10个，得有2.5个摄像头，跟手机也差不多了。"", ""14"": ""不过甭管乐观还是负面，一个道理总是错不了的，智能万联时代，信息汲取，他需要一个入口，视觉，靠摄像头光学是最重要的来源。"", ""15"": ""长期拥抱光学资产，从上游的芯片到下游的模组企业。"", ""16"": ""（3）风电"", ""17"": ""这个风电，能源行业就看运营商对折旧的容忍度，要让运营商相信可以把成本降下来，快速实现平价，但风电成本曲线与光伏不同，与产品规模效应和大型化相关，我们之前缺乏大型海工平台，然后陆地跟近海优势风力资源可开发资源不多。"", ""18"": ""无非就是成本上，就是随着超大风机12MW以上以及漂浮式技术的出现，海上风电度电成本快速下降。"", ""19"": ""暂时列入观察窗口！"", ""20"": ""看装机能否上去，成本能否下来！""}"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
  | tags  | a. "font-size"：字体大小，分为三类：本文最常出现的大小（0），比常见大小更大（1）， 更小（2）; <br> b. "color"：前景颜色，只要有前景颜色即为1，否则为0； <br> c. "background-color"：背景颜色，只要有背景颜色，即为1，否则为0；<br> d. "sns-small-title"：是否是小标题；<br> e. "sns-blob-tl"：是否是副标题；<br> f. "strong"：是否加粗；<br> g. "supertalk"：是否是话题标识符（#），是即为1，否则为0; <br> h. "blockquote"：是否是引用语；<br>  i. "po"：段落序号； <br> j. "pi"：段落内编号；<br> k. "h4"：是否是四级标题； | list of json string | "[{""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 1, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 2, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 2, ""pi"": 2}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 2, ""pi"": 3}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 2, ""pi"": 4}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 2, ""pi"": 5}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 3, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 3, ""pi"": 2}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 3, ""pi"": 3}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 4, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 5, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 6, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 7, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 7, ""pi"": 2}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 7, ""pi"": 3}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 7, ""pi"": 4}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 8, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 9, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 10, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 11, ""pi"": 1}, {""font-size"": -1, ""color"": -1, ""background-color"": -1, ""strong"": 0, ""sns-small-title"": 0, ""sns-blob-tl"": 0, ""supertalk"": 0, ""blockquote"": 0, ""h4"": 0, ""po"": 11, ""pi"": 2}]" |
  | trgs  | 标注结果, results字段： <br> a. MajorClaim - 主论点; <br> b. Claim_{i} - 第i个子论点; <br> c.Premise_\<i>_\<j> - 第i个子论点的第j个子论据, 0<=i<=8, 0<=j<=4; 一个主论点，最多8个子论点，每个子论点最多4个论据；<br> d. relations字段: <br> &ensp;&ensp; 子论点和主论点的关系： 默认值是-1，表示不存在该关系，1是支持，0是反驳，2是有关，3是无关；<br>&ensp;&ensp; 子论点和论据的关系：默认值是-1，表示不存在该关系，1是支持，0是反驳；                                                                                                   | json string         | "{""results"": {""MajorClaim"": [7], ""Claim_1"": [2], ""Claim_2"": [8], ""Claim_3"": [12, 15], ""Claim_4"": [16, 19, 20], ""Claim_5"": [], ""Claim_6"": [], ""Claim_7"": [], ""Claim_8"": [], ""Premise_1_1"": [3, 4], ""Premise_1_2"": [5], ""Premise_1_3"": [], ""Premise_1_4"": [], ""Premise_2_1"": [6], ""Premise_2_2"": [], ""Premise_2_3"": [9, 10], ""Premise_2_4"": [], ""Premise_3_1"": [], ""Premise_3_2"": [13, 14], ""Premise_3_3"": [], ""Premise_3_4"": [], ""Premise_4_1"": [17], ""Premise_4_2"": [18], ""Premise_4_3"": [], ""Premise_4_4"": [], ""Premise_5_1"": [], ""Premise_5_2"": [], ""Premise_5_3"": [], ""Premise_5_4"": [], ""Premise_6_1"": [], ""Premise_6_2"": [], ""Premise_6_3"": [], ""Premise_6_4"": [], ""Premise_7_1"": [], ""Premise_7_2"": [], ""Premise_7_3"": [], ""Premise_7_4"": [], ""Premise_8_1"": [], ""Premise_8_2"": [], ""Premise_8_3"": [], ""Premise_8_4"": []}, ""relations"": {""Claim_1"": 3, ""Claim_2"": 1, ""Claim_3"": 0, ""Claim_4"": 1, ""Claim_5"": -1, ""Claim_6"": -1, ""Claim_7"": -1, ""Claim_8"": -1, ""Premise_1_1"": 1, ""Premise_1_2"": 1, ""Premise_1_3"": -1, ""Premise_1_4"": -1, ""Premise_2_1"": 1, ""Premise_2_2"": 1, ""Premise_2_3"": 1, ""Premise_2_4"": -1, ""Premise_3_1"": -1, ""Premise_3_2"": 1, ""Premise_3_3"": -1, ""Premise_3_4"": -1, ""Premise_4_1"": 1, ""Premise_4_2"": 1, ""Premise_4_3"": -1, ""Premise_4_4"": -1, ""Premise_5_1"": -1, ""Premise_5_2"": -1, ""Premise_5_3"": -1, ""Premise_5_4"": -1, ""Premise_6_1"": -1, ""Premise_6_2"": -1, ""Premise_6_3"": -1, ""Premise_6_4"": -1, ""Premise_7_1"": -1, ""Premise_7_2"": -1, ""Premise_7_3"": -1, ""Premise_7_4"": -1, ""Premise_8_1"": -1, ""Premise_8_2"": -1, ""Premise_8_3"": -1, ""Premise_8_4"": -1}, ""url"": ""https://alphaq.alipay.com/index_manage.htm#/mark?taskId=545757&subTaskId=10148412928&isPreview=1&tntInstId=caeaf603""}"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |


### 模型架构
<img src="https://user-images.githubusercontent.com/113573331/206981754-ada095b9-2f4f-4834-9735-7a2acf0c37a7.png">


### 处理数据和训练模型

````
# 预处理数据得到模型第一阶段可读取的格式：*_1.hdf5
python preprocess/new_preprocess.py 

# 模型训练及预测: 修改config/config.py文件中的 saved_path 和 _c.model.name为'FirstStageModel'
CUDA_VISIBLE_DEVICES=0 python first_main.py --config use_word 

# 预处理数据得到模型第二阶段可读取的格式: *_2.hdf5
# 要是不想重新训练char_model和word_model，可以使用训练好的：checkpoints/char/models-9.pt & checkpoints/word/models-12.pt
python preprocess/second_stage.py 

# 模型训练及预测: 修改config/config.py文件中的 saved_path 和 _c.model.name为'SecondStageModel'
CUDA_VISIBLE_DEVICES=0 python second_main.py --config use_gru
````

## Citation

If you find our work useful, please consider citing:

```
@article{Zhao2022AntCriticAM,
  title={AntCritic: Argument Mining for Free-Form and Visually-Rich Financial Comments},
  author={Yang Zhao and Wenqiang Xu and Xuan Lin and Jingjing Huo and Hong Chen and Zhou Zhao},
  journal={ArXiv},
  year={2022},
  volume={abs/2208.09612}
}
```

