## Introduction
This folder is for cross-modal. 

## Papers
### [MDL-CW: A Multimodal Deep Learning Framework With Cross Weights][1]
1. CVPR2016, **multi-modal** cross weights
2. 主要的想法与ICLR2017那篇很类似，但不同的是学习了**交叉权重**；交叉权重从某种程度上也可以看作是一个**公共的映射空间**；具体任务也是做不同的**retrieval, ranking**任务。
3. 这篇先不要看。

### [Ambient Sound Provides Supervision for Visual Learning][2]
1. ECCV2016, no citatioin yet.

### [Visually Indicated Sounds][3]
1. CVPR2016, 4 cited
2. Datasets for experiments
3. 本工作与上面一个工作，可以看作是比较新的图像-音频方面的工作。

### [Dense net][4]
1. 与residual net的一个较大不同是，不同层的**特征融合并非相加**，而是**直接concatenate起来**
2. 如果将输入和每一层的**feature map**看作是**state**，那么传统的网络在做两件事：不断的**改变state**；将**需要保存的信息向前传递**。
3. 老板吐血推荐的工作，但我仍然觉得跟cross-modal没啥关系，而且不同模态网络间特征学习的融合并没有合适的intuition。

### [DeMIAN: Deep Modality Invariant Adversarial Network][5]
1. 最基本的intuition是说：对于两个模态的数据，**同时收集数据与标签比较困难**。所以可行的做法是，收集**一个模态的带有标签的数据**，然后收集**多模态的数据**。所以可以首先在具有**丰富标签的数据上学习分类器**，然后**用在另外一个目标源上**。
2. 基本流程：
	- 学习模态间的**共享特征**。
	- 在**带有标签**的某模态数据的**共享特征上训练分类器**。
	- 上述分类器可以直接使用在**另一模态数据的共享特征上**。
3. Modality-Invariant特征有两个重要的特征：
	- 在共享特征中，**modality-paired样本**会**离的更近**。
	- 两个模态当中的数据在共享特征中**分布会更加接近**。
4. 用到了shared-representation的最基本想法，不过并不是很符合我们要的场景。

### [Towards End-to-End Audio-Sheet-Music Retrieval][6]
- NIPS2016 workshop
- 同样是学习共享特征空间，没有特别的创新

### [Lip Reading Sentences in the Wild][7]
- Nov 2016, work from Google
- 唇语可以继续深挖。

## ICLR2017 workshop
### [Gated Multimodal Units for Information Fusion][8]
- 工作的想法比较简单，首先将**文字**与**图像**的特征**各自提取出来**；然后通过一个**gate**结构**有选择的进行融合**，得到最终用于分类的特征。
- 本工作有点老板那个意思，不过并没有多层的交替融合。

### [Is a picture worth a thousand words? A Deep Multi-Modal Fusion Architecture for Product Classification in e-commerce][9]
- 从题目来看非常契合我们要的东西。

### [CAT2VEC: LEARNING DISTRIBUTED REPRESENTATION OF MULTI-FIELD CATEGORICAL DATA][10]
- 标题不错。

### [Multi-modal Variational Encoder-Decoders][11]
- 标题不错。

### [Joint Multimodal Learning with Deep Generative Models][12]
- 标题不错。


## Review of multi-modal
### [Multi-View Representation Learning: A Survey from Shallow Methods to Deep Methods][13]
- October 2016

## Faculty
### ICML workshop: [Multi-View Representation Learning][14]
### [Karen Livescu][15]: 2293 cited
- [Multi-view Recurrent Neural Acoustic Word Embeddings][16]
	- ICLR2017 poster
	- 基本的想法很简单：用LSTM分别将**单词的语音序列**和**字母序列**映射到**同一个向量空间**中，然后在这个空间中用**triplet loss**对整个网络求导
	- 亮点：比较有意思的是，在这种多模态的学习下，利用**text来查询acoustic**，竟然可以得到**比acoustic-acoustic查询更高的AP值**。

- [Discriminative Acoustic Word Embeddings: Recurrent Neural Network-Based Approaches][17]

### [Dataset-Zhiyao Duan][18]
- [sound-visual association][19]
- [visually pitch analysis][20]
	- 可以借鉴的工作，视频的作用在于去除纯音频数据中的某些不确定性；与唇语有某种相似性。






[1]:	http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Rastegar_MDL-CW_A_Multimodal_CVPR_2016_paper.html
[2]:	https://arxiv.org/abs/1608.07017
[3]:	http://vis.csail.mit.edu/
[4]:	https://arxiv.org/abs/1608.06993
[5]:	https://arxiv.org/abs/1612.07976
[6]:	https://arxiv.org/abs/1612.05070
[7]:	https://arxiv.org/abs/1611.05358v1
[8]:	https://arxiv.org/abs/1702.01992
[9]:	https://arxiv.org/abs/1611.09534
[10]:	https://openreview.net/pdf?id=HyNxRZ9xg
[11]:	https://arxiv.org/abs/1612.00377
[12]:	https://arxiv.org/abs/1611.01891
[13]:	https://arxiv.org/abs/1610.01206
[14]:	http://ttic.uchicago.edu/~wwang5/ICML2016_MVRL/
[15]:	http://ttic.uchicago.edu/~klivescu/
[16]:	https://arxiv.org/abs/1611.04496
[17]:	https://arxiv.org/abs/1611.02550
[18]:	https://arxiv.org/abs/1612.08727
[19]:	http://www.ece.rochester.edu/projects/air/publications/li2017see.pdf
[20]:	http://www.ece.rochester.edu/projects/air/publications/dinesh2017visually.pdf
