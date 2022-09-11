# Fault Diagnosis Based on Machine Learning 基于机器学习的故障诊断



### **代码环境**

代码是在python 3.9.6 的编译下执行的：

各种主要库版本：

numpy == 1.23.1

matplotlib == 3.5.2

scipy == 1.8.1

sklearn == 1.1.1

---

## 描述

本人硕士期间的研究方向是机械故障诊断，侧重于机器学习和深度学习在故障诊断中的应用。在学习李航老师的《统计学方法》中，尝试着如何将常见的机器学习方法应用于机械故障诊断中。

- 数据描述

  **Mnist数据**：10类手写体数字(0-9)

  **CWRU轴承故障数据**：美国凯斯西储大学的轴承故障数据，包含正常工况，不通尺寸的内圈、外圈、滚动体故障，下载网址：https://engineering.case.edu/bearingdatacenter/download-data-file (Access 2022.09)

  在fault_data文件夹下：选取了1750 rpm/min 下的正常、0.007in 内圈、外圈、滚动体故障，四类工况构造数据集，每种工况400 个样本，其中CSWU_4Class大小1600x1024，直接使用了振动信号作为特征；CSWUFeature大小为1600x24，通过提取了24个特征指标。

- 文件描述

> ----
>
> Chapter1 Gradient descent.ipynb												 梯度下降法 \
> Chapter2-1 Perceptron.ipynb														感知机算法\
> Chapter2-2 Perceptron_Iris_classify.ipynb									基于感知器的Iris数据分类\
> Chapter2-3 Perceptron_Mnist_classify.ipynb							    基于感知器的Mnist数据分类\
> Chapter3-1 K_nearest_neighbor.ipynb										  KNN算法\
> Chapter3-2 KNN_Mnist_Classify.ipynb										 基于KNN的Mnist数据分类\
> Chapter3-3 KNN_CWRU_Fault_Diagnosis.ipynb						  基于KNN的CWRU故障诊断\
> Chapter4-1 Naive_Bayes.ipynb													 朴素贝叶斯算法\
> Chapter4-2 Naive_Bayes_Mnist_Classify.ipynb							基于朴素贝叶斯的Mnist数据分类\
> Chapter4-3 Naive_Bayes_CWRU_Fault_Diagnosis.ipynb			 基于朴素贝叶斯的CWRU故障诊断\
> Chapter5-1 Decision_Tree.ipynb													决策树算法\
> Chapter5-2 Decision_Tree_Mnist_Classify.ipynb						   基于决策树的Mnist数据分类\
> Chapter6-1 Logistic_Regression.ipynb										 逻辑斯蒂回归算法\
> Chapter6-2 Maximum_Entropy_Model.ipynb								最大熵模型算法\
> Chapter6-3 Logistic_Regression_Mnist_Classify.ipynb				 基于逻辑斯蒂回归算法的Mnist数据分类\
> Chapter6-4 Max_Entropy_Mnist_Classify.ipynb							基于最大熵的Mnist数据分类\
> Chapter6-5 Logistic_Regression_CWRU_Fault_Diagnosis.ipynb 基于逻辑斯蒂回归的CWRU故障诊断\
> Chapter7-1 Support_Vector_Machine.ipynb								 支持向量机算法\
> Chapter7-2 SVM_Mnist_Classify.ipynb										 基于SVM的Mnist数据分类\
> Chapter7-3 SVM_CWRU_Fault_Diagnosis.ipynb						  基于SVM的CWRU故障诊断\
> Chapter8-1 Boosting.ipynb															Boosting算法\
> Chapter8-2 AdaBoosting_Mnist_Classify.ipynb							 基于AdaBoosting的Mnist数据分类\
> Chapter8-3 AdaBoosting_CWRU_Fault_Dignosis.ipynb				基于AdaBoosting的CWRU的故障诊断\
> Chapter9-1 EM.ipynb																	  EM算法\
> Chapter9-2 EM_Gaussian_Mixture_model.ipynb						   基于EM算法的模型\
> Chapter10-1 Hidden_Markov_Model.ipynb									隐马尔可夫算法\
> Chapter10-2 HMM_NLP.ipynb														HMM在自然语言处理NLP的应用\
> Chapter11-1 Condition_Random_Field.ipynb								条件随机场算法\
> Appendix_CWRU_DCNN.ipynb													  基于DCNN的轴承故障诊断方法\

---

### 参考文献

1 深度学习之眼 《统计学方法》课程——正版购买

2 李航 《统计学方法》(第2版)

3 周志华 《机器学习》

4 微信公众号《机器学习初学者》- https://github.com/fengdu78/lihang-code











