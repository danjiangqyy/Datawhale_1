2020/9

# EDA





## 1.1  EDA的作用：

- 熟悉数据集，了解数据集，对数据集进行验证来确定所获得数据集可以用于接下来的机器学习或者深度学习使用。
- 了解变量间的相互关系以及变量与预测值之间的存在关系。
- 引导数据科学从业者进行数据处理以及特征工程的步骤,使数据集的结构和特征集让接下来的预测问题更加可靠



## 1.2  这三天的目标以及内容：











## 1.3 代码示范和解释

1.3.1 导入库和读取文件

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')
```

```python
Train_data = pd.read_csv('./train.csv')
Test_data = pd.read_csv('./testA.csv')
```

1.3.2 观察数据

```python
Train_data.shape
Test_data.shape
Train_data.describe()          #相关统计量
Train_data.info()              #数据类型和特殊符号异常
```

1.3.3 缺失值观察

```python
Train_data.isnull().sum()
Test_data.isnull().sum()
```

```python
missing = data_train.isnull().sum()/len(data_train)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
```

<img src="/Users/chouyangyu/Desktop/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031313532373730382e706e67.png" alt="68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031313532373730382e706e67"  />

<u>目前观察得知</u>

<u>1:22列特征数值缺失。</u>

<u>2:没有特征数值缺失率超过50%。</u>

<u>3:除此之外，'policyCode' 这一栏特征的数值都是一，是唯一值</u>。



1.3.4 查看特征数值类型和对象类型

类别特征：数值关系/非数值关系（object/non-object)

数值特征：连续型/离散型

```python
#数值特征
numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
#类别特征
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
```



```python
category_fea
#
['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine']
```



- 划分数值型变量中的连续变量和离散型变量

```python
#过滤数值型类别特征
def get_numerical_serial_fea(data,feas):
  
    numerical_serial_fea = []            #连续
    numerical_noserial_fea = []          #离散
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea,numerical_noserial_fea
numerical_serial_fea,numerical_noserial_fea = get_numerical_serial_fea(data_train,numerical_fea)
```



离散型变量的特征数值单独找出来，自己观察，分析。

连续型变量的特征数值建议分布可视化观察。

```python
f = pd.melt(data_train, value_vars=numerical_serial_fea) #数据转换
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)  
#col_wrap 三列
#share 是否共享xy轴

g = g.map(sns.distplot, "value")
```





正态分布调整



非数值关系的特征数值 可以自己单独找出来观察，分析，因为每一个非数值的特征有自己的独特内涵。



















