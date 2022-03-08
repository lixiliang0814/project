import itertools

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data =pd.read_csv(r'D:\跟着迪哥学python相关数据与代码\第6章：信用卡欺诈检测\逻辑回归-信用卡欺诈检测\creditcard.csv')
count_classes = pd.value_counts(data['Class'],sort=True).sort_index()
count_classes.plot(kind='bar')
plt.xlabel('class')
plt.ylabel('frequency')
""""
上面是对原始数据进行展示，对于Class来说，0代表数据正常，1代表数据出现异常
"""
from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
'''在数据中新建一个’nornAmount列，然后StandardScaler().fit_transform（）是对数据进行特征标准化，然reshape（-1,1）是对数据写成一列'''
data =data.drop(['Time','Amount'],axis=1)
X =data.iloc[:,data.columns!='Class']
y =data.iloc[:,data.columns =='Class']
number_records_frand =len(data[data.Class==1])
frand_indices =np.array(data[data.Class==1].index)
'''这个是异常样本的索引的值'''
normal_indices =np.array(data[data.Class==0].index)
random_normal_indices = np.random.choice(normal_indices,number_records_frand,replace=False)
'''由于要让异常样本和普通样本数据量相同，所以在正常数据中获取相同数量的正常样本'''
random_normal_indices = np.array(random_normal_indices)
under_sample_indices =np.concatenate([frand_indices,random_normal_indices])
'''将两个数据进行合并，然后下面在搜寻这个索引所对应的值'''
under_sample_data = data.iloc[under_sample_indices,:]
'''X为特征值，y为样本'''
X_undersample =under_sample_data.iloc[:,under_sample_data.columns!='Class']
y_undersample =under_sample_data.iloc[:,under_sample_data.columns == 'Class']
print("正常样本所占整体比例",len(under_sample_data[under_sample_data.Class ==0])/len(under_sample_data))
print("异常样本所占整体比例",len(under_sample_data[under_sample_data.Class ==1])/len(under_sample_data))
print('样本总数',len(under_sample_data))
'''以下进行数据分割'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=0)
'''这里对所有的数据进行了切分，很重要！！！！！！'''
print("原始训练集包含样本数量",len(X_train))
print('原始测试集包含样本数量',len(X_test))
print('原始样本总数',len(X_train)+len(X_test))
X_train_undersample,X_test_undersample,y_train_undersample,y_test_undersample =train_test_split(X_undersample,y_undersample,test_size=0.3,random_state=0)
print('下采集训练集包含样本数量',len(X_train_undersample))
print('下采集测试集包含样本数量',len(X_test_undersample))
print('下采集样本总数',len(X_test_undersample)+len(X_train_undersample))

from sklearn.model_selection import KFold
'''进行交叉验证'''
from sklearn.linear_model import LogisticRegression
'''进行逻辑线性回归'''
from sklearn.metrics import recall_score
'''进行计算召回值'''

def printing_Kfold_scores(x_train_data, y_train_data):
    # 导入 KFold的方式不同引起
    # from sklearn.cross_validation import KFold
    # fold = KFold(len(y_train_data),5,shuffle=False)
    '''表示划分为5块'''
    # from sklearn.model_selection import KFold
    fold = KFold(5, shuffle=False)

    # 定义不同力度的正则化惩罚力度
    c_param_range = [0.01, 0.1, 1, 10, 100]
    # 展示结果用的表格
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # k-fold 表示K折的交叉验证，这里会得到两个索引集合: 训练集 = indices[0], 验证集 = indices[1]
    j = 0
    # 循环遍历不同的参数
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('正则化惩罚力度: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []

        # 一步步分解来执行交叉验证
        ''''enumerate()是将数据自动添加索引'''
        '''这里是将交叉验证后的索引记录下来'''
        for iteration, indices in enumerate(fold.split(x_train_data)):
            # 指定算法模型，并且给定参数
            # lr = LogisticRegression(C = c_param, penalty = 'l1')
            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')

            # 训练模型，注意索引不要给错了，训练的时候一定传入的是训练集，所以X和Y的索引都是0
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            # 建立好模型后，预测模型结果，这里用的就是验证集，索引为1
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            # 有了预测结果之后就可以来进行评估了，这里recall_score需要传入预测值和真实值。
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            # 一会还要算平均，所以把每一步的结果都先保存起来。
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ': 召回率 = ', recall_acc)

        # 当执行完所有的交叉验证后，计算平均结果
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('平均召回率 ', np.mean(recall_accs))
        print('')

    # 找到最好的参数，哪一个Recall高，自然就是最好的了。
    best_c = results_table.loc[results_table['Mean recall score'].astype('float64').idxmax()]['C_parameter']

    # 打印最好的结果
    print('*********************************************************************************')
    print('效果最好的模型所选参数 = ', best_c)
    print('*********************************************************************************')

    return best_c


best_c =printing_Kfold_scores(X_train_undersample,y_train_undersample)
def plot_confusion_matrix(cm,classes,title ='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=0)
    plt.yticks(tick_marks,classes)
    thresh =cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment='center',
                 color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
'''建立了混淆矩阵的模型'''
import itertools
from sklearn.metrics import confusion_matrix
lr =LogisticRegression(C=best_c,penalty='l1', solver='liblinear')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_underexample=lr.predict(X_test_undersample.values)
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_underexample)
np.set_printoptions(precision=2)
print('召回值:',cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
class_names=[0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix,class_names,title='Confusion matrix')

'''使用原始数据的测试集进行分析'''
'''方法和上面一样，只是改变了数据集'''
lr =LogisticRegression(C=best_c,penalty='l1',solver='liblinear')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict((X_test.values))
cnf_matrix =confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
print('召回率为：',cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
class_names=[0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix,class_names,title='Confusion matrix')
plt.show()

lr = LogisticRegression(C =best_c,penalty='l1',solver='liblinear')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_underexample_proba =lr.predict_proba(X_test_undersample.values)
threshold=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plt.figure(figsize=(10,10))
j=1
for i in threshold:
    y_test_predictions_high_recall =y_pred_underexample_proba[:,1]>i
    plt.subplot(3,3,j)
    j +=1
    cnf_matrix =confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    print("召回率是：",cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    class_names=[0,1]
    plot_confusion_matrix(cnf_matrix,classes =class_names,title='Threshold>=%s'%i)
    plt.show()


