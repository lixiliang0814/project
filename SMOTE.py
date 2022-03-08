import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
'''进行交叉验证'''
from sklearn.linear_model import LogisticRegression
'''进行逻辑线性回归'''
from sklearn.metrics import recall_score
'''进行计算召回值'''
import numpy as np
import itertools
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

data =pd.read_csv(r'D:\跟着迪哥学python相关数据与代码\第6章：信用卡欺诈检测\逻辑回归-信用卡欺诈检测\creditcard.csv')
columns =data.columns
features_columns = columns.delete(len(columns)-1)
features =data[features_columns]
labels =data['Class']
features_train,features_test,labels_train,labels_test =train_test_split(features,labels,test_size=0.3,random_state=0)
oversampler=SMOTE(random_state=0)
os_features,os_labels = oversampler.fit_resample(features_train,labels_train)
print(len(os_labels[os_labels==1]))
os_features=pd.DataFrame(os_features)
os_labels=pd.DataFrame(os_labels)
best_c =printing_Kfold_scores(os_features,os_labels)
lr=LogisticRegression(C =best_c,penalty='l1',solver='liblinear')
lr.fit(os_features,os_labels.values.ravel())
y_pred =lr.predict(features_test.values)
cnf_matrix=confusion_matrix(labels_test,y_pred)
np.set_printoptions(precision=2)
print('召回率:',cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
class_name=[0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_name,title='Confusion matrix')
plt.show()