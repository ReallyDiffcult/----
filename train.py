#encoding:utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#读取文件
def read_csv(name):

    return pd.read_csv(name, encoding="gbk")
#筛选特征
def screen_features(df):
        df_train  = pd.DataFrame(df).drop(['id',u'体检日期',u'乙肝表面抗原',u'乙肝表面抗体',u'乙肝核心抗体',u'乙肝e抗原',u'乙肝e抗体'],axis=1)
        # df_train = df_train[df_train[u'血糖'] >= 20]

        # # 得到相关系数前二十的的特征---------------------------------------------------------------------------------
        # features_list = [u'甘油三酯', u'年龄', u'低密度脂蛋白胆固醇', u'总胆固醇', u'*碱性磷酸酶', u'尿素', u'血红蛋白',
        #                  u'红细胞平均血红蛋白浓度', u'*丙氨酸氨基转换酶', u'*r-谷氨酰基转换酶', u'红细胞计数', u'红细胞压积',
        #                  u'*天门冬氨酸氨基转换酶', u'肌酐', u'白细胞计数', u'红细胞平均血红蛋白量', u'*球蛋白', u'中性粒细胞%',
        #                  u'*总蛋白', u'血糖']
        # df_train = pd.DataFrame(df)[features_list]
        #--------------------------------------------------------------------------------------------------
        #填充空值
        fill_nan(df_train)
        #删除异常值
        # print len(df_train),0
        df_train = df_train[df_train[u'血糖'] <= 20]
        # # print df_train.shape,1
        # df_train = df_train[df_train[u'*天门冬氨酸氨基转换酶'] <= 200]
        # # # df_train.to_csv("tt.csv",index=False)
        # # print len(df_train),2
        # df_train = df_train[df_train[u'*丙氨酸氨基转换酶'] <= 200]
        # # print len(df_train),3
        # df_train = df_train[df_train[u'*r-谷氨酰基转换酶'] <= 500]
        # # print len(df_train),4
        # df_train = df_train[df_train[u'白球比例'] <= 3]
        # df_train = df_train[df_train[u'*球蛋白'] <= 60]
        # # print len(df_train),5
        # df_train = df_train[df_train[u'总胆固醇'] <= 10]
        # print len(df_train),6
        # df_train = df_train[df_train[u'甘油三酯'] <= 20]#######
        # print len(df_train),7
        # df_train = df_train[df_train[u'肌酐'] <= 140]
        # print len(df_train),8
        # df_train = df_train[df_train[u'白细胞计数'] <= 20]
        # print len(df_train),9
        # df_train = df_train[df_train[u'白细胞计数'] >= 3]
        # df_train = df_train[df_train[u'血小板计数'] <= 600]
        # print len(df_train),10
        # df_train = df_train[df_train[u'血小板体积分布宽度'] <= 22.5]
        # print len(df_train),11
        # df_train = df_train[df_train[u'嗜酸细胞%'] <= 20]
        # print len(df_train),12
        # df_train = df_train[df_train[u'中性粒细胞%'] >= 30]
        # df_train = df_train[df_train[u'淋巴细胞%'] <= 70]
        # print len(df_train),13
        return  df_train
def screen(df):
        df_train = pd.DataFrame(df).drop(['id', u'体检日期', u'乙肝表面抗原', u'乙肝表面抗体', u'乙肝核心抗体', u'乙肝e抗原', u'乙肝e抗体'], axis=1)
        fill_nan(df_train)
        return df_train
#填充空值
def fill_nan(df):
    return pd.DataFrame(df).fillna(-999,inplace=True)
def acc(pre, y_test):
    return ((pre - y_test)**2).sum()/(2*len(pre))
#处理特征
def handle_features():
        #读入数据
        train_data = read_csv("train.csv")
        test_data = read_csv("test.csv")
        # train_data = train_data[train_data[u'血糖'] <= 15]
        # ------得到特征值
        d = screen_features(train_data)
        train_X = d.ix[:, :-1]
        train_Y = d.ix[:, -1]

        test_X = screen(test_data).ix[:, :]

        # ------清洗数据

        # 填充缺失值
        # train_X = fill_nan(train_X)
        # test_X = fill_nan(test_X)
        # 离散男女性别
        sex_feature = train_X.ix[:, 0]
        sex_features = pd.get_dummies(sex_feature)

        sex_feature1 = test_X.ix[:, 0]
        sex_features1 = pd.get_dummies(sex_feature1)
        # del sex and contact sex_features
        train_X = sex_features.join(train_X).drop([u'性别'], axis=1)
        test_X = sex_features1.join(test_X).drop([u'性别'], axis=1)
        # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3,
        #                                                     random_state=0)
        return train_X, train_Y,test_X
#模型训练
def train_model(X_train,  X_test, y_train, y_test, name = 'xgboost'):

        #xgboost模型
        if name =='xgboost':
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)

                param = {'silent': 1, 'objective': 'count:poisson', 'booster': 'gbtree', 'base_score': 1,
                         'max_depth': 3, 'eta': 0.3,

                         }
                watchlist = [(dtest, 'eval'), (dtrain, 'train')]
                num_round = 50
                bst = xgb.train(param, dtrain, num_round, watchlist)
                preds = bst.predict(xgb.DMatrix(X_test))
                # labels = dtest.get_label()
                # print "real:",acc(preds, y_test)
                return acc(preds, y_test)

                #
                # # 计算 auc 分数、预测
                # preds = xlf.predict(X_test)
                # print type(preds)
                # print acc(preds, y_test)
                # # plt.plot(range(0, len(preds)), y_test)
                # plt.plot(range(0,len(preds)), preds)
                # plt.show()
        #随机森林模型
        if name == 'rf':
                lsvc = RandomForestRegressor(n_estimators=300, n_jobs=-1)
                lsvc = GradientBoostingRegressor(n_estimators=20)
                lsvc.fit(X_train, y_train)
                preds = lsvc.predict(X_test)
                print acc(preds, y_test)
                # plt.plot(range(0, len(preds)), y_test)
                # plt.plot(range(0, len(preds)), preds)
                # plt.show()
#模型预测
def train_and_predict(train_X, train_Y,test_X):
        dtrain = xgb.DMatrix(train_X, label=train_Y)
        # dtest = xgb.DMatrix(test_X)
        #
        param = {'silent': 1, 'objective': 'count:poisson', 'booster': 'gbtree', 'base_score': 1,'max_depth':3,'eta':0.3}
        # watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 50
        bst = xgb.train(param, dtrain, num_round)
        preds = bst.predict(xgb.DMatrix(test_X))

        pd.DataFrame(preds).to_csv('result.csv',index=False)
#线下检验
def select_testfile():
        train_data = read_csv("train.csv")
        #----------------------------------------------------------------
        # print "tol:",train_data.shape
        # train_data = train_data[train_data[u'血糖'] > 15]
        d = screen_features(train_data)
        train_X = d.ix[:, :-1]
        train_Y = d.ix[:, -1]
        # train_X = fill_nan(train_X)
        # print train_X.shape, train_Y.shape
        #
        sex_feature = train_X.ix[:, 0]
        sex_features = pd.get_dummies(sex_feature)

        # del sex and contact sex_features
        train_X = sex_features.join(train_X).drop([u'性别'], axis=1)

        #-----------------------------------------------------------------------------------------------------------

        # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3,
        #                                                     random_state=0)
        # res = train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        # print res
#-------------交叉验证-----------------------------------------------------
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10,shuffle=True)
        train_X = np.array(train_X)
        train_Y =np.array(train_Y)
        result = []
        for train_index, test_index in kf.split(train_X):
                X_train, X_test = train_X[train_index], train_X[test_index]
                y_train, y_test = train_Y[train_index], train_Y[test_index]
                res = train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
                result.append(res)
        # print result
        print "test:",np.mean(result)
def cal_acc():
        a = read_csv('result.csv')
        b = read_csv('final.csv')
        formater = "{0:.03f}".format
        a.applymap(formater)
        b.applymap(formater)

        print "real",acc(a.values,b.values)
        # plt.plot(a.values)
        # plt.plot(b.values)
        # plt.show()

def add_features(train, test):
        train[u'霉'] = train[u'*天门冬氨酸氨基转换酶'] + train[u'*丙氨酸氨基转换酶'] + train[u'*碱性磷酸酶'] + train[u'*r-谷氨酰基转换酶']
        test[u'霉'] = test[u'*天门冬氨酸氨基转换酶'] + test[u'*丙氨酸氨基转换酶'] + test[u'*碱性磷酸酶'] + test[u'*r-谷氨酰基转换酶']

        train[u'尿酸/肌酐'] = train[u'尿酸'] / train[u'肌酐']
        test[u'尿酸/肌酐'] = test[u'尿酸'] / test[u'肌酐']

        train[u'肾'] = train[u'尿酸'] + train[u'尿素'] + train[u'肌酐']
        test[u'肾'] = test[u'尿酸'] + test[u'尿素'] + test[u'肌酐']

        train[u'红细胞计数*红细胞平均血红蛋白量'] = train[u'红细胞计数'] * train[u'红细胞平均血红蛋白量']
        test[u'红细胞计数*红细胞平均血红蛋白量'] = test[u'红细胞计数'] * test[u'红细胞平均血红蛋白量']

        train[u'红细胞计数*红细胞平均血红蛋白浓度'] = train[u'红细胞计数'] * train[u'红细胞平均血红蛋白浓度']
        test[u'红细胞计数*红细胞平均血红蛋白浓度'] = test[u'红细胞计数'] * test[u'红细胞平均血红蛋白浓度']

        train[u'红细胞计数*红细胞平均体积'] = train[u'红细胞计数'] * train[u'红细胞平均体积']
        test[u'红细胞计数*红细胞平均体积'] = test[u'红细胞计数'] * test[u'红细胞平均体积']

        # train['红细胞平均血红蛋白量*红细胞平均血红蛋白浓度'] = train['红细胞平均血红蛋白量'] * train['红细胞平均血红蛋白浓度']
        # test['红细胞平均血红蛋白量*红细胞平均血红蛋白浓度'] = test['红细胞平均血红蛋白量'] * test['红细胞平均血红蛋白浓度']
        #
        train[u'嗜酸细胞'] = train[u'白细胞计数'] * train[u'嗜酸细胞%']
        test[u'嗜酸细胞'] = test[u'白细胞计数'] * test[u'嗜酸细胞%']
        return train, test

if __name__ == '__main__':
        # X_train, X_test, y_train, y_test =  handle_features()
        # X_tes1t,y_test1 = select_testfile()

        # X_test = pd.concat((X_test,X_tes1t),axis=0)
        # y_test = pd.concat((y_test,y_test1),axis=0 )

        # select_testfile()
        # print X_test
        # train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test = y_test)
        # print X_test.shape, X_train.shape
#--------------线下测试---------------------------------
        # select_testfile()
#-----------------------完成对样本预测-------------------------------------------------
        #线下测试
        select_testfile()
        #线上测试
        train_X, train_Y, test_X = handle_features()
        train_X, test_X = add_features(train_X, test_X)
        train_and_predict(train_X, train_Y, test_X)
        cal_acc()
        # #0.830
        #0.85649574832
