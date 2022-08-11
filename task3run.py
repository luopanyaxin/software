from torch.autograd import Variable
import torch.nn as nn
from pandas import DataFrame
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("TKAgg")
import torch
def predict(df1,learning_rate,epoch,look_back,dijige):
    # 一、数据准备
    # print(df1)
    data = df1.values
    # print(datas)
    # 归一化处理，这一步必不可少，不然后面训练数据误差会很大，模型没法用
    max_value = np.max(data)
    min_value = np.min(data)
    scalar = max_value - min_value
    datas = list(map(lambda x: x / scalar, data))
    # print(datas)
    # 数据集和目标值赋值，dataset为数据，look_back为以几行数据为特征维度数量
    def creat_dataset(dataset, look_back):
        data_x = []
        data_y = []
        for i in range(len(dataset) - look_back):
            data_x.append(dataset[i:i + look_back])
            data_y.append(dataset[i + look_back])
        data_x.append(dataset[len(dataset) - look_back:len(dataset) - look_back + look_back])
        return np.asarray(data_x), np.asarray(data_y)  # 转为ndarray数据

    # 以2为特征维度，得到数据集
    dataX, dataY = creat_dataset(datas, look_back)
    # print(dataX)
    # print(dataY)
    train_size = int(len(dataX) * 0.7)
    x_train = dataX[:train_size]  # 训练数据
    y_train = dataY[:train_size]  # 训练数据目标值
    x_train = x_train.reshape(-1, 1, 2)  # 将训练数据调整成pytorch中lstm算法的输入维度
    y_train = y_train.reshape(-1, 1, 1)  # 将目标值调整成pytorch中lstm算法的输出维度
    # 将ndarray数据转换为张量，因为pytorch用的数据类型是张量
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    # 二、创建LSTM模型
    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()  # 面向对象中的继承
            self.lstm = nn.LSTM(2, 6, 2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
            self.out = nn.Linear(6, 1)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

        def forward(self, x):
            x1, _ = self.lstm(x)
            a, b, c = x1.shape
            out = self.out(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
            out1 = out.view(a, b, -1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
            return out1

    rnn = RNN()
    # 参数寻优，计算损失函数
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    # 三、训练模型
    ls = []
    for i in range(epoch):
        var_x = Variable(x_train).type(torch.FloatTensor)
        var_y = Variable(y_train).type(torch.FloatTensor)
        out = rnn(var_x)
        loss = loss_func(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))
        ls.append(loss.item()*0.1)
    # 画图 loss
    if dijige<10:
        loss_data = pd.DataFrame(
            ls,
            columns=["loss值"]
        )
        st.line_chart(loss_data)
    # 四、进行预测
    p_value = []
    for i in range(2):  # 预测用户后两次缴费金额
        dataX1 = dataX.reshape(-1, 1, 2)
        dataX2 = torch.from_numpy(dataX1)
        var_dataX = Variable(dataX2).type(torch.FloatTensor)
        pred = rnn(var_dataX)
        pred_test = pred.view(-1).data.numpy()  # 转换成一维的ndarray数据，这是预测值
        pred_value = list(map(lambda x: round(x * scalar), pred_test))
        # print(pred_value)
        p_value.append(pred_value[-1])
        p = list(map(lambda x: x / scalar, pred_value))
        dataX,y = creat_dataset(p, 2)
    # print("预测值:",p_value)
    return p_value
# 预测后更新的用户信息,写入
def new_user(total):
    # 数据写入
    import pandas as pd
    path = r'users.xlsx'
    df = pd.read_excel(path, usecols=None)  # 直接使用 read_excel() 方法读取, 不读取列名
    lines = df.values.tolist()
    user = []
    i = 0
    for line in lines:
        ls1 = []
        c = line[1] + 1  # 缴费次数加1
        m = line[2] + total[i][0]  # 金额加预测值
        ls1.append(line[0])
        ls1.append(c)
        ls1.append(m)
        user.append(ls1)
        i += 1
    print(user)
    # 数据写入
    # -*- coding: utf-8 -*-
    import pandas as pd
    def pd_toExcel(data, fileName):  # pandas库储存数据到excel
        ids = []
        counts = []
        prices = []
        for i in range(len(data)):
            ids.append(data[i][0])
            counts.append(data[i][1])
            prices.append(data[i][2])
        dfData = {  # 用字典设置DataFrame所需数据
            '用户编号': ids,
            '缴费次数': counts,
            '缴费金额': prices
        }
        df = pd.DataFrame(dfData)  # 创建DataFrame
        df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）

    fileName = 'user_pred.xlsx'
    pd_toExcel(user, fileName)
    return user
# -----------
def output(path):
    # df=data
    df = pd.read_excel(path, usecols=None)  # 直接使用 read_excel() 方法读取, 不读取列名
    lines = df.values.tolist()
    result = []
    avg_counts = 6.62
    avg_money = 702.31

    # 客户分类
    # 先除去已经是高价值型的客户
    df = pd.read_csv(r'居民客户的用电缴费习惯分析 2.csv',encoding='utf-8')
    high = []
    for i in range(100):
        if df['客户类型'][i] == "高价值型客户":
            high.append(df['用户编号'][i])
    # print(high)
    for line in lines:
        if line[0] not in high:
            ls = []
            if line[1] > avg_counts and line[2] > avg_money:
                type =  "高价值型客户"
                ls.append(str(line[0]))
                ls.append(line[2])
                ls.append(type)
                result.append(ls)

    # 按金额排序
    for i in range(len(result)):
        for j in range(i + 1, (len(result))):
            if result[i][1] < result[j][1]:
                result[i][1], result[j][1] = result[j][1], result[i][1]
                result[i][0], result[j][0] = result[j][0], result[i][0]
    # print(result)
    f_result = []
    for i in range(5):
        a = [i+1,result[i][0],result[i][2],result[i][1] ]
        f_result.append(a)
    print(f_result)
    # 数据写入
    import csv
    f = open(u'居民客户的用电缴费习惯分析 3.csv', 'w', encoding='utf-8-sig', newline='')
    csv_write = csv.writer(f)
    csv_write.writerow(['用户排名', '用户编号', '客户类型', '缴费金额'])
    for data in f_result:
        csv_write.writerow([data[0], data[1],data[2],data[3]])


# ----------一下是布局
# streamlit的页面布局顺序是与代码位置一致的，因此我先在最前面加一个大标题
# def task5():
st.sidebar.expander('')
st.sidebar.expander('')
st.sidebar.subheader('在下方调节你的参数')
wenjian = st.sidebar.radio('是否选择上传本地测试文件', ['是', '否'])
number = st.sidebar.number_input('请输入迭代次数:')
epoch=int(number)
learning=st.sidebar.number_input('请输入学习率:')
learning_rate=learning
look_back=st.sidebar.radio('请输入考虑之前数据的组数',list(range(2,5)))

st.markdown('''## <b style="color:white;"><center>1.任务简介</center></b>''', unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">需求分析</b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;">由测试数据集，采用时间序列分析方法，训练得到用户价值预测模型，预测出最有可能成为高价值客户的前五人，将结果以csv格式保存。</b>''', unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">方案设计</b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;">首先利用LSTM模型预对每位用户下一次缴费的金额进行时间序列预测，之后在结合任务1、2中所得的数据，确定出最有可能成为高价值类型客户的top5。</b>''', unsafe_allow_html=True)
st.markdown('''## <b style="color:white;"><center>2.原始数据</center></b>''',unsafe_allow_html=True)
st.write('''<b style="color:white;">本地数据是每位用户在2018和2019年期间的购电日期和每次购买金额的数据,您也可以选择上传自己的本地文件。展示如下：</b>''', unsafe_allow_html=True)


def main():
    st.dataframe(df)
    st.markdown('''## <b style="color:white;"><center>3.预测过程</center> </b>''', unsafe_allow_html=True)
    param = pd.read_excel('param.xlsx')
    param['迭代次数'][0] = epoch
    param['学习率'][0] = learning_rate
    st.markdown('''#### <b style="color:white;">LSTM模型的具体参数设置</b>''', unsafe_allow_html=True)
    if epoch != 0:
        st.dataframe(param)

        st.markdown('''#### <b style="color:white;"><center>训练过程损失曲线</center></b>''', unsafe_allow_html=True)
        # st.image("损失值.jpg")
        id = 1000000001
        data = []
        true_data = []
        for i in range(len(df['用户编号'])):
            if i != len(df['用户编号']) - 1:
                if df['用户编号'][i] == id:
                    true_data.append(int(df['缴费金额（元）'][i]))
                else:
                    id = df['用户编号'][i]
                    data.append(true_data)
                    true_data = []
                    true_data.append(int(df['缴费金额（元）'][i]))
            else:
                true_data.append(int(df['缴费金额（元）'][i]))
                data.append(true_data)
        total = []
        pred = []
        ls = []
        for i in range(len(data)):
            # print("user:", i+1,"训练过程的损失曲线")
            if i < 10:
                st.write(f'''<b style="color: white;">用户{i+1}训练过程的损失曲线</b>''' , unsafe_allow_html=True)
            df1 = DataFrame(data[i])
            ls = list(df1[0])
            if (len(df1) > 1):
                # learning_rate=0.0
                pred = predict(df1, learning_rate, epoch, look_back, i)
            else:
                pred = [ls[0], ls[0], ls[0]]
            total.append(pred)

        st.markdown('''#### <b style="color:white;"><center>用户缴费金额实际值与预测值对比曲线</center></b>''', unsafe_allow_html=True)
        st.image('预测值与真实值.png')

        st.write('''<b style="color:white;"> 在对用户下一次缴费金额预测完毕后，更新客户的数据信息，将预测的数据添加到原有的客户信息中，得到新的用户数据文件。</b>''',
                 unsafe_allow_html=True)
        new = new_user(total)
        st.write('''<b style="color:white;">\"user_pred.xlsx\"</b>''')
        st.dataframe(new)
        st.markdown('''## <b style="color:white;"><center>4.预测结果</center></b>''', unsafe_allow_html=True)
        path = r'user_pred.xlsx'
        output(path)
        result = pd.read_csv('居民客户的用电缴费习惯分析 3.csv')
        st.dataframe(result)
        st.write('''<b style="color:white;">点击链接可以下载表格</b>''', unsafe_allow_html=True)
        data = open('居民客户的用电缴费习惯分析 3.csv', 'rb').read()  # 以只读模式读取且读取为二进制文件
        b64 = base64.b64encode(data).decode('UTF-8')  # 解码并加密为base64
        href = f'<a href="data:file/data;base64,{b64}" download = "居民客户的用电缴费习惯分析 3.csv"> 下载 “居民客户的用电缴费习惯分析 3.csv” </a>'  # 定义下载链接，默认的下载文件名是myresults.xlsx
        st.markdown(href, unsafe_allow_html=True)  # 输出到浏览器
if wenjian=='是':
    st.write(''' <b style="color:white;"><center>请上传本地文件</center></b>''', unsafe_allow_html=True)
    file=st.file_uploader("", type=None, accept_multiple_files=False, key=None, \
                     help=None, on_change=None)
    if file is not None:
        df= pd.read_excel(file)
        main()
else:
    df = pd.read_excel('任务三data.xlsx')
    main()

# 修改背景样式
img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)