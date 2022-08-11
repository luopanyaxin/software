import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch
import numpy as np
import joblib
#------ 侧边框的东西-------
st.sidebar.expander('')
st.sidebar.expander('')
st.sidebar.subheader('在下方调节你的参数')
moxing=st.sidebar.radio('请选择是否需要使用已经训练好的模型',['否','是'])
number = st.sidebar.number_input('请输入迭代次数:')
epoch = int(number)
learning = st.sidebar.number_input('请输入学习率:')
learning_rate = learning
seq=st.sidebar.selectbox('请输入考虑之前数据的组数',list(range(2,6)))
fangshi = st.sidebar.selectbox('请选择数据集划分比例', ['8:2', '7:3', '6:4'])
# -------------------
#----- 模型系数及一些超参数-------
input_dim = 7  # 数据的特征数
hidden_dim = 64  # 隐藏层的神经元个数
num_layers = 1  # LSTM的层数
output_dim = 1  # 预测值的特征数
pre_days = 7  # 以1周的数据为一组
seq =seq # 用1周前的数据去预测现在数据
batch_size = 128 # 每组数据量
num_epochs = epoch # 模型遍历次数
# ---------------------
#读数据
df_main = pd.read_csv('data_1.csv')
sel_col = ["power_consumption", "low_temp", "high_temp", "kind", "wind", "level", "holiday"]

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
for col in sel_col:
    df_main[col] = scaler.fit_transform(df_main[col].values.reshape(-1, 1))
df_main['target'] = df_main['power_consumption'].shift(-1)


# LSTM模型 输入为7输出为1 32个神经元
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout = 0.5)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        return out


# 创建两个列表，用来存储数据的特征和标签
data_feat, data_target = [], []

for index in range(len(df_main) - seq):
    # 构建特征集
    data_feat.append(df_main[["power_consumption", "low_temp", "high_temp", "kind", "wind", "level", "holiday"]][
                     index: index + seq].values)
    # 构建target集
    data_target.append(df_main['target'][index:index + seq])

# 将特征集和标签集整理成numpy数组
data_feat = np.array(data_feat)
data_target = np.array(data_target)
# 这里按照8:2的比例划分训练集和测试集
if fangshi=="8:2":
    test_set_size = 122  # np.round(1)是四舍五入，
    train_size = data_feat.shape[0] - (test_set_size)
elif fangshi=="7:3":
    test_set_size = 183  # np.round(1)是四舍五入，
    train_size = data_feat.shape[0] - (test_set_size)
elif fangshi=="6:4":
    test_set_size = 244  # np.round(1)是四舍五入，
    train_size = data_feat.shape[0] - (test_set_size)
trainX = torch.from_numpy(data_feat[:train_size].reshape(-1, seq, 7)).type(torch.Tensor)
testX = torch.from_numpy(data_feat[train_size:].reshape(-1, seq, 7)).type(torch.Tensor)
trainY = torch.from_numpy(data_target[:train_size].reshape(-1, seq, 1)).type(torch.Tensor)
testY = torch.from_numpy(data_target[train_size:].reshape(-1, seq, 1)).type(torch.Tensor)
xieruwenjian=torch.from_numpy(data_feat[:10].reshape(-1, seq, 7)).type(torch.Tensor)
torch.save(xieruwenjian,'任务四的测试')
print('x_train.shape = ', trainX.shape)
print('y_train.shape = ', trainY.shape)
print('x_test.shape = ', testX.shape)
print('y_test.shape = ', testY.shape)

# dataloader读入数据
train = torch.utils.data.TensorDataset(trainX, trainY)
test = torch.utils.data.TensorDataset(testX, testY)
train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)
# 封装模型
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

# 定义优化器和损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化算法
# 模型评估条件
loss_fn = torch.nn.MSELoss(size_average=True)  # 使用均方差作为损失函数



st.markdown('''## <b style="color:white;"><center>1.任务简介</center></b>''', unsafe_allow_html=True)  # 展示任务
st.markdown('''### <b style="color:white;">需求分析</b>''', unsafe_allow_html=True)

st.write('''<b style="color:white
;">建立电力预测模型，根据以往的用户用电数据预测未来的用户用电情况，保证企业能够及时调整电力市场的需求水平、需求时间，以良好的服务质量满足用户合理用电的要求，实现电力供求之间的相互协调。</b>''',
         unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">方案设计</b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;">使用1500
名用户在一年半内的用电情况数据，利用各用户一年半内的用电情况、一年半内的天气情况和一年半内的节假日情况，在完成对数据的预处理和可视化分析后，对其中特征进行选取后，使用LSTM
对数据进行模型训练，通过调整参数以及评估分析，达到一个较优的模型效果。作为最终的企业电力营销模型。</b>''', unsafe_allow_html=True)

st.markdown('''## <b style="color:white;"><center>2.模型数据</center></b>''', unsafe_allow_html=True)  # 展示任务
st.write(
    '''<b style="color:white;">采用了约1500名用户在一年半内的用电情况数据，总数据包括三个文件，分别介绍了各用户一年半内的用电情况、一年半内的天气情况和一年半内的节假日情况。</b>''',
    unsafe_allow_html=True)
df = pd.read_csv('train_data.csv')
st.dataframe(df)
st.markdown('''### <b style="color:white;">数据预处理</b>''', unsafe_allow_html=True)
st.write("""<b style="color:white;">
★ 合并数据集 </b>""", unsafe_allow_html=True)
st.write("""<b style="color:white;">
★ 填充缺失值 </b>""", unsafe_allow_html=True)
st.write("""<b style="color:white;">
★ 字符型特征数值化 </b>""", unsafe_allow_html=True)

st.markdown('''## <b style="color:white;"><center>3.数据可视化分析</center></b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;">从图中可以得出用户id和其余变量之间的相关性系数为0，因此不能将不同用户之间的数据一起进行模型训练。同时，可以得出其余变量对用户用电量都有一定的影响，因此我们不对特征进行筛除，将7个变量对应的数据统一运用到模型训练中。
除此之外，根据数据的特征，我们选取了LSTM模型作为预测模型。</b>''', unsafe_allow_html=True)
st.markdown('''#### <b style="color:white;"><center>各用户的用电量示意图</center></b>''', unsafe_allow_html=True)
st.image('居民用电情况.jpg')
# st.image('''<b style="color:white;"><center>'居民用电情况.jpg'</center></b>''',unsafe_allow_html=True)
st.markdown('''#### <b style="color:white;"><center>各特征之间相关性系数热力图</center></b>''', unsafe_allow_html=True)
st.image('相关系数矩阵.png')

st.markdown('''## <b style="color:white;"><center>4.模型实现与结果</center></b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;">与其他神经网络不同，LSTM通过在原始RNN隐藏层增加单元状态c来保持长期状态，且解决了RNN
存在的长期依赖问题。由于其兼具非线性和时序性的特点，被逐渐应用于电力预测领域并取得了良好的效果。LSTM学习过程由信号的正向传播和误差反向传播两个过程组成。</b>''', unsafe_allow_html=True)
st.markdown('''#### <b style="color:white;"><center>模型结构图</center></b>''', unsafe_allow_html=True)
st.image('模型流程图.png')


# 数据处理
if moxing=='否':
# 训练模型
    if epoch!=0:
        st.markdown('''#### <b style="color:white;"><center>训练过程损失曲线</center></b>''', unsafe_allow_html=True)
    hist = np.zeros(num_epochs)
    ls = []
    for t in range(num_epochs):
        y_train_pred = model(trainX)
        loss = loss_fn(y_train_pred, trainY)
        ls.append(loss.item())
        if t % 10 == 0 and t != 0:  # 每训练十次，打印一次均方差
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        # 梯度归零
        optimiser.zero_grad()
        # Backward
        loss.backward()
        # 更新参数
        optimiser.step()
    loss_data = pd.DataFrame(
        ls,
        columns=["loss值"]
    )
    st.line_chart(loss_data)
else:

    st.markdown('''#### <b style="color:white;"><center>训练过程损失曲线</center></b>''', unsafe_allow_html=True)
    ls = [0.5947179794311523, 0.4507795572280884, 0.32666218280792236, 0.22864139080047607, 0.1810951679944992,
            0.20072788000106812, 0.219834104180336, 0.1996694654226303, 0.16286492347717285, 0.1347438246011734,
            0.12425760924816132, 0.12667830288410187, 0.1316240429878235, 0.13067269325256348, 0.1217602789402008,
            0.10766869783401489, 0.0929853618144989, 0.08185668289661407, 0.07632635533809662, 0.07534956187009811,
            0.07535826414823532, 0.07276003807783127, 0.06653910875320435, 0.05853329598903656, 0.051560405641794205,
            0.047397494316101074, 0.04598104581236839, 0.04590829461812973, 0.04557082802057266, 0.04411137476563454,
            0.041716318577528, 0.03926895931363106, 0.03768862038850784, 0.03732573613524437, 0.037740547209978104,
            0.038026466965675354, 0.037466708570718765, 0.03600345924496651, 0.03416650742292404, 0.032608285546302795,
            0.03166688606142998, 0.031239405274391174, 0.03095424361526966, 0.03045312874019146, 0.029595160856842995,
            0.02849569357931614, 0.0274167750030756, 0.026590043678879738, 0.02607543207705021, 0.02574039250612259,
            0.02537364326417446, 0.02484879642724991, 0.024207692593336105, 0.023601701483130455, 0.023153360933065414,
            0.02285877615213394, 0.022600146010518074, 0.022242626175284386, 0.021733202040195465, 0.02113015204668045,
            0.020548896864056587, 0.020070672035217285, 0.019687429070472717, 0.01932711713016033, 0.018933316692709923,
            0.018521204590797424, 0.018156995996832848, 0.017887337133288383, 0.017692873254418373,
            0.017508702352643013, 0.017285726964473724, 0.017031125724315643, 0.016792116686701775,
            0.016604460775852203, 0.01646040380001068, 0.01632745936512947, 0.016191935166716576, 0.01607416197657585,
            0.01599947363138199, 0.015963546931743622, 0.015934403985738754, 0.015886016190052032, 0.015821749344468117,
            0.015762269496917725, 0.01571686938405037, 0.015674743801355362, 0.015623818151652813, 0.015568578615784645,
            0.015522986650466919, 0.015490490943193436, 0.015459226444363594, 0.015417144633829594, 0.01536527182906866,
            0.015312861651182175, 0.015263956040143967, 0.015214378945529461, 0.015160984359681606,
            0.015107913874089718, 0.01506117731332779, 0.015020622871816158, 0.014980807900428772, 0.014938442967832088,
            0.01489558070898056, 0.01485530473291874, 0.014817298389971256, 0.014779355376958847, 0.014741463586688042,
            0.014706015586853027, 0.014674236066639423, 0.014644456095993519, 0.014614459127187729, 0.0145840710029006,
            0.01455447357147932, 0.014525962062180042, 0.014497620984911919, 0.01446906104683876, 0.014441151171922684,
            0.014414721168577671, 0.01438938733190298, 0.014364199712872505, 0.014338935725390911, 0.014314086176455021,
            0.0142898578196764, 0.014265892095863819, 0.014241994358599186, 0.014218521304428577, 0.014195832423865795,
            0.014173734001815319, 0.014151797629892826, 0.0141299432143569, 0.014108359813690186, 0.01408708281815052,
            0.014065933413803577, 0.014044865034520626, 0.01402406394481659, 0.014003628864884377, 0.013983413577079773,
            0.013963254168629646, 0.013943170197308064, 0.013923232443630695, 0.013903412036597729,
            0.013883628882467747, 0.01386391930282116, 0.013844364322721958, 0.013824944384396076, 0.013805573806166649,
            0.01378621719777584, 0.013766906224191189, 0.013747649267315865, 0.01372840628027916, 0.013709168881177902,
            0.013689970597624779, 0.013670817017555237, 0.013651669025421143, 0.013632498681545258,
            0.013613317161798477, 0.013594131916761398, 0.013574929907917976, 0.013555699028074741, 0.01353644859045744,
            0.013517189770936966, 0.013497901149094105, 0.013478566892445087, 0.013459189794957638,
            0.013439776375889778, 0.013420315459370613, 0.013400801457464695, 0.013381239026784897, 0.01336162630468607,
            0.01334195677191019, 0.013322215527296066, 0.01330240722745657, 0.013282528147101402, 0.013262572698295116,
            0.01324253436177969, 0.01322241686284542, 0.013202212750911713, 0.013181917369365692, 0.013161523267626762,
            0.013141029514372349, 0.013120434246957302, 0.013099730014801025, 0.013078916817903519,
            0.013057989999651909, 0.013036944903433323, 0.013015778735280037, 0.012994487769901752,
            0.012973069213330746, 0.012951517477631569, 0.012929831631481647, 0.012908006086945534,
            0.012886039912700653, 0.012863928452134132, 0.0128416633233428, 0.01281924732029438, 0.012796674855053425,
            0.012773944064974785, 0.012751052156090736, 0.012727992609143257, 0.012704764492809772]
    loss_data = pd.DataFrame(
        ls,
        columns=["loss值"]
    )
    st.line_chart(loss_data)
st.markdown('''#### <b style="color:white;"><center>测试集用户用电实际值与预测值对比曲线</center></b>''', unsafe_allow_html=True)
if moxing=='否':
# 模型预测
    y_test_pred = model(testX)
    list1 = []
    list2 = []
    y_test_pred = model(testX)
    y_train_pred=model(trainX)
    loss1 = loss_fn(y_test_pred[:-pre_days], testY[:-pre_days]).item()
    print(loss1)

    pred_value = y_train_pred.detach().numpy()[:, -1, 0]
    true_value = trainY.detach().numpy()[:, -1, 0]
    pt=[]
    test_pre=y_test_pred.detach().numpy()[:, -1, 0]
    test_ture=testY.detach().numpy()[:, -1, 0]
    pt.append(test_pre)
    pt.append(test_ture)
    listb = [[r[col] for r in pt] for col in range(len(pt[0]))]
    chart_data = pd.DataFrame(
        listb,
        columns=["预测值", "实际值"]
    )
    st.line_chart(chart_data)

else:
    model=joblib.load('企业电力营销模型.mkl')
    list1 = []
    list2 = []
    y_test_pred = model(testX)
    y_train_pred = model(trainX)
    loss1 = loss_fn(y_test_pred[:-pre_days], testY[:-pre_days]).item()
    print(loss1)

    pred_value = y_train_pred.detach().numpy()[:, -1, 0]
    true_value = trainY.detach().numpy()[:, -1, 0]
    pt = []
    test_pre = y_test_pred.detach().numpy()[:, -1, 0]
    test_ture = testY.detach().numpy()[:, -1, 0]
    pt.append(test_pre)
    pt.append(test_ture)
    listb = [[r[col] for r in pt] for col in range(len(pt[0]))]
    chart_data = pd.DataFrame(
        listb,
        columns=["预测值", "实际值"]
    )
    st.line_chart(chart_data)
st.markdown('''#### <b style="color:white;"><center>上传需要预测的数据,可以得到预测值:</center></b>''', unsafe_allow_html=True)
file=st.file_uploader("", type=None, accept_multiple_files=False, key=None, \
                     help=None, on_change=None)
if file is not None:
    data=torch.load(file)
    pred=model(data)
    pred_value = pred.detach().numpy()[:, -1, 0]
    ptyu=[]
    ptyu.append(pred_value)
    listbyu = [[r[col] for r in ptyu] for col in range(len(ptyu[0]))]
    chart_data_yu = pd.DataFrame(
        listbyu,
        columns=["预测值"]
    )
    st.write(chart_data_yu)
    st.line_chart(chart_data_yu)

img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'

# 修改背景样式
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)
