import streamlit as st
import matplotlib

matplotlib.use("TKAgg")

import os
os.environ['NUMEXPR_MAX_THREADS'] = "10"

# --------如果有左边的侧栏运行以下程序
# # streamlit的页面布局顺序是与代码位置一致的，因此我先在最前面加一个大标题
# st.sidebar.expander('')
# st.sidebar.expander('')
# st.sidebar.title("电力用户分析导航")
# with st.sidebar:
#     ouline=st.button('-----设计流程-----')
#     thought=st.button('-----设计思想-----')
#     rw_1=st.button("------任务一-------",)
#     rw_2 = st.button("------任务二-------")
#     rw_3 = st.button("------任务三-------")
#     rw_4 = st.button("------任务四-------")
#     rw_5 = st.button("------任务五-------")
# if ouline:
#     st.markdown('''# <b style="color:white;"><center>设计流程</center></b>''', unsafe_allow_html=True)
#     st.image('设计流程.png')
# elif thought:
#     st.markdown('''# <b style="color:white;"><center>软件设计思想</center></b>''', unsafe_allow_html=True)
#     st.markdown('''### <b style="color:white;">在设计过程中，我们更看重的是模型的可拓展性和可维护性。我们通过自顶向下的设计，由于任务中的数据与时间序列有较强的相关性，因此我们选定了具有长期记忆能力的LSTM模型。同样，对用户进行分类时，我们需要将n个数据对象划分k类，保证同一类对象相似且不同类对象相似度小，因此我们采用了当前较为成熟的k-means算法。</b>''',
#              unsafe_allow_html=True)
# -------以上是有侧栏
# ----------以下是没有侧栏
# st.markdown('''# <b style="color:white;"><center>任务一</center></b>''',unsafe_allow_html=True)
# task1.task1()
# st.markdown('''# <b style="color:white;"><center>任务二</center></b>''',unsafe_allow_html=True)
# task2.task2.task2()
# st.markdown('''# <b style="color:white;"><center>任务三</center></b>''',unsafe_allow_html=True)
# task3.task3.task3()
# st.markdown('''# <b style="color:white;"><center>任务四</center></b>''',unsafe_allow_html=True)
# task4.task4.task4()
# st.markdown('''# <b style="color:white;"><center>任务五</center></b>''',unsafe_allow_html=True)
# task5.task5.task5()

st.markdown('''# <b style="color:white;"><center>设计流程</center></b>''', unsafe_allow_html=True)
st.image('设计流程.png')

st.markdown('''# <b style="color:white;"><center>软件设计思想</center></b>''', unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">在设计过程中，我们更看重的是模型的可拓展性和可维护性。我们通过自顶向下的设计，由于任务中的数据与时间序列有较强的相关性，因此我们选定了具有长期记忆能力的LSTM模型。同样，对用户进行分类时，我们需要将n个数据对象划分k类，保证同一类对象相似且不同类对象相似度小，因此我们采用了当前较为成熟的k-means算法。</b>''',
             unsafe_allow_html=True)


img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'

st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)

st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div >
section.css-1lcbmhc.e1fqkh3o3 > div.css-1adrfps.e1fqkh3o2
{background:rgba(255,255,255,0.5)}</style>''', unsafe_allow_html=True)


