# 导入模块
# -*- coding: UTF-8 -*-
import itertools
import warnings

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime
# matplotlib inline
from matplotlib.pylab import rcParams
import statsmodels.api as sm
import pmdarima as pm
import os
import xlsxwriter

# 图片大小设置
rcParams['figure.figsize'] = 6, 3
file_folder = 'D:/treasture/MBR_abnormal detection/MBR相关指标数据明细/data_txt'
outfile_folder_1 = 'D:/treasture/MBR_abnormal detection/MBR相关指标数据明细/outfile_统计指标'
outfile_folder_2 = 'D:/treasture/MBR_abnormal detection/MBR相关指标数据明细/outfile_异动监测'
outfile_folder_2_res = 'D:/treasture/MBR_abnormal detection/MBR相关指标数据明细/outfile_异动监测/残差_excel'
outfile_folder_2_yd = 'D:/treasture/MBR_abnormal detection/MBR相关指标数据明细/outfile_异动监测/异动监测_excel'
outfile = 'D:/treasture/MBR_abnormal detection/MBR相关指标数据明细/outfile'
file_input = file_folder + '/'


# 获取第一个字段名
def df_colname(file):
    file_input = file_folder + '/' + file
    data_dfname = pd.read_csv(file_input, sep='\t')
    dfname_list = list(data_dfname)
    pred_datetime = data_dfname[dfname_list[0]]
    pred_datetime_1 = pred_datetime[0]  # 指标的第一个时间
    pred_datetime_1 = str(pred_datetime_1) + '01'
    # print(dfname_list[0])  # 从0开始记起
    # print(dfname_list[1])

    # 处理时间序列
    dateparse = lambda dates: datetime.datetime.strptime(dates, '%Y%m')  # '%Y%m
    data = pd.read_csv(file_input, sep='\t', parse_dates=[dfname_list[0]], index_col=dfname_list[0],
                       date_parser=dateparse)
    print(data.head())
    print(data)
    print(data.index)
    ts = data[dfname_list[1]]
    print(ts)
    return ts, pred_datetime_1


#
# ts, pred_datetime_1 = df_colname(file_name)


# 分解
def decompose(ts):
    from statsmodels.tsa.seasonal import seasonal_decompose
    # 使用seasonal_decompose查看时间序列概况
    decomposition = seasonal_decompose(ts, extrapolate_trend=1, model='additive')
    # 季节 & 趋势性 & 残差
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    print(residual)
    print(trend)
    print(seasonal)
    # 残差导出
    val = ts.values - trend.values - seasonal.values
    index = ts.index  # 时间序列(时间)
    # #
    # 季节&趋势性&残差分类图示
    # plt.cla()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.tick_params(labelsize=7)
    plt.subplot(411)
    plt.plot(ts, label='原始数据')
    plt.legend(loc='best')
    plt.tick_params(labelsize=7)
    plt.subplot(412)
    plt.plot(trend, label='趋势')
    plt.legend(loc='best')
    plt.tick_params(labelsize=7)
    plt.subplot(413)
    plt.plot(seasonal, label='季节性')
    plt.legend(loc='best')
    plt.tick_params(labelsize=7)
    plt.subplot(414)
    plt.plot(residual, label='残差值')
    plt.legend(loc='best')
    plt.tick_params(labelsize=7)
    plt.tight_layout()
    # plt.show()
    # 图片保存
    fig_str = str(file_name) + '.png'  # 更改图片路径
    plt.savefig(outfile_folder_1 + '/' + fig_str, bbox_inches='tight')
    plt.close()
    # # 存入Excel
    excel_str = str(file_name) + '.xlsx'  # 更改图片路径
    excel_str3 = str(file_name) + '残差值' + '.xlsx'  # 更改图片路径
    residual1 = pd.DataFrame([index,residual])
    residual1.to_excel(outfile_folder_2_res + '/' + excel_str3,sheet_name='Sheet1',encoding='utf-8', index=False)
    # book = xlsxwriter.Workbook(outfile)
    # sheet_name = 'sheet' + str(file_name)
    # sheet = book.add_worksheet(sheet_name)
    # sheet.insert_image('B2', outfile_folder_1 + '/' + fig_str)
    return residual, val, index


# residual, val, index = decompose(ts)
def residual_SArima(residual, pred_datetime_1):
    # 对残差进行SArima建模
    # SARIMAX模型预测参数选取:pdq
    p = d = q = range(0, 2)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")  # specify to ignore warning messages
    # AIC值输出，选取最小值时的参数选取结果
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    from statsmodels.tsa.arima_model import ARIMA
    #
    mod = sm.tsa.statespace.SARIMAX(residual,
                                    order=(1, 0, 0),
                                    seasonal_order=(0, 0, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print('SARIMAX模型建模结果输出：')
    print(results.summary().tables[1])
    print('模型诊断结果如下：')
    # results.plot_diagnostics(figsize=(15, 12))
    # 模型预测
    pred = results.get_prediction(start=pd.to_datetime(pred_datetime_1), dynamic=False)
    # 残差上下界
    pred_ci = pred.conf_int()
    # mte_forecast = pred.predicted_mean
    # mte_orginal  = residual['2019-01-01':]
    # print(mte_forecast)
    # mse = ((mte_forecast - mte_orginal) ** 2).mean()
    # print(u'预测值的均方误差(MSE)是{}'.format(round(mse, 2)))
    # print('预测值的均方根误差(RMSE)是: {:.4f}'.format(np.sqrt(sum((mte_forecast - mte_orginal) ** 2) / len(mte_forecast))))
    # 处理起始日期的两个值
    pred_ci['lower resid'][0] = 0
    pred_ci['upper resid'][0] = 0
    print(pred_ci)
    lower_resid = pred_ci['lower resid'].values
    upper_resid = pred_ci['upper resid'].values
    # ts_time = pred_datetime_1[0:4]  # 开始年份
    # ax = ts[ts_time:].plot(label='observed')
    # pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
    return lower_resid, upper_resid


# lower_resid,upper_resid = residual_SArima(residual, pred_datetime_1)


# 异动监测
def Movement_monitoring(residual, val, index, lower_resid, upper_resid):
    Variation_upper_bound = val - lower_resid
    Variation_lower_bound = val - upper_resid

    plt.cla()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.tick_params(labelsize=8)
    plt.title('指标异动监测')
    plt.plot(index, Variation_upper_bound, color='blue', label='异动上界值')
    plt.legend(loc='best')
    plt.plot(index, Variation_lower_bound, color='orange', label='异动下界值')
    plt.legend(loc='best')
    plt.plot(index, residual, color='gray', label='原始残差')
    plt.legend(loc='best')
    plt.legend()  # 显示图例
    # plt.show()
    # 图片保存
    fig_str_1 = str(file_name) + '.png'  # 更改图片路径
    plt.savefig(outfile_folder_2 + '/' + fig_str_1, bbox_inches='tight')
    plt.close()  # 防止图片发生叠加
    # 存入Excel
    excel_str = str(file_name) + '.xlsx'  # 更改图片路径
    excel_str1 = str(file_name) + '下限值' + '.xlsx'  # 更改图片路径
    excel_str2 = str(file_name) + '上限值' + '.xlsx'  # 更改图片路径
    Variation_upper_bound = pd.DataFrame([index,Variation_upper_bound])
    Variation_lower_bound = pd.DataFrame([index,Variation_lower_bound])
    Variation_lower_bound.to_excel(outfile_folder_2_res + '/' + excel_str1, sheet_name='Sheet2',encoding='utf-8', index=False)
    Variation_upper_bound.to_excel(outfile_folder_2_res + '/' + excel_str2, sheet_name='Sheet3', encoding='utf-8', index=False)
    # # 存入Excel
    # book = xlsxwriter.Workbook(outfile)
    # sheet_name = 'sheet' + str(file_name)
    # sheet = book.add_worksheet(sheet_name)
    # sheet.insert_image('B22', outfile_folder_2 + '/' + fig_str_1)
    # book.close()
    return Variation_upper_bound, Variation_lower_bound


# Movement_monitoring(residual, val, index,lower_resid,upper_resid)


# 遍历文件
for file in os.listdir(file_folder):
    print(file)
    file_name = file.split('.')[0]  # 文件名
    ts, pred_datetime_1 = df_colname(file)
    residual, val, index = decompose(ts)
    lower_resid, upper_resid = residual_SArima(residual, pred_datetime_1)
    Movement_monitoring(residual, val, index, lower_resid, upper_resid)

# 插入Excel中















