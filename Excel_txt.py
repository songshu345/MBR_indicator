import pandas as pd
import openpyxl
import time
import os

# 输入路径
inpath = 'D:/treasture/MBR_abnormal detection/MBR相关指标数据明细/7月MBR可提取指标数据明细'
# inpath =r'D:\treasture\label_BA\label_report\标签报告_for_bac\5BG分季度标签\dj_label'
# 输出路径
outpath = 'D:/treasture/MBR_abnormal detection/MBR相关指标数据明细/data_txt'
# outpath = r'D:\treasture\label_BA\label_report\标签报告_for_bac\5BG分季度标签\dj_label_txt\dj_label_1.xlsx'
print('START:' + str(time.ctime()))

# 读取excel文件
for afile in os.listdir(inpath):
    if afile[-4:].lower() == 'xlsx':
        print(afile)
        name = inpath + '/' + afile
        # 读取每一个sheet
        wb = openpyxl.load_workbook(name)
        sheets = wb.sheetnames
        for sheet in sheets:
            print(sheet)
            df = pd.read_excel(name, sheet_name=sheet, header=None)
            print('开始写入txt文件...')
            # 保存txt文件
            df.to_csv(outpath + '/' + afile[:-5] + '.txt', header=None, sep='\t', index=False)
            print('文件写入成功!')

print('END:' + str(time.ctime()))