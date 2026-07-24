# @time: 2026.5.10
# @Author: Junjun Chen, Zihao Wang
# @memo: code for uq

import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from scipy import interpolate


def site_concentration(column):
    site_concentration=pd.read_excel('.\\concentration.xlsx')
    column_values = site_concentration.columns.values
    site_value=np.ones((site_concentration.shape[0],2))
    column_0=np.where(column_values==column[0])[0][0]
    column_1=np.where(column_values==column[1])[0][0]
    site_value[:, 0] = site_concentration[column_values[column_0]]
    site_value[:, 1] = site_concentration[column_values[column_1]]
    value_nan_index = np.where(np.isnan(site_value))
    value_nan = site_value[value_nan_index[0], value_nan_index[1]].reshape((-1, 2))
    site_value = site_value[0:site_value.shape[0]-value_nan.shape[0]]
    return site_value


def aqui_con_read(t,elem_num,skiprow=10):
    for i in range (t+1):
        aqui_con=np.loadtxt('.\\aqui_con.dat', skiprows=skiprow, max_rows=elem_num, dtype=float)
        aqui_con=np.expand_dims(aqui_con,axis=1)
        if i==0:
            aqui_con_result=aqui_con
        else:
            aqui_con_result=np.concatenate((aqui_con_result,aqui_con),axis=1)
        skiprow+=elem_num+1
    return aqui_con_result


def get_aquicon_index(name):
    result = open('.\\aqui_con.dat', 'r')
    result_lines = result.readlines()
    line = result_lines[8]
    line = line.split(',')
    line_index=line[0:len(line)-1]
    for i in range (len(line_index)):
        a=line_index[i]
        if name in a:
            name_index=i
    return name_index


def simulation_data(name_aqui,name_site,aqui_con_t=5,aqui_con_enum=17):
    aqui_con = aqui_con_read(t=aqui_con_t, elem_num=aqui_con_enum)
    index = get_aquicon_index(name_aqui)
    Dis_index = get_aquicon_index('VARIABLES =X')
    model_data = aqui_con[:, -1, [Dis_index, index]]
    site_data = site_concentration(name_site)
    site_D=site_data[:,0]
    # 插值
    f=interpolate.interp1d(model_data[:,0],model_data[:,1],kind='quadratic')
    site_model=f(site_D)
    return site_model


def chemical_keywords_index(chemical_lines,Keywords):
    keywords_index = np.zeros(len(Keywords), dtype=int)
    for i in range(len(Keywords)):
        key=Keywords[i]
        for line in chemical_lines:
            index_key=chemical_lines.index(line)
            line1=chemical_lines[index_key]
            if key in line1:
                keywords_line=line1
                index=chemical_lines.index(keywords_line)
                keywords_index[i]=index
    return keywords_index


def flow_open():
    flow = open('.\\Multiphase Flow.inp', 'r')
    flow_lines = flow.readlines()
    flow.close()
    return flow_lines


def chemical_open():
    chemical=open('.\\Geochemical.inp','r')
    chemical_lines=chemical.readlines()
    chemical.close()
    return chemical_lines


def flow_write(flow_lines):
    newfile=open('.\\Multiphase Flow.inp','w')
    for newline in flow_lines:
        newfile.write(newline)
    newfile.close()
    return


def chemical_write(chemical_lines):
    newfile=open('.\\Geochemical.inp','w')
    for newline in chemical_lines:
        newfile.write(newline)
    newfile.close()
    return


def per_para(flow_lines,PER1,PER2,PER3):
    ROCKS_index=flow_lines.index("ROCKS----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8\n")
    ROCKS1=flow_lines[ROCKS_index+1]
    A=ROCKS1[0:30]
    B=ROCKS1[60:82]

    flow_lines[ROCKS_index+1]=A+"%-10.4e"%(PER1)+"%-10.4e"%(PER2)+"%-10.4e"%(PER3)+B

    return flow_lines


def GENER_para(flow_lines,GENER_value):
    GENER_num = GENER_value.shape[0]
    GENER_index0 = flow_lines.index(
        'GENER----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8\n')
    for i in range(GENER_num):
        GENER_index = GENER_index0 + i + 1
        GENER_line = flow_lines[GENER_index]
        GENER_line1 = GENER_line[0:40]
        GENER_line2 = GENER_value[i]
        GENER_line = GENER_line1 + "{:10.3E}".format(GENER_line2) + '\n'
        flow_lines[GENER_index] = GENER_line
    return flow_lines


def Exchange_cation_coeff(chemical_lines,parameter,add_index=3):
    Keywords=['EXCHANGEABLE CATIONS','INITIAL AND BOUDARY WATER TYPES']
    index=chemical_keywords_index(chemical_lines,Keywords)
    for i in range(len(parameter)):
        index_para=index[0]+add_index+i
        para_line=chemical_lines[index_para]
        para_line=para_line.split()
        para_line[3]=parameter[i]
        chemical_lines[index_para]="%-20s"%(para_line[0])+"%-13s"%(para_line[1])+"%-13s"%(para_line[2])+\
                                   "%-10.4e"%(para_line[3])+'\n'
    return chemical_lines


def Initial_boudary(chemical_lines,parameter,add_ini_index=5,reduce_bou_index=2):
    Keywords=['INITIAL AND BOUDARY WATER TYPES','INITIAL MINERAL ZONES']
    index=chemical_keywords_index(chemical_lines,Keywords)
    for i in range (max(parameter.shape)):
        # 初始水
        index_inipara=index[0]+add_ini_index+i
        inipara_line=chemical_lines[index_inipara]
        inipara_line=inipara_line.split()
        inipara_line[2]=parameter[0,i]
        inipara_line[3]=parameter[0,i]
        chemical_lines[index_inipara]="%-11s"%(inipara_line[0])+"%-9s"%(inipara_line[1])+"%-15.3e"%(inipara_line[2])+\
                                      "%-12.3e"%(inipara_line[3])+"%-4s"%(inipara_line[4])+"%-5s"%(inipara_line[5])+\
                                      "%-8s"%(inipara_line[6])+'\n'
        # 边界水
        index_boupara=index[1]-reduce_bou_index-parameter.shape[1]+i
        boupara_line=chemical_lines[index_boupara]
        boupara_line=boupara_line.split()
        boupara_line[2]=parameter[1,i]
        boupara_line[3]=parameter[1,i]
        chemical_lines[index_boupara]="%-11s"%(boupara_line[0])+"%-9s"%(boupara_line[1])+"%-15.3e"%(boupara_line[2])+\
                                      "%-12.3e"%(boupara_line[3])+"%-4s"%(boupara_line[4])+"%-5s"%(boupara_line[5])+\
                                      "%-8s"%(boupara_line[6])+'\n'
    return chemical_lines


def para_CEC(parameter,chemical_lines,add_index=3):
    Keywords=['INITIAL ZONES OF CATION EXCHANGE','end']
    index = chemical_keywords_index(chemical_lines, Keywords)
    index_CEC=index[0]+add_index
    CEC_line=chemical_lines[index_CEC]
    CEC_line=CEC_line.split()
    CEC_line[1]=float(np.asarray(parameter, dtype=float).reshape(-1)[0])
    chemical_lines[index_CEC]="%-20s"%(CEC_line[0])+"%-6.4f"%(CEC_line[1])+'\n'
    return chemical_lines


def parameter_sample(sample_num,num_para,para_low,para_up):
    para_norm=np.ones((sample_num,num_para))
    for i in range (num_para):
        para_norm[:,i]=np.random.uniform(0,1,sample_num)
    para_low_tile=np.tile(para_low,(sample_num,1))
    para_up_tile=np.tile(para_up,(sample_num,1))
    para=para_norm*(para_up_tile-para_low_tile)+para_low_tile
    return para_norm,para


def para_inverse_fix(fix_para,fix_para_item,inverse_para):
    total_dim=len(fix_para)+len(inverse_para)
    total_item=np.arange(total_dim)
    inverse_item=np.setdiff1d(total_item,fix_para_item)
    para=np.ones(total_dim)
    for i in range(total_dim):
        if i in fix_para_item:
            fix_idx = np.where(fix_para_item == i)[0][0]
            para[i] = float(fix_para[fix_idx])
        else:
            inverse_idx = np.where(inverse_item == i)[0][0]
            para[i] = float(inverse_para[inverse_idx])
    return para



# ============================================================
# main
XQ_FILE = os.path.join("..", "results", "Xq_for_UQ.npz")
xq_data = np.load(XQ_FILE)

X_q_norm = np.asarray(xq_data["X_q_norm"], dtype=float)
X_q_real = np.asarray(xq_data["X_q_real"], dtype=float)

sample_num = X_q_real.shape[0]

# 反演项目
para_item = ['flow', 'cation_ecof', 'CEC', 'inicon', 'bouncon']
para_num = [1, 4, 1, 6, 6]

real_params = {
    'flow': np.full((sample_num, 1), 3e-12, dtype=float),

    'cation_ecof': np.column_stack([
        np.full(sample_num, 0.2, dtype=float),
        X_q_real[:, 0],
        X_q_real[:, 1],
        X_q_real[:, 2],
    ]),

    'CEC': X_q_real[:, 3:4],

    'inicon': np.tile(
        np.array([
            10 ** -6.7,
            6.1e-3,
            1.2e-2,
            8.85e-2,
            2.16e-3,
            2.16e-2,
        ], dtype=float),
        (sample_num, 1),
    ),

    'bouncon': np.column_stack([
        np.full(sample_num, 10 ** -6.9, dtype=float),
        X_q_real[:, 4],
        np.full(sample_num, 3.19e-6, dtype=float),
        np.full(sample_num, 9.64e-5, dtype=float),
        np.full(sample_num, 8e-5, dtype=float),
        X_q_real[:, 5],
    ]),
}

print(f"已读取 X_q 样本数：{sample_num}")


# 打开flow_lines
flow_lines = flow_open()
chemical_lines = chemical_open()

# 阳离子交换系数
fix_cation_ecof = np.array([1.0])
fix_cation_ecof_item = np.array([0])
# 初始水浓度
fix_inicon = np.array([1.0e+0, 2.7e-4, 1.018e-1, 1.0e-5, 5.0e-2])
fix_inicon_item = np.array([0, 7, 8, 9, 10])
# 边界水浓度
fix_bouncon = np.array([1.0, 1.0e-6, 1.0e-4, 1.0e-50, 5.0e-50])
fix_bouncon_item = np.array([0, 7, 8, 9, 10])

# 准备列表来收集结果（比在循环里不断 concatenate 更高效且不会报错）
results_dict = {
    'PH': [], 'Mg': [], 'Ca': [], 'Na': [], 'K': [], 'HCO3': []
}

for i in tqdm(range(sample_num)):
    per = real_params['flow'][i][0] * np.ones((1, 3))
    flow_lines = per_para(flow_lines, per[0, 0], per[0, 1], per[0, 2])

    ## 完成flow.inp文件改写
    flow_write(flow_lines)
    cation_ecof = para_inverse_fix(fix_cation_ecof, fix_cation_ecof_item, real_params['cation_ecof'][i])
    chemical_lines = Exchange_cation_coeff(chemical_lines, cation_ecof)

    # 初始水&边界水浓度
    inicon = para_inverse_fix(fix_inicon, fix_inicon_item, real_params['inicon'][i])
    bouncon = para_inverse_fix(fix_bouncon, fix_bouncon_item, real_params['bouncon'][i])
    inicon = np.expand_dims(inicon, 0)
    bouncon = np.expand_dims(bouncon, 0)
    ini_boun = np.concatenate((inicon, bouncon), 0)
    chemical_lines = Initial_boudary(chemical_lines, ini_boun)

    # CEC
    chemical_lines = para_CEC(real_params['CEC'][i], chemical_lines)
    # write chemical.inp
    chemical_write(chemical_lines)

    # 执行exe
    os.system('echo.|SOWCOM_V2_EOS9.exe')

    # 提取结果
    results_dict['PH'].append(np.expand_dims(simulation_data('pH', ['PH_D', 'PH']), axis=1))
    results_dict['Mg'].append(np.expand_dims(simulation_data('t_mg', ['Mg_D', 'Mg']) * 1000, axis=1))
    results_dict['Ca'].append(np.expand_dims(simulation_data('t_ca', ['Ca_D', 'Ca']) * 1000, axis=1))
    results_dict['Na'].append(np.expand_dims(simulation_data('t_na', ['Na_D', 'Na']) * 1000, axis=1))
    results_dict['K'].append(np.expand_dims(simulation_data('t_k', ['K_D', 'K']) * 1000, axis=1))
    results_dict['HCO3'].append(np.expand_dims(simulation_data('t_hco3', ['HCO3_D', 'HCO3']) * 1000, axis=1))

# 保存结果
UQ_OUT = os.path.join('.', 'results', 'uq')
for item_name, data_list in results_dict.items():
    concatenated_data = np.concatenate(data_list, axis=1)
    np.save(
        os.path.join(UQ_OUT, f'{item_name}_model.npy'),
        concatenated_data,
    )



