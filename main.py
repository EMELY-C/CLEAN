import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from CLEAN import spectrum_featurization as spefea, maching_learning as machlea

def multi_process(msms_input, database, train_test, fnr_knn):
    dfs = match_database(msms_input, database)
    df_maybe_L3 = clean_validation(dfs[0])
    df_L3 = fg_predict(train_test=train_test, fnr_knn=fnr_knn, df_maybe_L3=df_maybe_L3)
    result = pd.concat([dfs[1], df_L3], axis=0)

    # 清理结果中的特殊字符
    if 'Molecular Formula' in result.columns:
        result['Molecular Formula'] = result['Molecular Formula'].apply(clean_special_characters)

    # 按 Level 列分类并保留指定列
    level1_data = result[result['Level'] == 'Level_1']
    level2_data = result[result['Level'] == 'Level_2']
    level3_data = result[result['Level'] == 'Level_3']
    level4_data = result[result['Level'] == 'Level_4']

    # 对 Level_1 到 Level_3 保留指定列
    required_columns_non4 = ['ID', 'mz', 'rt', 'MS2mz', 'MS2int', 'Lab ID', 'Name', 'CAS', 'HMDB', 'RT (s)',
                             'pre RT (s)', 'Exposure source', 'Component group', 'Molecular Formula',
                             'Monoisotopic mass', 'Labeled Formula', 'Labeled m/z', 'Level']
    available_columns_non4 = level1_data.columns.tolist()
    filtered_columns_non4 = [col for col in required_columns_non4 if col in available_columns_non4]
    level1_data = level1_data[filtered_columns_non4]
    level2_data = level2_data[filtered_columns_non4]
    level3_data = level3_data[filtered_columns_non4]

    # 对 Level_4 保留指定列
    required_columns_4 = ['ID', 'mz', 'rt', 'MS2mz', 'MS2int', 'Level']
    available_columns_4 = level4_data.columns.tolist()
    filtered_columns_4 = [col for col in required_columns_4 if col in available_columns_4]
    level4_data = level4_data[filtered_columns_4]

    # 对 Level_4 去重
    level4_data = level4_data.drop_duplicates(subset=['mz', 'rt'])

    # 合并所有 Level 的数据
    result = pd.concat([level1_data, level2_data, level3_data, level4_data])

    # 按照 Level 列的优先级排序
    level_order = {'Level_1': 1, 'Level_2': 2, 'Level_3': 3, 'Level_4': 4}
    result['Level_Order'] = result['Level'].map(level_order)
    result = result.sort_values(by='Level_Order')
    result = result.drop(columns=['Level_Order'])

    result.to_csv('result.csv', index=False, encoding='utf-8-sig')

def main_ml_comparison(train_test):
    mols = list((train_test["SMILES"]))
    fp_train_test = np.zeros(shape=(len(train_test), 8034))

    for i in range(len(train_test)):
        fp_train_test[i] = machlea.get_cdk_fingerprints(mols[i])
    fp_0 = fp_train_test.copy()

    fp_sum = []
    for i in range(8034):
        fp_num = 0
        for j in range(len(train_test)):
            fp_num = fp_num + fp_train_test[j][i]
        fp_sum.append(fp_num)
    index = []
    for i in range(8034):
        if (int(len(train_test) * 0.1) < fp_sum[i] < int(len(train_test) * 0.9)):
            index.append(i)
    fp_train_test = fp_train_test[:, index]

    index2 = [0]
    for i in range(len(index)):
        flag = 1
        for j in range(len(index2)):
            if (list(fp_train_test[:, i]) == list(fp_train_test[:, index2[j]])):
                flag = 0
                break
        if (flag == 1):
            index2.append(i)

    index3 = []
    for i in index2:
        index3.append(index[i])
    fp_train_test = fp_0[:, index3]

    # 在训练测试数据中创建片段核矩阵
    fr_train_test = np.zeros(shape=(len(train_test), len(train_test)))
    fnr_train_test = np.zeros(shape=(len(train_test), len(train_test)))
    frag_train_test = np.zeros(shape=(len(train_test), 120000))

    for i in range(len(train_test)):
        for j in range(len(train_test)):
            fr_train_test[i][j] = spefea.FR(train_test["MSMS spectrum"][i], train_test["MSMS spectrum"][j],
                                            train_test["m/z"][i], train_test["m/z"][j])
            fnr_train_test[i][j] = spefea.FNR(train_test["MSMS spectrum"][i], train_test["MSMS spectrum"][j],
                                              train_test["m/z"][i], train_test["m/z"][j])
    for i in range(len(train_test)):
        spe_i = spefea.MSMSpre1(train_test["MSMS spectrum"][i], train_test["m/z"][i], minmz=50, maxmz=2000)
        if (len(spe_i) == 0):
            continue
        for j in range(len(spe_i)):
            mz_j = int(spe_i[j, 0] * 100 - 5000 + 0.5)
            frag_train_test[i][mz_j] = spe_i[j, 1]

    df_fr_knn_linear = machlea.mach_lea(fr_train_test, fp_train_test, machlea.knn5)
    df_fnr_knn_linear = machlea.mach_lea(fnr_train_test, fp_train_test, machlea.knn5)
    df_frag_knn_linear = machlea.mach_lea(frag_train_test, fp_train_test, machlea.knn5)
    df_shuffle_fnr_knn_linear = machlea.shuffle(fnr_train_test, fp_train_test, machlea.knn5)
    df_fnr_knn_linear.to_csv("fnr_knn.csv", index=False)

    
    # 可视化结果
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))
    plt.violinplot
    axes[0, 0].violinplot([df_fr_knn_linear["acc"], df_fnr_knn_linear["acc"], df_frag_knn_linear["acc"],
                           df_shuffle_fnr_knn_linear["acc"]], showmeans=False, showmedians=True)
    axes[0, 1].violinplot([df_fr_knn_linear["pre"], df_fnr_knn_linear["pre"], df_frag_knn_linear["pre"],
                           df_shuffle_fnr_knn_linear["pre"]], showmeans=False, showmedians=True)
    axes[1, 0].violinplot([df_fr_knn_linear["rec"], df_fnr_knn_linear["rec"], df_frag_knn_linear["rec"],
                           df_shuffle_fnr_knn_linear["rec"]], showmeans=False, showmedians=True)
    axes[1, 1].violinplot(
        [df_fr_knn_linear["f1"], df_fnr_knn_linear["f1"], df_frag_knn_linear["f1"], df_shuffle_fnr_knn_linear["f1"]],
        showmeans=False, showmedians=True)
    font = {"family": "Arial", "weight": "normal", "size": 30}
    axes[0, 0].set_ylabel('Accuracy', font)
    axes[0, 1].set_ylabel('Precision', font)
    axes[1, 0].set_ylabel('Recall', font)
    axes[1, 1].set_ylabel('F1 Score', font)
    # plt.setp(axes, xticklabels=["","FR","",'FNR',"","Frag","","Random"])
    xticklabels = ["", "FR", "", 'FNR', "", "Frag", "", "Shu"]
    yticklabel1 = ["", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    yticklabel2 = ["", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    axes[0, 0].set_xticklabels(xticklabels, fontproperties="Arial", size=30)
    axes[0, 1].set_xticklabels(xticklabels, fontproperties="Arial", size=30)
    axes[1, 0].set_xticklabels(xticklabels, fontproperties="Arial", size=30)
    axes[1, 1].set_xticklabels(xticklabels, fontproperties="Arial", size=30)
    axes[0, 0].set_yticklabels(yticklabel1, fontproperties="Arial", size=30)
    axes[0, 1].set_yticklabels(yticklabel2, fontproperties="Arial", size=30)
    axes[1, 0].set_yticklabels(yticklabel2, fontproperties="Arial", size=30)
    axes[1, 1].set_yticklabels(yticklabel2, fontproperties="Arial", size=30)
    plt.show()
    

    df_svm=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.svm_linear)
    df_logi=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.logi)
    df_bay=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.bay)
    df_dec=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.dec)
    df_ran=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.ran)
    df_knn=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.knn5)
    df_ann=machlea.mach_lea(fnr_train_test,fp_train_test,machlea.ann)
    df_shuffle_fnr_knn_linear=machlea.shuffle(fnr_train_test,fp_train_test,machlea.knn5)

    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))
    plt.violinplot
    axes[0, 0].violinplot([df_svm["acc"], df_logi["acc"], df_bay["acc"], df_dec["acc"], df_ran["acc"], df_knn["acc"], df_ann["acc"], df_shuffle_fnr_knn_linear["acc"]], showmeans=False, showmedians=True)
    axes[0, 1].violinplot([df_svm["pre"], df_logi["pre"],df_bay["pre"],df_dec["pre"],df_ran["pre"],df_knn["pre"],df_ann["pre"],df_shuffle_fnr_knn_linear["pre"]], showmeans=False, showmedians=True)
    axes[1, 0].violinplot([df_svm["rec"], df_logi["rec"], df_bay["rec"], df_dec["rec"], df_ran["rec"], df_knn["rec"], df_ann["rec"], df_shuffle_fnr_knn_linear["rec"]], showmeans=False, showmedians=True)
    axes[1, 1].violinplot([df_svm["f1"], df_logi["f1"], df_bay["f1"], df_dec["f1"], df_ran["f1"], df_knn["f1"], df_ann["f1"], df_shuffle_fnr_knn_linear["f1"]], showmeans=False, showmedians=True)
    font={"family":"Arial","weight":"normal","size":30}
    axes[0, 0].set_ylabel('Accuracy',font)
    axes[0, 1].set_ylabel('Precision',font)
    axes[1, 0].set_ylabel('Recall',font)
    axes[1, 1].set_ylabel('F1 Score',font)
    xticklabels=["",'SVM',"LOG","BAY","DEC","RAN","KNN","ANN","Shu"]
    yticklabel1=["","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
    yticklabel2=["","0.0","0.2","0.4","0.6","0.8","1.0"]
    axes[0, 0].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
    axes[0, 1].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
    axes[1, 0].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
    axes[1, 1].set_xticklabels(xticklabels,fontproperties="Arial",size=30)
    axes[0, 0].set_yticklabels(yticklabel1,fontproperties="Arial",size=30)
    axes[0, 1].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
    axes[1, 0].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
    axes[1, 1].set_yticklabels(yticklabel2,fontproperties="Arial",size=30)
    plt.show()
    

    validation = train_test
    list_smiles = validation["SMILES"].tolist()
    list_val_name = validation["Name"].tolist()
    list_name = validation["Name"].tolist()
    num_validation = len(validation)
    num_smiles = len(list_smiles)
    list_score = np.zeros((num_validation, num_smiles))

    for i in range(num_validation):
        fnr_ = np.zeros(shape=(1, len(train_test)))

        for j in range(len(train_test)):
            fnr_[0][j] = spefea.FNR(train_test["MSMS spectrum"][j], validation["MSMS spectrum"][i],
                                    train_test["m/z"][j], validation["m/z"][i])

        fp_pre = np.zeros(shape=(1, fp_train_test.shape[1]))

        for k in tqdm(range(fp_train_test.shape[1]), desc="Processing Features"):
            fp_pre[0, k] = machlea.svm_(fnr_train_test, fp_train_test[:, k], fnr_)

        for m in range(num_smiles):
            fp_cac = np.array([machlea.get_cdk_fingerprints(list_smiles[m])])
            fp_cac = fp_cac[:, index3]
            score_ = machlea.score(fp_pre, fp_cac, df_fnr_knn_linear)
            list_score[i, m] = score_

    df_fp = pd.DataFrame(list_score, columns=list_val_name)
    # df_fp.insert(0, 'Name', list_name)
    df_fp.index = [i for i in range(len(list_name))]
    sum_f1 = df_fnr_knn_linear['f1'].sum()
    df_fp = df_fp / sum_f1
    df_fp.to_excel("hotplot_data.xlsx", index=False)

    df_score = df_fp
    df_score = df_score.astype(float)

    
    plt.figure(figsize=(38, 35))
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    heatmap = sns.heatmap(df_score, annot=False, cmap=cmap, cbar=True, fmt=".2f",
                          annot_kws={"size": 8}, linewidths=.5, linecolor='gray', vmin=0, vmax=1)
    plt.title('Tanimoto Coefficient Heatmap', fontsize=16, pad=20)
    plt.xlabel('Compounds', fontsize=15)
    plt.ylabel('Compounds', fontsize=15)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=90)
    labels = df_score.columns.tolist()
    plt.yticks(ticks=range(len(labels)), labels=labels, rotation=0)
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.tick_params(axis='both', labelsize=10)

    plt.show()

def match_database(msms_input, database):
    # 去掉列名前后空格，顺便把中间多个空格也压成一个
    database.columns = database.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    PrecursorMZ = msms_input['mz'].values
    Labeled_mz = database['Labeled m/z'].values

    matched_rows = []
    for i, precursor in enumerate(PrecursorMZ):
        for j, labeled in enumerate(Labeled_mz):
            if abs(precursor - labeled) / labeled * 1e6 < 5:
                matched_rows.append((i, j))

    matched_msms_input = msms_input.iloc[[i for i, _ in matched_rows]].reset_index(drop=True)
    matched_database = database.iloc[[j for _, j in matched_rows]].reset_index(drop=True)

    merged_df = pd.concat([matched_msms_input, matched_database], axis=1)

    matched_indices = {i for i, _ in matched_rows}
    unmatched_msms_input = msms_input[~msms_input.index.isin(matched_indices)].reset_index(drop=True)
    level4_df = unmatched_msms_input
    level4_df['Level'] = 'Level_4'

    rt_df = merged_df[merged_df['RT (s)'].notna() & (merged_df['RT (s)'].astype(str).str.strip() != '')]
    L1_index_list = []
    L4_index_list = []
    for i, rt_acc, rt in zip(range(len(rt_df)), rt_df['rt'], rt_df['RT (s)']):
        if abs(rt_acc - rt) / 60 <= 0.2:
            L1_index_list.append(i)
        else:
            L4_index_list.append(i)

    # 确保索引在有效范围内
    max_valid_index_rt = len(rt_df) - 1
    valid_L1_indices = [idx for idx in L1_index_list if idx <= max_valid_index_rt]
    valid_L4_indices = [idx for idx in L4_index_list if idx <= max_valid_index_rt]

    L1_df = rt_df.iloc[valid_L1_indices].reset_index(drop=True)
    L1_df['Level'] = "Level_1"
    L4_df_1 = rt_df.iloc[valid_L4_indices].reset_index(drop=True)
    L4_df_1['Level'] = "Level_4"

    pre_rt1 = merged_df[merged_df['pre RT (s)'].notna() & (merged_df['pre RT (s)'].astype(str).str.strip() != '') &
                        merged_df['Public Database'].notna() & (merged_df['Public Database'].astype(str).str.strip() != '')]
    L2_index_list = []
    L4_index_list_pre = []
    for i, rt_acc, rt in zip(range(len(pre_rt1)), pre_rt1['rt'], pre_rt1['pre RT (s)']):
        if abs(rt_acc - rt) / 60 <= 2:
            L2_index_list.append(i)
        else:
            L4_index_list_pre.append(i)

    # 确保索引在有效范围内
    max_valid_index_pre = len(pre_rt1) - 1
    valid_L2_indices = [idx for idx in L2_index_list if idx <= max_valid_index_pre]
    valid_L4_indices_pre = [idx for idx in L4_index_list_pre if idx <= max_valid_index_pre]

    L2_df = pre_rt1.iloc[valid_L2_indices].reset_index(drop=True)
    L2_df['Level'] = "Level_2"
    L4_df_2 = pre_rt1.iloc[valid_L4_indices_pre].reset_index(drop=True)
    L4_df_2['Level'] = "Level_4"

    fg_df = merged_df[merged_df['pre RT (s)'].notna() & (merged_df['pre RT (s)'].astype(str).str.strip() != '') &
                      merged_df['Public Database'].isna()].reset_index(drop=True)
    L3_index_list = []
    L4_index_list_fg = []
    for i, rt_acc, rt in zip(range(len(fg_df)), fg_df['rt'], fg_df['pre RT (s)']):
        if abs(rt_acc - rt) / 60 <= 5:
            L3_index_list.append(i)
        else:
            L4_index_list_fg.append(i)

    # 确保索引在有效范围内
    max_valid_index_fg = len(fg_df) - 1
    valid_L3_indices = [idx for idx in L3_index_list if idx <= max_valid_index_fg]
    valid_L4_indices_fg = [idx for idx in L4_index_list_fg if idx <= max_valid_index_fg]

    maybe_L3_df = fg_df.iloc[valid_L3_indices].reset_index(drop=True)
    L4_df_fg = fg_df.iloc[valid_L4_indices_fg].reset_index(drop=True)
    L4_df_fg['Level'] = "Level_4"

    frames = [L1_df, L2_df, level4_df, L4_df_1, L4_df_2, L4_df_fg]
    result = pd.concat(frames)

    return [maybe_L3_df, result]

def clean_validation(df_maybe_l3):
    df = df_maybe_l3

    merged_list = []
    for index, row in df.iterrows():
        mz_list = row['MS2mz'].split(',')
        int_list = row['MS2int'].split(',')
        merged = ' '.join([f"{mz}:{int_value}" for mz, int_value in zip(mz_list, int_list)])
        merged_list.append(merged)

    df['MSMS spectrum'] = merged_list

    return df

def fg_predict(train_test, fnr_knn, df_maybe_L3):
    mols = list((train_test["SMILES"]))
    fp_train_test = np.zeros(shape=(len(train_test), 8034))

    for i in range(len(train_test)):
        fp_train_test[i] = machlea.get_cdk_fingerprints(mols[i])
    fp_0 = fp_train_test.copy()

    fp_sum = []
    for i in range(8034):
        fp_num = 0
        for j in range(len(train_test)):
            fp_num = fp_num + fp_train_test[j][i]
        fp_sum.append(fp_num)
    index = []
    for i in range(8034):
        if (int(len(train_test) * 0.1) < fp_sum[i] < int(len(train_test) * 0.9)):
            index.append(i)
    fp_train_test = fp_train_test[:, index]

    index2 = [0]
    for i in range(len(index)):
        flag = 1
        for j in range(len(index2)):
            if (list(fp_train_test[:, i]) == list(fp_train_test[:, index2[j]])):
                flag = 0
                break
        if (flag == 1):
            index2.append(i)

    index3 = []
    for i in index2:
        index3.append(index[i])
    fp_train_test = fp_0[:, index3]

    fnr_train_test = np.zeros(shape=(len(train_test), len(train_test)))
    for i in range(len(train_test)):
        for j in range(len(train_test)):
            fnr_train_test[i][j] = spefea.FNR(train_test["MSMS spectrum"][i], train_test["MSMS spectrum"][j],train_test["m/z"][i], train_test["m/z"][j])

    validation = df_maybe_L3

    num_validation = len(validation)
    list_score = []  # 存储分数
    for i in range(num_validation):
        fnr_ = np.zeros(shape=(1, len(train_test)))

        # 计算 FNR
        for j in range(len(train_test)):
            fnr_[0][j] = spefea.FNR(train_test["MSMS spectrum"][j], validation["MSMS spectrum"][i], train_test["m/z"][j],
                             validation["mz"][i])

        fp_pre = np.zeros(shape=(1, fp_train_test.shape[1]))

        for k in range(fp_train_test.shape[1]):
            fp_pre[0, k] = knn_(fnr_train_test, fp_train_test[:, k], fnr_)

        df_fnr_svm_linear = fnr_knn

        fp_cac = np.array([machlea.get_cdk_fingerprints(validation['SMILES'][i])])
        fp_cac = fp_cac[:, index3]
        score_ = machlea.score(fp_pre, fp_cac, df_fnr_svm_linear)
        list_score.append(score_)

    total_score = fnr_knn['f1'].sum()
    list_array = np.array(list_score)
    list_array = list_array / total_score
    df_maybe_L3['score'] = list_array

    df_final_L3 = df_maybe_L3[df_maybe_L3['score'] >= 0.8].reset_index(drop=True)
    df_final_L3['Level'] = 'Level_3'
    df_final_L3 = df_final_L3.rename(columns={'m/z': 'mz'})

    return df_final_L3

def knn_(X,y,hss):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    return classifier.predict(hss)

def clean_special_characters(text):
    if isinstance(text, str):
        # 替换常见的特殊空格字符
        text = text.replace('\u00A0', ' ')
        text = text.replace('聽', '')
        text = text.strip()
    return text


# 主程序
msms_input = pd.read_csv('test_MS2_input.csv')
database = pd.read_excel('database.xlsx', sheet_name='DNSCL')
train_test = pd.read_excel('DNS&MPEA_train-test.xlsx')

# 先运行机器学习比较，生成 fnr_knn
main_ml_comparison(train_test=train_test)

# 读取生成的 fnr_knn
fnr_knn = pd.read_csv('fnr_knn.csv')

# 然后运行多进程处理
multi_process(msms_input=msms_input, database=database, train_test=train_test, fnr_knn=fnr_knn)