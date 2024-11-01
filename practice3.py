# train1, train2, test EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from catboost import CatBoostClassifier
from imblearn.metrics import geometric_mean_score
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from math import sqrt







# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project4/data_week4.csv

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 시간 변수 추가
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)


# 불필요한 컬럼 제거2  :  heating_furnace 도 제거할지 고려
df = df.drop(['upper_mold_temp3', 'lower_mold_temp3', 'count', 'heating_furnace'], axis=1)  

# -------------------------------------------- 새로운 데이터 프레임.
# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)


# --------------------------------------------- 이상치 제거
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000)]

df4= df.copy()
df4 = df4[(df4['lower_mold_temp3'] < 60000)]


# 데이터 셋 구분하기

hue_kde(df4, 'passorfail')


df4['tryshot_signal'] = df4['tryshot_signal'].fillna('unknown')

df4.groupby('tryshot_signal').agg(pressure = ('cast_pressure','median'),upper_temp1=('upper_mold_temp1','median'),upper_temp2=('upper_mold_temp2','median') ,lower_temp1=('lower_mold_temp1','median'),lower_temp2=('lower_mold_temp2','median') )




def kde(df, palette='dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns   # number에는 boolean도 포함됨
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.kdeplot(data=df, x=col, fill=True , palette=palette, alpha=alpha)
		plt.title(f'{col}의 확률밀도', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌
	
def hue_kde(df, cat_col, palette='dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns   # number에는 boolean도 포함됨
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.kdeplot(data=df, x=col, fill=True , hue=cat_col, palette=palette, alpha=alpha)
		plt.title(f'{col}의 확률밀도', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌

def box(df, palette='dark'):
	numeric_cols = df.select_dtypes(include=['number']).columns   # number에는 boolean도 포함됨
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.boxplot(data=df, x=col, palette=palette)
		plt.title(f'{col}의 상자그림', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌

def hue_box(df, hue, palette='dark'):
	numeric_cols = df.select_dtypes(include=['number']).columns   # number에는 boolean도 포함됨
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.boxplot(data=df, x=col, hue='passorfail', palette=palette)
		plt.title(f'{col}의 상자그림', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌
     



	
def timeline(df, date_col , y , x_time_n=None, x_time_s = 'MS' ,palette='blue'):
    from matplotlib.ticker import MaxNLocator
    col = df.select_dtypes(include=['boolean','object']).columns
    w_n = sum([len(df[i].unique()) for i in col])
    n = int(np.ceil(w_n/4))  
    u = []
    for i in col:
        for v in df[i].unique():
            u.append([i, v])
    plt.clf()
    plt.figure(figsize=(18, 4*n))
    for index, col_u in enumerate(u, 1):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, index)
        df2 = df[df[col_u[0]] == col_u[1]]
        sns.lineplot(data=df2, x=date_col, y=y, color=palette)
        if x_time_n is not None:
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=x_time_n))  # 눈금 개수 조정
        else:
            ticks_term = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=x_time_s)
            plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
        plt.title(f'{col_u[0]}의 {col_u[1]} 범주', fontsize=15)
    plt.tight_layout()  
    plt.show()
	
def hue_timeline(df, date_col, hue, x_time_n=None, x_time_s='MS', palette='blue'):
    from matplotlib.ticker import MaxNLocator
    num_col = df.select_dtypes(include='number').columns
    plt.clf()
    plt.figure(figsize=(18, 4 * len(num_col)))    
    for index, col in enumerate(num_col, 1):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(len(num_col), 4, index)        
        sns.lineplot(data=df, x=date_col, y=col, hue=hue, color=palette)       
        if x_time_n is not None:
            unique_dates = df[date_col].unique()
            plt.xticks(np.linspace(0, len(unique_dates) - 1, num=x_time_n).astype(int), 
                       unique_dates[np.linspace(0, len(unique_dates) - 1, num=x_time_n).astype(int)])
        elif x_time_s=='W':
            ticks_term = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=x_time_s)
            ticks_days = ticks_term.dayofyear  # ticks_term의 day_of_year 값
            plt.xticks(ticks_days, ticks_term.strftime('%m-%d'))
        else:
            ticks_term = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=x_time_s)
            plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
        plt.title(f'{col}의 시계열', fontsize=15)
    plt.tight_layout()  
    plt.show()

hue_timeline(df3[['molten_temp','cast_pressure','registration_time', 'passorfail']], 'registration_time', 'passorfail', x_time_s='W')

def countbar(df, palette='dark', alpha=0.5):
    col = df.select_dtypes(include=['object','boolean']).columns
    w_n = sum([len(df[i].unique()) for i in col])
    n = int(np.ceil(w_n/4))
    plt.clf()
    plt.figure(figsize=(6*4, 5*n))
    for i, col in enumerate(col, 1): 
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, i)
        ax = sns.countplot(data=df, x=col, order=df[col].value_counts().sort_values().index, palette=palette, alpha=alpha)
        for p in ax.patches:
            plt.text(p.get_x() + p.get_width()/2, p.get_height() , p.get_height(), ha='center',va='bottom')
            # plt.text(p.get_width(), p.get_y() + p.get_height()/2 , p.get_width())
        plt.title(f'{col}의 빈도 그래프', fontsize=20)
    plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
    plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌

def hue_countbar(df, cat, palette='dark', alpha=0.5):
    col = df.select_dtypes(include=['object','boolean']).columns
    w_n = sum([len(df[i].unique()) for i in col])
    n = int(np.ceil(w_n/4))
    plt.clf()
    plt.figure(figsize=(6*4, 5*n))
    for i, col in enumerate(col, 1): 
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, i)
        ax = sns.countplot(data=df, x=col, hue=cat, palette=palette, alpha=alpha)
        for p in ax.patches:
            plt.text(p.get_x() + p.get_width()/2, p.get_height() , p.get_height(), ha='center',va='bottom')
        plt.title(f'{col}의 빈도 그래프', fontsize=20)
    plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
    plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


def scatter(df, palette='dark', alpha=0.5):
    import itertools
    numeric_cols = df.select_dtypes(include=['number']).columns 
    combinations=list(itertools.combinations(numeric_cols, 2))
    n = int(np.ceil(len(combinations)/4))
    plt.clf()
    plt.figure(figsize=(5*4, 4*n))
    for index, col in enumerate(combinations, 1):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, index)
        sns.scatterplot(data=df, x=col[0], y=col[1] , palette=palette, alpha=alpha)
        plt.title(f'{col}', fontsize=20)
    plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
    plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌
    
    
def hue_scatter(df, hue ,palette='dark', alpha=0.5, df2=None):
    import itertools
    num_cols = df.select_dtypes(include='number').columns
    combinations=list(itertools.combinations(num_cols, 2))
    n = int(np.ceil(len(combinations)/4))
    plt.figure(figsize=(5*4, 4*n))
    for index, col in enumerate(combinations, 1):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, index)    
        sns.scatterplot(data=df, x=col[0], y=col[1], hue=hue, palette=palette, alpha=alpha)
        if df2 is not None:
            sns.scatterplot(data=df2, x=col[0], y=col[1], color='green')
        plt.title(f'{col}', fontsize=20)
        plt.legend()
    plt.tight_layout()
    plt.show()


hue_scatter(df2[['upper_mold_temp3','sleeve_temperature','passorfail']], 'passorfail',palette='dark', alpha=0.5)



# --------------------------------------------------

train1 = df2[df2['registration_time']<='2019-03-24']
train2 = df2[(df2['registration_time']<='2019-03-24')&(df2['registration_time']>='2019-02-15')]
test = df2[df2['registration_time']>='2019-03-25']


def IQR_outlier(data) :
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR) 
    upper_bound = Q3 + (1.5 * IQR)
    out_df = pd.concat([lower_bound, upper_bound], axis = 1).T
    out_df.index = ['하한','상한']
    return out_df


num_train1 = train1.select_dtypes(include=('number'))
for col in num_train1.columns:
	train1[f'{col}_outlier'] = np.where((train1[col]<IQR_outlier(num_train1).loc['하한',col])|(train1[col]>IQR_outlier(num_train1).loc['상한',col]),True,False)

num_train2 = train2.select_dtypes(include=('number'))
for col in num_train2.columns:
	train2[f'{col}_outlier'] = np.where((train2[col]<IQR_outlier(num_train2).loc['하한',col])|(train2[col]>IQR_outlier(num_train2).loc['상한',col]),True,False)


train1 = train1.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier'], axis=1)
train2 = train2.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier'], axis=1)






hue_scatter(df2[df2['molten_temp_outlier']==True], 'passorfail')


timeline(df2, 'registration_time', 'molten_temp')
hue_timeline(df2, 'registration_time' , 'mold_code')





# --------------------------------------

hue_kde(train1, 'passorfail')
hue_kde(train2, 'passorfail')
hue_countbar(train1, 'passorfail')
hue_countbar(train2, 'passorfail')
hue_box(train1, 'passorfail')
hue_box(train2, 'passorfail')
out_df = train1[train1['low_section_speed']>60000]
hue_scatter(train1, 'passorfail', df2=out_df)
out_df = train2[train2['low_section_speed']>60000]
hue_scatter(train2, 'passorfail', df2=out_df)
out_df = train1[(train1['low_section_speed']>60000)]
hue_scatter(train1[(train1['physical_strength'] < 60000)], 'passorfail', df2=out_df)

out_df2 = train1[train1['physical_strength'] > 60000]
hue_scatter(train1, 'passorfail', df2=out_df2)


hue_scatter(train1[(train1['low_section_speed'] < 60000)], 'passorfail')

d = train2[(train2['low_section_speed'] < 60000)]
hue_scatter(d, 'passorfail')

# ---------------------------------------  이상치 탐색

pd.set_option('display.max_columns', None)  # 2019-03-16 20:39:50
train1[train1['low_section_speed']>60000]

pd.set_option('display.max_columns', None)   # 2019-03-16 20:39:50
train2[train2['low_section_speed']>60000]

pd.set_option('display.max_columns', None)  # 날짜 다 다름 (1월임)
train1[train1['physical_strength'] > 60000]

pd.set_option('display.max_columns', None)   # 없음
train2[train2['physical_strength'] > 60000]


pd.set_option('display.max_columns', None)  # 
pd.set_option('display.max_rows', None)
train1[train1['molten_temp']<600]


# 'molten_temp'<600 인 mold_code별 비율
g1 = train1[train1['molten_temp']<600].groupby('mold_code').agg(개수 = ('mold_code','count'))
g2 = train1.groupby('mold_code').agg(총=('mold_code','count')).loc[[8412,8573,8600,8722,8917],:]
g3 = pd.merge(g1,g2, on='mold_code')
g3['비율'] = g3['개수']/g3['총']

g1 = train2[train2['molten_temp']<600].groupby('mold_code').agg(개수 = ('mold_code','count'))
g2 = train2.groupby('mold_code').agg(총=('mold_code','count')).loc[[8722],:]
g3 = pd.merge(g1,g2, on='mold_code')
g3['비율'] = g3['개수']/g3['총']


# train1과 train2의 범위 같아?
train1[train1['passorfail']==False]['cast_pressure'].min() # 149.0
train1[train1['passorfail']==False]['cast_pressure'].max() # 348.0
train1[train1['passorfail']==False]['cast_pressure'].sort_values().head(40) # => train1 295초과 정상
train2[train2['passorfail']==False]['cast_pressure'].min() # 149.0
train2[train2['passorfail']==False]['cast_pressure'].max() # 348.0  
train2[train2['passorfail']==False]['cast_pressure'].sort_values().head(40) # => train2 295초과 정상

train1[train1['passorfail']==False]['biscuit_thickness'].min()  # 1.0
train1[train1['passorfail']==False]['biscuit_thickness'].max()  # 60.0
train1[train1['passorfail']==False]['biscuit_thickness'].sort_values() # => train1 20초과 60이하 정상
train2[train2['passorfail']==False]['biscuit_thickness'].min()  # 1.0
train2[train2['passorfail']==False]['biscuit_thickness'].max()  # 60.0
train2[train2['passorfail']==False]['biscuit_thickness'].sort_values() # => train2 20초과 60이하 정상



train1[train1['passorfail']==True]['cast_pressure'].min() # 149.0
train1[train1['passorfail']==True]['cast_pressure'].max() # 348.0
train1[(train1['mold_code']==8722)&(train1['passorfail']==True)]['facility_operation_cycleTime'].sort_values().head(40) # => train1 295초과 정상
train2[train2['passorfail']==False]['cast_pressure'].min() # 149.0
train2[train2['passorfail']==False]['cast_pressure'].max() # 348.0  
train2[train2['passorfail']==False]['cast_pressure'].sort_values().head(40) # => train2 295초과 정상




train1[train1['passorfail']==False]['low_section_speed'].min()  # 1.0
train1[train1['passorfail']==False]['low_section_speed'].max()  # 140.0
train1[train1['passorfail']==False]['low_section_speed'].sort_values().head(40) # => train1 50이상 140이하 정상
train1[train1['passorfail']==False]['low_section_speed'].sort_values().tail(60)
train2[train2['passorfail']==False]['low_section_speed'].min()  # 1.0
train2[train2['passorfail']==False]['low_section_speed'].max()  # 111.0
train2[train2['passorfail']==False]['low_section_speed'].sort_values().head(40) # => train2 50이상 140이하 정상
train2[train2['passorfail']==False]['low_section_speed'].sort_values().tail(40)

train1[train1['passorfail']==False]['high_section_speed'].min()  # 90.0
train1[train1['passorfail']==False]['high_section_speed'].max()  # 345.0
train1[train1['passorfail']==False]['high_section_speed'].sort_values().head(40) # => train1 90이상 205이하 정상
train1[train1['passorfail']==False]['high_section_speed'].sort_values().tail(60)
train2[train2['passorfail']==False]['high_section_speed'].min()  # 90.0
train2[train2['passorfail']==False]['high_section_speed'].max()  # 345.0
train2[train2['passorfail']==False]['high_section_speed'].sort_values().head(40) # => train2 90이상 205이하 정상
train2[train2['passorfail']==False]['high_section_speed'].sort_values().tail(40)






train1.columns

train1['mold_code'].unique()
hue_scatter(train1[train1['mold_code']==8722], 'passorfail')
hue_scatter(train1[train1['mold_code']==8412], 'passorfail')




hue_scatter(train1, 'passorfail')
hue_scatter(train2, 'passorfail')
hue_scatter(test, 'passorfail')







old = df2[df2['registration_time'] < '2019-02-15']
new = df2[df2['registration_time'] >= '2019-02-15']


old['month'] = old['registration_time'].dt.month
old['day'] = old['registration_time'].dt.day
old['hour'] = old['registration_time'].dt.hour
old['minute'] = old['registration_time'].dt.minute
old['second'] = old['registration_time'].dt.second
old['day_of_year'] = old['registration_time'].dt.dayofyear
old['is_special_time'] = old['hour'].apply(lambda x: 1 if x in [7, 8, 19, 20] else 0)
old['molten_temp_g'] = np.where(old['molten_temp']<600, 1,0)  # boxplot
old['cast_pressure_g'] = np.where(old['cast_pressure'] <= 295, 1, 0) # scatter
old['biscuit_thickness_g'] = np.where((old['biscuit_thickness']>60) |(old['biscuit_thickness'] <= 20), 1, 0)   # scatter
old['physical_strength_g'] = np.where(old['physical_strength'] < 600, 1, 0)  # scatter
old['low_section_speed_g'] = np.where((old['low_section_speed'] < 50)|(old['low_section_speed'] > 140), 1, 0)  # scatter
old['high_section_speed_g'] = np.where((old['high_section_speed'] < 90)|(old['high_section_speed'] > 205), 1, 0)  # scatter
num_old = old.select_dtypes(include=('number'))
for col in num_old.columns:
	old[f'{col}_outlier'] = np.where((old[col]<IQR_outlier(num_old).loc['하한',col])|(old[col]>IQR_outlier(num_old).loc['상한',col]),True,False)
old = old.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier'], axis=1)

new['month'] = new['registration_time'].dt.month
new['day'] = new['registration_time'].dt.day
new['hour'] = new['registration_time'].dt.hour
new['minute'] = new['registration_time'].dt.minute
new['second'] = new['registration_time'].dt.second
new['day_of_year'] = new['registration_time'].dt.dayofyear
new['is_special_time'] = new['hour'].apply(lambda x: 1 if x in [7, 8, 19, 20] else 0)
new['molten_temp_g'] = np.where(new['molten_temp']<600, 1,0)  # boxplot
new['cast_pressure_g'] = np.where(new['cast_pressure'] <= 295, 1, 0) # scatter
new['biscuit_thickness_g'] = np.where((new['biscuit_thickness']>60) |(new['biscuit_thickness'] <= 20), 1, 0)   # scatter
new['physical_strength_g'] = np.where(new['physical_strength'] < 600, 1, 0)  # scatter
new['low_section_speed_g'] = np.where((new['low_section_speed'] < 50)|(new['low_section_speed'] > 140), 1, 0)  # scatter
new['high_section_speed_g'] = np.where((new['high_section_speed'] < 90)|(new['high_section_speed'] > 205), 1, 0)  # scatter
num_new = new.select_dtypes(include=('number'))
for col in num_new.columns:
	new[f'{col}_outlier'] = np.where((new[col]<IQR_outlier(num_new).loc['하한',col])|(new[col]>IQR_outlier(num_new).loc['상한',col]),True,False)
new = new.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier'], axis=1)


old = old.drop(['day_of_year_outlier'], axis=1)
old = old.drop(['is_special_time_outlier'], axis=1)
new = new.drop(['day_of_year_outlier'], axis=1)
new = new.drop(['is_special_time_outlier'], axis=1)

new2 = new[new['registration_time'] <= '2019-03-24']

old = old.drop(['registration_time'], axis=1) 
new = new.drop(['registration_time'], axis=1) 
new2 = new2.drop(['registration_time'], axis=1) 


def calculate_psi(old_data, new_data, feature, bins=10):
    if pd.api.types.is_numeric_dtype(old_data[feature]):
        return calculate_psi_numeric(old_data, new_data, feature, bins)
    elif pd.api.types.is_categorical_dtype(old_data[feature]) or pd.api.types.is_object_dtype(old_data[feature]):
        return calculate_psi_categorical(old_data, new_data, feature)
    elif old_data[feature].dtype == 'bool':
    # np.issubdtype(old_data[feature].dtype, np.bool_):
    # pd.api.types.is_bool_dtype(old_data[feature]):
        return calculate_psi_boolean(old_data, new_data, feature)
    else:
        raise ValueError("Unsupported data type")

def calculate_psi_numeric(old_data, new_data, feature, bins=10):
    min_val = min(old_data[feature].min(), new_data[feature].min())
    max_val = max(old_data[feature].max(), new_data[feature].max())
    
    old_counts, _ = np.histogram(old_data[feature], bins=bins, range=(min_val, max_val))
    new_counts, _ = np.histogram(new_data[feature], bins=bins, range=(min_val, max_val))
    
    old_proportions = old_counts / old_counts.sum()
    new_proportions = new_counts / new_counts.sum()

    old_proportions = np.where(old_proportions == 0, 1e-6, old_proportions)
    new_proportions = np.where(new_proportions == 0, 1e-6, new_proportions)

    psi = np.sum((old_proportions - new_proportions) * np.log(old_proportions / new_proportions))
    return psi

def calculate_psi_categorical(old_data, new_data, feature):
    old_counts = old_data[feature].value_counts(normalize=True).sort_index()
    new_counts = new_data[feature].value_counts(normalize=True).sort_index()

    all_categories = old_counts.index.union(new_counts.index)
    old_proportions = old_counts.reindex(all_categories, fill_value=0)
    new_proportions = new_counts.reindex(all_categories, fill_value=0)

    old_proportions = np.where(old_proportions == 0, 1e-6, old_proportions)
    new_proportions = np.where(new_proportions == 0, 1e-6, new_proportions)

    psi = np.sum((old_proportions - new_proportions) * np.log(old_proportions / new_proportions))
    return psi

def calculate_psi_boolean(old_data, new_data, feature):
    # 불리언 컬럼을 0과 1로 변환
    old_counts = old_data[feature].astype(int).value_counts(normalize=True).reindex([1, 0], fill_value=0)
    new_counts = new_data[feature].astype(int).value_counts(normalize=True).reindex([1, 0], fill_value=0)

    # 비율이 0인 경우를 처리하기 위해서 1e-6 추가
    old_proportions = np.where(old_counts.values == 0, 1e-6, old_counts.values)
    new_proportions = np.where(new_counts.values == 0, 1e-6, new_counts.values)

    # PSI 계산
    psi = np.sum((old_proportions - new_proportions) * np.log(old_proportions / new_proportions))
    return psi


psi_df = pd.DataFrame({'columns':[], 'old_new_psi':[]  })
psi_df['columns']=old.columns

for index, i in enumerate(old.columns):
    try:
        psi = calculate_psi(old, new, i)
        #print(f"PSI for {i}컬럼: {psi:.4f}")
        psi_df.loc[index ,'old_new_psi'] = psi
    except Exception as e:
        #print(f"Error calculating PSI for {i}컬럼: {e}")
        psi_df.loc[index ,'old_new_psi'] = np.nan


for i in old.select_dtypes(include=['bool','boolean']).columns:
    psi = calculate_psi_boolean(old, new, i)
    print(f"PSI for {i}컬럼: {psi:.4f}")
    psi_df.loc[psi_df['columns'] == i, 'old_new_psi'] = psi


psi_df = psi_df.drop(index=psi_df[psi_df['columns']=='day_of_year_outlier'].index)
psi_df = psi_df.drop(index=psi_df[psi_df['columns']=='is_special_time_outlier'].index)





for i in train1.columns:
    try:
        psi = calculate_psi(train1, test, i)
        #print(f"PSI for {i}컬럼: {psi:.4f}")
        psi_df.loc[psi_df['columns']==i ,'train1_test_psi'] = psi
    except Exception as e:
        #print(f"Error calculating PSI for {i}컬럼: {e}")
        psi_df.loc[psi_df['columns']==i ,'train1_test_psi'] = np.nan


for i in train1.select_dtypes(include=['bool','boolean']).columns:
    psi = calculate_psi_boolean(train1, test, i)
    print(f"PSI for {i}컬럼: {psi:.4f}")
    psi_df.loc[psi_df['columns'] == i, 'train1_test_psi'] = psi



for i in old.columns:
    try:
        psi = calculate_psi(old, new2, i)
        #print(f"PSI for {i}컬럼: {psi:.4f}")
        psi_df.loc[psi_df['columns']==i ,'old_new2_psi'] = psi
    except Exception as e:
        #print(f"Error calculating PSI for {i}컬럼: {e}")
        psi_df.loc[psi_df['columns']==i ,'old_new2_psi'] = np.nan


for i in old.select_dtypes(include=['bool','boolean']).columns:
    psi = calculate_psi_boolean(old, new2, i)
    print(f"PSI for {i}컬럼: {psi:.4f}")
    psi_df.loc[psi_df['columns'] == i, 'old_new2_psi'] = psi





psi_df[psi_df['old_new_psi']>=0.1].sort_values('old_new_psi',ascending=False)
psi_df[psi_df['train1_test_psi']>=0.1].sort_values('train1_test_psi',ascending=False)
psi_df[psi_df['old_new2_psi']>=0.1].sort_values('old_new2_psi',ascending=False)
psi_df[psi_df['old_new2_psi']>=0.1]

psi_df[['columns', 'old_new2_psi']][psi_df['old_new2_psi']>=0.1].sort_values('old_new2_psi',ascending=False)


len(psi_df[psi_df['old_new_psi']>=0.1]) 
len(psi_df)


a = train1[list(psi_df[psi_df['old_new_psi']>=0.1]['columns'].values) +['passorfail']]
hue_scatter(a, 'passorfail')
b = test[list(psi_df[psi_df['old_new_psi']>=0.1]['columns'].values) +['passorfail']]
hue_scatter(b, 'passorfail')

hue_timeline(df2, 'day_of_year', 'passorfail', x_time_n=10)

hue_timeline(df2, 'hour', 'passorfail', x_time_n=24)












# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project4/data_week4.csv

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 시간 변수 추가 [넣기 / 빼기]
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second
df['day_of_year'] = df['registration_time'].dt.dayofyear
df['is_special_time'] = df['hour'].apply(lambda x: 1 if x in [7, 8, 19, 20] else 0)


# 불필요한 컬럼 제거 [무조건 빼기]
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

# 불필요한 컬럼 제거2  [무조건 빼기]
df = df.drop(['upper_mold_temp3', 'lower_mold_temp3', 'count', 'heating_furnace'], axis=1) 

# -------------------------------------------- 새로운 데이터 프레임.
# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# --------------------------------------------- 이상치 제거 [무조건 하기]
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000)]

# 데이터 1, 2, 테스트데이터 나누기.
train1 = df2[df2['registration_time']<='2019-03-24']
train2 = df2[(df2['registration_time']<='2019-03-24')&(df2['registration_time']>='2019-02-15')]
test = df2[df2['registration_time']>='2019-03-25']

train1 = train1.drop(['registration_time'], axis=1)
train2 = train2.drop(['registration_time'], axis=1)
test = test.drop(['registration_time'], axis=1)


# 데이터1에 대한 수치형 결측치 mean 값으로 채우기
for col in ['molten_temp','molten_volume']:
    train1[col] = train1[col].fillna(train1[col].mean())

# 데이터2에 대한 수치형 결측치 mean 값으로 채우기
for col in ['molten_temp','molten_volume']:
    train2[col] = train2[col].fillna(train2[col].mean())
    
# 테스트 데이터에 대한  수치형 결측치 mean 값으로 채우기
for col in ['molten_temp','molten_volume']:
    test[col] = test[col].fillna(test[col].mean()) 
    
    
# ------------------------------------------- **_g 파생변수
train1['molten_temp_g'] = np.where(train1['molten_temp']<600, 1,0)  # boxplot
train2['molten_temp_g'] = np.where(train2['molten_temp']<600, 1,0)  # boxplot
test['molten_temp_g'] = np.where(test['molten_temp']<600, 1,0)  # boxplot

train1['cast_pressure_g'] = np.where(train1['cast_pressure'] <= 295, 1, 0) # scatter
train2['cast_pressure_g'] = np.where(train2['cast_pressure'] <= 295, 1, 0) # scatter
test['cast_pressure_g'] = np.where(test['cast_pressure'] <= 295, 1, 0) # scatter

train1['biscuit_thickness_g'] = np.where((train1['biscuit_thickness']>60) |(train1['biscuit_thickness'] <= 20), 1, 0)   # scatter
train2['biscuit_thickness_g'] = np.where((train2['biscuit_thickness']>60) |(train2['biscuit_thickness'] <= 20), 1, 0)   # scatter
test['biscuit_thickness_g'] = np.where((test['biscuit_thickness']>60) |(test['biscuit_thickness'] <= 20), 1, 0)   # scatter

train1['physical_strength_g'] = np.where(train1['physical_strength'] < 600, 1, 0)  # scatter
train2['physical_strength_g'] = np.where(train2['physical_strength'] < 600, 1, 0)  # scatter
test['physical_strength_g'] = np.where(test['physical_strength'] < 600, 1, 0)  # scatter

train1['low_section_speed_g'] = np.where((train1['low_section_speed'] < 50)|(train1['low_section_speed'] > 140), 1, 0)  # scatter
train2['low_section_speed_g'] = np.where((train2['low_section_speed'] < 50)|(train2['low_section_speed'] > 140), 1, 0)  # scatter
test['low_section_speed_g'] = np.where((test['low_section_speed'] < 50)|(test['low_section_speed'] > 140), 1, 0)  # scatter

train1['high_section_speed_g'] = np.where((train1['high_section_speed'] < 90)|(train1['high_section_speed'] > 205), 1, 0)  # scatter
train2['high_section_speed_g'] = np.where((train2['high_section_speed'] < 90)|(train2['high_section_speed'] > 205), 1, 0)  # scatter 
test['high_section_speed_g'] = np.where((test['high_section_speed'] < 90)|(test['high_section_speed'] > 205), 1, 0)  # scatter   
    
    
# -------------------------------------------- 이상치 파생변수

# 이상치 여부 컬럼 만들기
def IQR_outlier(data) :
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR) 
    upper_bound = Q3 + (1.5 * IQR)
    out_df = pd.concat([lower_bound, upper_bound], axis = 1).T
    out_df.index = ['하한','상한']
    return out_df

# 데이터1에 대한 이상치 파생변수
num_train1 = train1.select_dtypes(include=('number'))
for col in num_train1.columns:
	train1[f'{col}_outlier'] = np.where((train1[col]<IQR_outlier(num_train1).loc['하한',col])|(train1[col]>IQR_outlier(num_train1).loc['상한',col]),True,False)

# 월일시분초 , day_of_year, is_special_time 변수가 존재했을 때만, 해당 outlier가 생겨서 제거해야 함. 	
train1 = train1.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier', 'day_of_year_outlier', 'is_special_time_outlier'], axis=1)


# 데이터2에 대한 이상치 파생변수
num_train2 = train2.select_dtypes(include=('number'))
for col in num_train2.columns:
	train2[f'{col}_outlier'] = np.where((train2[col]<IQR_outlier(num_train2).loc['하한',col])|(train2[col]>IQR_outlier(num_train2).loc['상한',col]),True,False)

# 월일시분초 , day_of_year, is_special_time 변수가 존재했을 때만, 해당 outlier가 생겨서 제거해야 함. 	
train2 = train2.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier', 'day_of_year_outlier', 'is_special_time_outlier'], axis=1)


# test에 대한 이상치 파생변수
num_test = test.select_dtypes(include=('number'))
for col in num_test.columns:
	test[f'{col}_outlier'] = np.where((test[col]<IQR_outlier(num_train2).loc['하한',col])|(test[col]>IQR_outlier(num_train2).loc['상한',col]),True,False)

# 월일시분초 , day_of_year, is_special_time 변수가 존재했을 때만, 해당 outlier가 생겨서 제거해야 함. 	
test = test.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier', 'day_of_year_outlier', 'is_special_time_outlier'], axis=1)




hue_scatter(old, 'passorfail')
hue_scatter(new, 'passorfail')
hue_box(old, 'passorfail')

c = df2[df2['registration_time']<='2019-03-24']
hue_timeline(c,'day_of_year', 'passorfail', x_time_n=10)

df2.head(30)
df2[df2['registration_time']>'2019-02-14'].head(30)

df2[df2['registration_time'] >= '2019-02-24']
train1


df3 = df2.copy()
prod_date = df2.groupby(['month','day'],as_index=False).agg(생산량=('registration_time','count'), 불량수 = ('passorfail','sum'))
prod_date['Defect_ratio'] = prod_date['불량수']/prod_date['생산량']
df3 = pd.merge(left=df3, right=prod_date, left_on=['month','day'], right_on=['month','day'])


from matplotlib.ticker import MaxNLocator
d=df3[df3['registration_time']<'2019-03-25']
x_time_n=7
plt.clf()
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=d, x='day_of_year', y='Defect_ratio')
if x_time_n is not None:
    unique_dates = d['day_of_year'].unique()
    plt.xticks(np.linspace(0, len(unique_dates) - 1, num=x_time_n).astype(int), 
                unique_dates[np.linspace(0, len(unique_dates) - 1, num=x_time_n).astype(int)])
else:
    ticks_term = pd.date_range(start=d['day_of_year'].min(), end=d['day_of_year'].max(), freq=x_time_s)
    plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 불량률', fontsize=15)
plt.tight_layout()  
plt.show()


hue_timeline(df3,'day_of_year', 'passorfail',x_time_s='W')

hue_box(old,'passorfail')
hue_box(new2,'passorfail')


old['EMS_operation_time'].unique()
old['EMS_operation_time'].value_counts()
new2['EMS_operation_time'].unique()
new2['EMS_operation_time'].value_counts()

old.groupby(['EMS_operation_time','passorfail']).agg(불량 = ('passorfail','count'))
new2.groupby(['EMS_operation_time','passorfail']).agg(불량 = ('passorfail','count'))
df3['day_of_year'].max() 


d=df3[df3['registration_time']<'2019-03-25']
#x_time_n=7
plt.clf()
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=d, x='day_of_year', y='Defect_ratio')
#if x_time_n is not None:
#    unique_dates = d['day_of_year'].unique()
#    plt.xticks(np.linspace(0, len(unique_dates) - 1, num=x_time_n).astype(int), 
#                unique_dates[np.linspace(0, len(unique_dates) - 1, num=x_time_n).astype(int)])
#else:
ticks_term = pd.date_range(start=d['registration_time'].min(), end=d['registration_time'].max(), freq='W')
plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 불량률', fontsize=15)
plt.tight_layout()  
plt.show()



d=df3[df3['registration_time']<'2019-03-25']
plt.clf()
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=d, x='day_of_year', y='Defect_ratio')
ticks_term = pd.date_range(start=d['registration_time'].min(), end=d['registration_time'].max(), freq='W')
ticks_days = ticks_term.dayofyear  # ticks_term의 day_of_year 값
plt.xticks(ticks_days, ticks_term.strftime('%m-%d'))
plt.title(f'날짜별 불량률', fontsize=15)
plt.tight_layout()  
plt.show()


df3[df3['registration_time']>='2019-02-15']  # 46번째 날

old['molten_volume'].unique()
new2['molten_volume'].unique()


timeline(df2, )

hue_scatter(old[old['upper_mold_temp1']<1400][['upper_mold_temp1','lower_mold_temp2','passorfail']],'passorfail')


for i in df.columns:
    print(f"{i}컬럼의 결측치 개수 :",df[i].isna().sum())





for i in ['upper_mold_temp3', 'lower_mold_temp3', 'heating_furnace']:
    print(f"{i}컬럼의 결측치 개수 :",df[i].isna().sum())


