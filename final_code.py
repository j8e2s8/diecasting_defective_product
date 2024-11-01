import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from scipy.stats import chi2_contingency


# ------------------------------------------------- df : 원본 데이터
# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project4/data_week4.csv


# df 확인
df.head()

df.describe()

# 자료형 확인
df.info()


# 고유값 개수 확인 -> 1개인거 제거할거임
for i in df.columns:
	print(f'{i}컬럼의 unique 개수 :',len(df[i].unique()))
      

# 고유값 종류 확인
for i in df.columns:
	if len(df[i].unique()) <= 15:
		print(f'{i}컬럼의 unique :', df[i].unique())


# 결측치 개수 확인
for i in df.columns:
    print(f"{i}컬럼의 결측치 개수 :",df[i].isna().sum())
    



# ------------------------------------------------- df_dropna : 타켓변수 결측치 제거
# 타겟변수에 있는 결측치 1개 제거하기
df_dropna = df.dropna(subset=['passorfail'])

# 결측치 개수 확인
for i in df_dropna.columns:
    print(f"{i}컬럼의 결측치 개수 :", df_dropna[i].isna().sum())



# ------------------------------------------------- df_type : 타켓변수 결측치 제거, 자료형 변환
# 자료형 변경
df_type = df_dropna.copy()
df_type['mold_code'] = df_type['mold_code'].astype('object')
df_type['registration_time'] = pd.to_datetime(df_type['registration_time'])
df_type['passorfail'] = df_type['passorfail'].astype('bool')


# 자료형 확인
df_type.info()

# ------------------------------------------------- df_drop1 : 타켓변수 결측치 제거, 자료형 변환, 변수 제거1
# 불필요한 컬럼 제거
df_drop1 = df_type.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)



# ------------------------------------------------- df_add : 타켓변수 결측치 제거, 자료형 변환, 변수 제거1, 시간 파생변수
df_add = df_drop1.copy()

# 시간 관련 파생변수
df_add['month'] = df_add['registration_time'].dt.month
df_add['day'] = df_add['registration_time'].dt.day
df_add['hour'] = df_add['registration_time'].dt.hour
df_add['minute'] = df_add['registration_time'].dt.minute
df_add['second'] = df_add['registration_time'].dt.second
df_add['day_of_year'] = df_add['registration_time'].dt.dayofyear
df_add['is_special_time'] = df_add['hour'].apply(lambda x: 1 if x in [7, 8, 19, 20] else 0)


# groupting 파생변수
df_add['molten_temp_g'] = np.where(df_add['molten_temp']<600, 1,0)  # boxplot
df_add['cast_pressure_g'] = np.where(df_add['cast_pressure'] <= 295, 1, 0) # scatter
df_add['biscuit_thickness_g'] = np.where((df_add['biscuit_thickness']>60) |(df_add['biscuit_thickness'] <= 20), 1, 0)   # scatter
df_add['physical_strength_g'] = np.where(df_add['physical_strength'] < 600, 1, 0)  # scatter
df_add['low_section_speed_g'] = np.where((df_add['low_section_speed'] < 50)|(df_add['low_section_speed'] > 140), 1, 0)  # scatter
df_add['high_section_speed_g'] = np.where((df_add['high_section_speed'] < 90)|(df_add['high_section_speed'] > 205), 1, 0)  # scatter



# 시간변수 값 확인
df_add[['registration_time', 'month','day','hour','minute','second','day_of_year','is_special_time']]


# ------------------------------------------------- df_drop2 : 타켓변수 결측치 제거, 자료형 변환, 변수 제거1, 시간 파생변수, 변수 제거2
# 불필요한 컬럼 제거2  :  heating_furnace 도 제거할지 고려
df_drop2 = df_add.drop(['upper_mold_temp3', 'lower_mold_temp3', 'count', 'heating_furnace'], axis=1)  



# ------------------------------------------------- df_tryshot : 타켓변수 결측치 제거, 자료형 변환, 변수 제거1, 시간 파생변수, 변수 제거2, tryshot D 제거
# tryshot D일 때와 아닐 때의 압력과 온도 차이 확인
df_tryshotunknown = df_drop2.copy()
df_tryshotunknown['tryshot_signal'] = df_tryshotunknown['tryshot_signal'].fillna('unknown')
df_tryshotunknown.groupby('tryshot_signal').agg(pressure = ('cast_pressure','median'),upper_temp1=('upper_mold_temp1','median'),upper_temp2=('upper_mold_temp2','median') ,lower_temp1=('lower_mold_temp1','median'),lower_temp2=('lower_mold_temp2','median') )


# tryshot_signal == nan 일 때만 이용하기 (상용 제품일 때만)
df_tryshot = df_drop2[df_drop2['tryshot_signal'].isna()]
df_tryshot = df_tryshot.drop('tryshot_signal', axis=1)



# ------------------------------------------------- df2 : 타켓변수 결측치 제거, 자료형 변환, 변수 제거1, 시간 파생변수, 변수 제거2, tryshot D 제거, 이상치 제거
# 이상치 제거
df2 = df_tryshot[(df_tryshot['physical_strength'] < 60000) & (df_tryshot['low_section_speed'] < 60000)]



# ------------------------------------------------- 날짜별 불량률 시계열 그래프
df2_defect_plot = df2.copy()
defect_g = df2_defect_plot.groupby(['month','day'], as_index=False).agg(product_count=('passorfail','count'), defect_count = ('passorfail','sum'))
defect_g['defect_ratio'] = defect_g['defect_count']/defect_g['product_count']
df2_defect_plot = pd.merge(left=df2_defect_plot , right=defect_g, on =['month','day'])


info = dict(df=df2_defect_plot, x_time='day_of_year', y='defect_ratio', x_time_term = 7)
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
unique_dates = info['df'][info['x_time']].unique()
plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
plt.title(f'날짜별 불량률', fontsize=15)
plt.tight_layout()  
plt.show()





# ------------------------------------------------- heating_furnace 범주 빈도 그래프
df_unknown = df_drop1.copy()
df_unknown['heating_furnace'] = df_unknown['heating_furnace'].fillna('unknown')

info = dict(df = df_unknown, col = 'heating_furnace',palette = 'dark', al=0.5)
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
ax = sns.countplot(data=info['df'], x=info['col'], palette=info['palette'], alpha=info['al'])
for p in ax.patches:
	plt.text(p.get_x() + p.get_width()/2, p.get_height() , p.get_height(), ha='center',va='bottom')
plt.title(f'{info['col']}의 빈도 그래프', fontsize=20)
plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌




# ------------------------------------------------- 'upper_mold_temp3','sleeve_temperature' 산점도 그래프
info = dict(df = df_drop1, col = ['upper_mold_temp3','sleeve_temperature'] , hue='passorfail', palette = 'dark', al=0.5)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.scatterplot(data=info['df'], x=info['col'][0], y=info['col'][1], hue=info['hue'], palette=info['palette'], alpha=info['al'])
plt.title(f'{info['col'][0]}, {info['col'][1]}', fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()



# ------------------------------------------------- tryshot_signal 'passorfail'별 빈도 그래프
info = dict(df = df_tryshotunknown, col = 'tryshot_signal',hue='passorfail', palette = 'dark', alpha=0.5)
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
ax = sns.countplot(data=info['df'], x=info['col'], hue=info['hue'],palette=info['palette'], alpha=info['alpha'])
for p in ax.patches:
	plt.text(p.get_x() + p.get_width()/2, p.get_height() , p.get_height(), ha='center',va='bottom')
plt.title(f'{info['col']}의 빈도 그래프', fontsize=20)
plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


# ------------------------------------------------- tryshot_signal 'passorfail'별 빈도 그래프
info = dict(df = df_tryshotunknown, col = 'tryshot_signal',hue='passorfail', palette = 'dark', alpha=0.5)
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
ax = sns.countplot(data=info['df'], x=info['col'], hue=info['hue'],palette=info['palette'], alpha=info['alpha'])
for p in ax.patches:
	plt.text(p.get_x() + p.get_width()/2, p.get_height() , p.get_height(), ha='center',va='bottom')
plt.title(f'{info['col']}의 빈도 그래프', fontsize=20)
plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌



# ------------------------------------------------- 이상치 제거 전, physical_strength 'passorfail'별 box 그래프
info = dict(df = df_tryshot, col = 'physical_strength',hue='passorfail', palette = 'coolwarm')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.boxplot(data=info['df'], x=info['col'], hue=info['hue'], palette=info['palette'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


# ------------------------------------------------- 이상치 제거 전, low_section_speed 'passorfail'별 box 그래프
info = dict(df = df_tryshot, col = 'low_section_speed',hue='passorfail', palette = 'coolwarm')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.boxplot(data=info['df'], x=info['col'], hue=info['hue'], palette=info['palette'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


# ------------------------------------------------- 이상치 제거 후, low_section_speed 'passorfail'별 box 그래프
info = dict(df = df2[df2['month']==1], col = 'low_section_speed',hue='passorfail', palette = 'coolwarm')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.boxplot(data=info['df'], x=info['col'], hue=info['hue'], palette=info['palette'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌

info = dict(df = df2[df2['month']==2], col = 'low_section_speed',hue='passorfail', palette = 'coolwarm')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.boxplot(data=info['df'], x=info['col'], hue=info['hue'], palette=info['palette'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


# ------------------------------------------------- 시간별 tryshot D 개수 시계열 그래프
dcount_g = df_drop2.groupby('hour',as_index=False).agg(D_count=('tryshot_signal','count'))

info = dict(df=dcount_g, x_time='hour', y='D_count', x_time_term = 1)
plt.figure(figsize=(5,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
sns.scatterplot(data=info['df'], x=info['x_time'], y=info['y'])
unique_dates = info['df'][info['x_time']].unique()
plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
plt.title(f'날짜별 {info['y']}', fontsize=15)
plt.tight_layout()  
plt.show()


# ------------------------------------------------- 시간별 cast_pressure 'passorfail'별 시계열 그래프
info = dict(df=df2, x_time='hour', y='cast_pressure', hue='passorfail', x_time_term = 1)
plt.figure(figsize=(5,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], hue=info['hue'])
unique_dates = info['df'][info['x_time']].unique()
plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
plt.title(f'시간별 {info['y']}', fontsize=15)
plt.tight_layout()  
plt.show()


# ------------------------------------------------- 시간별 upper_mold_temp1 'passorfail'별 시계열 그래프
info = dict(df=df2, x_time='hour', y='upper_mold_temp1', hue='passorfail', x_time_term = 1)
plt.figure(figsize=(5,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], hue=info['hue'])
unique_dates = info['df'][info['x_time']].unique()
plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
plt.title(f'시간별 {info['y']}', fontsize=15)
plt.tight_layout()  
plt.show()

# ------------------------------------------------- 시간별 upper_mold_temp2 'passorfail'별 시계열 그래프
info = dict(df=df2, x_time='hour', y='lower_mold_temp2', hue='passorfail', x_time_term = 1)
plt.figure(figsize=(5,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], hue=info['hue'])
unique_dates = info['df'][info['x_time']].unique()
plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
plt.title(f'시간별 {info['y']}', fontsize=15)
plt.tight_layout()  
plt.show()


# ------------------------------------------------- molten_temp 'passorfail'별 box 그래프
info = dict(df = df2, col = 'molten_temp', hue='passorfail', palette = 'coolwarm')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.boxplot(data=info['df'], x=info['col'], hue=info['hue'], palette=info['palette'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


# ------------------------------------------------- 'molten_temp','cast_pressure' 산점도 그래프
info = dict(df = df_drop1, col = ['molten_temp','cast_pressure'] , hue='passorfail', palette = 'dark', al=0.5)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.scatterplot(data=info['df'], x=info['col'][0], y=info['col'][1], hue=info['hue'], palette=info['palette'], alpha=info['al'])
plt.title(f'{info['col'][0]}, {info['col'][1]}', fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------- passorfail 'passorfail'별 막대 그래프
info = dict(df = df2, col ='passorfail',hue='passorfail', palette = 'dark', alpha=0.5)
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
ax = sns.countplot(data=info['df'], x=info['col'], hue=info['hue'],palette=info['palette'], alpha=info['alpha'])
for p in ax.patches:
	plt.text(p.get_x() + p.get_width()/2, p.get_height() , p.get_height(), ha='center',va='bottom')
plt.title(f'{info['col']}의 빈도 그래프', fontsize=20)
plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


np.round(87998/(87998+2093)*100,1)
np.round(2093/(87998+2093)*100,1)


# ------------------------------------------------- 시간별 upper_mold_temp2 'passorfail'별 시계열 그래프
info = dict(df=df2, x_time='day_of_year', y='lower_mold_temp2', hue='passorfail', x_time_term = 7)
plt.figure(figsize=(5,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], hue=info['hue'])
unique_dates = info['df'][info['x_time']].unique()
plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
plt.title(f'날짜별 {info['y']}', fontsize=15)
plt.tight_layout()  
plt.show()


# ------------------------------------------------- 시간별 upper_mold_temp1 'passorfail'별 시계열 그래프
info = dict(df=df2, x_time='day_of_year', y='upper_mold_temp1', hue='passorfail', x_time_term = 7)
plt.figure(figsize=(5,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], hue=info['hue'])
unique_dates = info['df'][info['x_time']].unique()
plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
plt.title(f'날짜별 {info['y']}', fontsize=15)
plt.tight_layout()  
plt.show()


# ------------------------------------------------- 시간별 cast_pressure 'passorfail'별 시계열 그래프
info = dict(df=df2, x_time='day_of_year', y='cast_pressure', hue='passorfail', x_time_term = 7)
plt.figure(figsize=(5,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], hue=info['hue'])
unique_dates = info['df'][info['x_time']].unique()
plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
plt.title(f'날짜별 {info['y']}', fontsize=15)
plt.tight_layout()  
plt.show()




# ------------------------------------------------- old, new2 데이터 프레임 만들기
# 1월 2일 ~ 2월 14일 : old 데이터 프레임 만들기
old = df2[df2['registration_time'] < '2019-02-15']
# 2월 15일 ~ 3월 24일 : new2 데이터 프레임 만들기
new2 = df2[(df2['registration_time'] >= '2019-02-15') & (df2['registration_time']<'2019-03-25')]


# 이상치 여부 파생변수
def IQR_outlier(data) :
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR) 
    upper_bound = Q3 + (1.5 * IQR)
    out_df = pd.concat([lower_bound, upper_bound], axis = 1).T
    out_df.index = ['하한','상한']
    return out_df


num_old = old.select_dtypes(include=('number'))
for col in num_old.columns:
	old[f'{col}_outlier'] = np.where((old[col]<IQR_outlier(num_old).loc['하한',col])|(old[col]>IQR_outlier(num_old).loc['상한',col]),True,False)

num_new2 = new2.select_dtypes(include=('number'))
for col in num_new2.columns:
	new2[f'{col}_outlier'] = np.where((new2[col]<IQR_outlier(num_new2).loc['하한',col])|(new2[col]>IQR_outlier(num_new2).loc['상한',col]),True,False)


old.columns.to_list().index('EMS_operation_time_outlier')
old = old.iloc[:,:49]
new2 = new2.iloc[:,:49]



# 결측치 확인
for i in old.columns:
    print(f"{i}컬럼의 결측치 개수 :", old[i].isna().sum())
    
for i in new2.columns:
    print(f"{i}컬럼의 결측치 개수 :", new2[i].isna().sum())
    

# 결측치 대체하기
for col in ['molten_temp','molten_volume']:
	old[col] = old[col].fillna(old[col].mean())
     
for col in ['molten_temp','molten_volume']:
	new2[col] = new2[col].fillna(new2[col].mean())



# ------------------------------------------------- PSI 구하기
# PSI 계산 함수 만들기
def calculate_psi_numeric(old_data, new_data, feature):
    combined_data = np.concatenate([old_data[feature], new_data[feature]])
    quantiles = np.percentile(combined_data, np.linspace(0, 100, 11))
    
    old_counts, _ = np.histogram(old_data[feature], bins=quantiles)
    new_counts, _ = np.histogram(new_data[feature], bins=quantiles)

    print("Old counts:", old_counts)
    print("New counts:", new_counts)
    print("Sum of old counts:", old_counts.sum())
    print("Sum of new counts:", new_counts.sum())

    if old_counts.sum() == 0 or new_counts.sum() == 0:
        raise ValueError("Old or new data counts are zero, cannot calculate PSI.")

    old_proportions = old_counts / old_counts.sum()
    new_proportions = new_counts / new_counts.sum()

    print("Old proportions:", old_proportions)
    print("New proportions:", new_proportions)

    old_proportions = np.where(old_proportions == 0, 1e-6, old_proportions)
    new_proportions = np.where(new_proportions == 0, 1e-6, new_proportions)

    psi = np.sum((old_proportions - new_proportions) * np.log(old_proportions / new_proportions))

    plt.figure(figsize=(10, 6))
    plt.hist(old_data[feature], bins=quantiles, alpha=0.5, label='Old Data', edgecolor='black')
    plt.hist(new_data[feature], bins=quantiles, alpha=0.5, label='New Data', edgecolor='red')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title('Histogram of Old and New Data')
    plt.legend()
    plt.show()
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


psi_df = pd.DataFrame({'columns':[], 'old_new2_psi':[] })
psi_df['columns']=old.columns


for i in old.select_dtypes(include=['number']).columns:
    psi = calculate_psi_numeric(old, new2, i)
    # print(f"PSI for {i}컬럼: {psi:.4f}")
    psi_df.loc[psi_df['columns'] == i, 'old_new2_psi'] = psi
    

for i in old.select_dtypes(include=['object']).columns:
    psi = calculate_psi_categorical(old, new2, i)
    # print(f"PSI for {i}컬럼: {psi:.4f}")
    psi_df.loc[psi_df['columns'] == i, 'old_new2_psi'] = psi
    


for i in old.select_dtypes(include=['bool','boolean']).columns:
    psi = calculate_psi_boolean(old, new2, i)
    # print(f"PSI for {i}컬럼: {psi:.4f}")
    psi_df.loc[psi_df['columns'] == i, 'old_new2_psi'] = psi
    

psi_df[psi_df['old_new2_psi']>0.1].sort_values('old_new2_psi', ascending=False)


# ------------------------------------------------- 2월 15일 전 후로 lower_mold_temp2 확률밀도 변화 비교
info = dict( df = old , df2 =new2, col ='lower_mold_temp2', palette = 'dark', alpha=0.5)
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.kdeplot(data=info['df'], x=info['col'], fill=True , palette=info['palette'], alpha=info['alpha'])
sns.kdeplot(data=info['df2'], x=info['col'], fill=True , palette=info['palette'], alpha=info['alpha'])
plt.title(f'{info['col']}의 확률밀도 분포', fontsize=16)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌






# ------------------------------------------------- 2월 15일 전 후로 컬럼별 분포 변화 비교
plt.figure(figsize=(5*2, 4*2))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
sns.violinplot(data=old, x='passorfail', y='upper_mold_temp1', inner='quartile', palette='dark', alpha=0.5)
sns.boxplot(data=old, x='passorfail', y='upper_mold_temp1', whis=1.5, color='k', boxprops=dict(alpha=0.3))
plt.ylim([0,400])
plt.title(f'이전 upper_mold_temp1', fontsize=20)

plt.subplot(2, 2, 2)
sns.violinplot(data=new2, x='passorfail', y='upper_mold_temp1', inner='quartile', palette='dark', alpha=0.5)
sns.boxplot(data=new2, x='passorfail', y='upper_mold_temp1', whis=1.5, color='k', boxprops=dict(alpha=0.3))
plt.ylim([0,400])
plt.title(f'이후 upper_mold_temp1', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌




plt.figure(figsize=(5*2, 4*2))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
sns.violinplot(data=old, x='passorfail', y='lower_mold_temp2', inner='quartile', palette='dark', alpha=0.5)
sns.boxplot(data=old, x='passorfail', y='lower_mold_temp2', whis=1.5, color='k', boxprops=dict(alpha=0.3))
plt.ylim([0,550])
plt.title(f'이전 lower_mold_temp2', fontsize=20)

plt.subplot(2, 2, 2)
sns.violinplot(data=new2, x='passorfail', y='lower_mold_temp2', inner='quartile', palette='dark', alpha=0.5)
sns.boxplot(data=new2, x='passorfail', y='lower_mold_temp2', whis=1.5, color='k', boxprops=dict(alpha=0.3))
plt.ylim([0,550])
plt.title(f'이후 lower_mold_temp2', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌



plt.figure(figsize=(5*2, 4*2))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
sns.scatterplot(data=old, x='upper_mold_temp1', y='lower_mold_temp2', hue='passorfail', palette='dark', alpha=0.5)
plt.xlim([0,360])
plt.ylim([0,530])
plt.title(f'이전', fontsize=20)

plt.subplot(2, 2, 2)
sns.scatterplot(data=new2, x='upper_mold_temp1', y='lower_mold_temp2', hue='passorfail', palette='dark', alpha=0.5)
plt.xlim([0,360])
plt.ylim([0,530])
plt.title(f'이후', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌




# ------------------------------------------------- 중요변수 전후 분포 비교
plt.figure(figsize=(8, 10))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
sns.violinplot(data=old, x='passorfail', y='cast_pressure', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,400])
plt.title(f'2019-01-02 ~ 2019-02-15', fontsize=18)

plt.subplot(2, 2, 2)
sns.violinplot(data=new2, x='passorfail', y='cast_pressure', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,400])
plt.title(f'2019-02-15 ~ 2019-03-24', fontsize=18)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


plt.figure(figsize=(8, 10))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
sns.violinplot(data=old, x='passorfail', y='lower_mold_temp2', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,360])
plt.title(f'2019-01-02 ~ 2019-02-15', fontsize=18)

plt.subplot(2, 2, 2)
sns.violinplot(data=new2, x='passorfail', y='lower_mold_temp2', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,360])
plt.title(f'2019-02-15 ~ 2019-03-24', fontsize=18)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


plt.figure(figsize=(8, 10))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
sns.violinplot(data=old, x='passorfail', y='upper_mold_temp1', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,360])
plt.title(f'2019-01-02 ~ 2019-02-15', fontsize=18)

plt.subplot(2, 2, 2)
sns.violinplot(data=new2, x='passorfail', y='upper_mold_temp1', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,360])
plt.title(f'2019-02-15 ~ 2019-03-24', fontsize=18)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌





plt.figure(figsize=(8, 10))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
sns.violinplot(data=old, x='passorfail', y='low_section_speed', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,360])
plt.title(f'2019-01-02 ~ 2019-02-15', fontsize=18)

plt.subplot(2, 2, 2)
sns.violinplot(data=new2, x='passorfail', y='low_section_speed', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,360])
plt.title(f'2019-02-15 ~ 2019-03-24', fontsize=18)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌



plt.figure(figsize=(8, 10))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
sns.violinplot(data=old, x='passorfail', y='high_section_speed', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,360])
plt.title(f'2019-01-02 ~ 2019-02-15', fontsize=18)

plt.subplot(2, 2, 2)
sns.violinplot(data=new2, x='passorfail', y='high_section_speed', inner='quartile', palette='dark', alpha=0.5)
plt.ylim([0,360])
plt.title(f'2019-02-15 ~ 2019-03-24', fontsize=18)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌



# ------------------------------------------------- 'low_section_speed','high_section_speed' 산점도 그래프
info = dict(df = df2, col = ['low_section_speed','high_section_speed'] , hue='passorfail', palette = 'dark', al=0.5)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.scatterplot(data=info['df'], x=info['col'][0], y=info['col'][1], hue=info['hue'], palette=info['palette'], alpha=info['al'])
plt.title(f'{info['col'][0]}, {info['col'][1]}', fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()


