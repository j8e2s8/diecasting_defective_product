import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.metrics import geometric_mean_score

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지


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
df = df.drop(['upper_mold_temp3', 'lower_mold_temp3'], axis=1) 

# -------------------------------------------- 새로운 데이터 프레임.
# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# --------------------------------------------- 이상치 제거 [무조건 하기]
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000)]

# 범주형 변수를 one-hot encoding
# df2_dummie = pd.get_dummies(df2, columns = ['working',	'mold_code'], drop_first=True)

# 데이터 1, 2, 테스트데이터 나누기.
train1 = df2[df2['registration_time']<'2019-03-25']
train2 = df2[(df2['registration_time']<'2019-03-25')&(df2['registration_time']>='2019-02-15')]
test = df2[df2['registration_time']>='2019-03-25']

# X, y 나누기
X_train1 = train1.drop(['registration_time', 'passorfail'], axis=1)
y_train1 = train1['passorfail']
X_train2 = train2.drop(['registration_time', 'passorfail'], axis=1)
y_train2 = train2['passorfail']
X_test = test.drop(['registration_time', 'passorfail'], axis=1)
y_test = test['passorfail']


# 데이터1에 대한 수치형 결측치 mean 값으로 채우기
for col in ['molten_temp','molten_volume']:
    X_train1[col] = X_train1[col].fillna(X_train1[col].mean())

# 데이터2에 대한 수치형 결측치 mean 값으로 채우기
for col in ['molten_temp','molten_volume']:
    X_train2[col] = X_train2[col].fillna(X_train2[col].mean())
    
# 테스트 데이터에 대한  수치형 결측치 mean 값으로 채우기
for col in ['molten_temp','molten_volume']:
    X_test[col] = X_test[col].fillna(X_test[col].mean()) 
    
    
# ------------------------------------------- **_g 파생변수
X_train1['molten_temp_g'] = np.where(X_train1['molten_temp']<600, 1,0)  # boxplot
X_train2['molten_temp_g'] = np.where(X_train2['molten_temp']<600, 1,0)  # boxplot
X_test['molten_temp_g'] = np.where(X_test['molten_temp']<600, 1,0)  # boxplot

X_train1['cast_pressure_g'] = np.where(X_train1['cast_pressure'] <= 295, 1, 0) # scatter
X_train2['cast_pressure_g'] = np.where(X_train2['cast_pressure'] <= 295, 1, 0) # scatter
X_test['cast_pressure_g'] = np.where(X_test['cast_pressure'] <= 295, 1, 0) # scatter

X_train1['biscuit_thickness_g'] = np.where((X_train1['biscuit_thickness']>60) |(X_train1['biscuit_thickness'] <= 20), 1, 0)   # scatter
X_train2['biscuit_thickness_g'] = np.where((X_train2['biscuit_thickness']>60) |(X_train2['biscuit_thickness'] <= 20), 1, 0)   # scatter
X_test['biscuit_thickness_g'] = np.where((X_test['biscuit_thickness']>60) |(X_test['biscuit_thickness'] <= 20), 1, 0)   # scatter

X_train1['physical_strength_g'] = np.where(X_train1['physical_strength'] < 600, 1, 0)  # scatter
X_train2['physical_strength_g'] = np.where(X_train2['physical_strength'] < 600, 1, 0)  # scatter
X_test['physical_strength_g'] = np.where(X_test['physical_strength'] < 600, 1, 0)  # scatter

X_train1['low_section_speed_g'] = np.where((X_train1['low_section_speed'] < 50)|(X_train1['low_section_speed'] > 140), 1, 0)  # scatter
X_train2['low_section_speed_g'] = np.where((X_train2['low_section_speed'] < 50)|(X_train2['low_section_speed'] > 140), 1, 0)  # scatter
X_test['low_section_speed_g'] = np.where((X_test['low_section_speed'] < 50)|(X_test['low_section_speed'] > 140), 1, 0)  # scatter

X_train1['high_section_speed_g'] = np.where((X_train1['high_section_speed'] < 90)|(X_train1['high_section_speed'] > 205), 1, 0)  # scatter
X_train2['high_section_speed_g'] = np.where((X_train2['high_section_speed'] < 90)|(X_train2['high_section_speed'] > 205), 1, 0)  # scatter 
X_test['high_section_speed_g'] = np.where((X_test['high_section_speed'] < 90)|(X_test['high_section_speed'] > 205), 1, 0)  # scatter   
    
    
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
num_X_train1 = X_train1.select_dtypes(include=('number'))
for col in num_X_train1.columns:
	X_train1[f'{col}_outlier'] = np.where((X_train1[col]<IQR_outlier(num_X_train1).loc['하한',col])|(X_train1[col]>IQR_outlier(num_X_train1).loc['상한',col]),True,False)

# 월일시분초 , day_of_year, is_special_time 변수가 존재했을 때만, 해당 outlier가 생겨서 제거해야 함. 	
X_train1 = X_train1.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier', 'day_of_year_outlier', 'is_special_time_outlier'], axis=1)


# 데이터2에 대한 이상치 파생변수
num_X_train2 = X_train2.select_dtypes(include=('number'))
for col in num_X_train2.columns:
	X_train2[f'{col}_outlier'] = np.where((X_train2[col]<IQR_outlier(num_X_train2).loc['하한',col])|(X_train2[col]>IQR_outlier(num_X_train2).loc['상한',col]),True,False)

# 월일시분초 , day_of_year, is_special_time 변수가 존재했을 때만, 해당 outlier가 생겨서 제거해야 함. 	
X_train2 = X_train2.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier', 'day_of_year_outlier', 'is_special_time_outlier'], axis=1)


# test에 대한 이상치 파생변수
num_X_test = X_test.select_dtypes(include=('number'))
for col in num_X_test.columns:
	X_test[f'{col}_outlier'] = np.where((X_test[col]<IQR_outlier(num_X_train2).loc['하한',col])|(X_test[col]>IQR_outlier(num_X_train2).loc['상한',col]),True,False)

# 월일시분초 , day_of_year, is_special_time 변수가 존재했을 때만, 해당 outlier가 생겨서 제거해야 함. 	
X_test = X_test.drop(['month_outlier', 'day_outlier', 'hour_outlier', 'minute_outlier', 'second_outlier', 'day_of_year_outlier', 'is_special_time_outlier'], axis=1)




cat_X_train1 = X_train1.select_dtypes(include=('object','bool'))
for col in cat_X_train1.columns:
	contingency_table = pd.crosstab(X_train1[col], y_train1)
	chi2, p, dof, expected = chi2_contingency(contingency_table)

	# 예상 빈도 출력
	print(f"{col}의 예상 빈도:")
	print(expected)

	# 각 예상 빈도가 5 이상인지 확인
	if (expected < 5).any():
		print(f"{col}의 예상 빈도 중 5 미만인 값이 있습니다.")
	else:
		print(f"{col}의 모든 예상 빈도가 5 이상입니다.")
		
	print(f"{col}의 p-값 :", p)

	# 귀무가설 채택 여부 판단
	if p < 0.05:
		print(f"{col}의 귀무가설 기각: 독립변수와 타겟 변수는 연관성이 있다.", "\n","-"*30)
	else:
		print(f"{col}의 귀무가설 채택: 독립변수와 타겟 변수는 연관성이 없다.", "\n","-"*30)






import scipy.stats as stats
for col in ['working', 'lower_mold_temp1_outlier']:
	contingency_table = pd.crosstab(X_train1[col], y_train1)

	# 피셔의 정확 검정 실행
	oddsratio, p_value = stats.fisher_exact(contingency_table)

	print(f"오즈비(odds ratio): {oddsratio}")
	print(f"{col}의 p-값 : {p_value}")

	# 귀무가설 채택 여부 판단
	if p_value < 0.05:
		print(f"{col}의 귀무가설 기각: 독립변수와 타겟 변수는 연관성이 있다.", "\n","-"*30)
	else:
		print(f"{col}의 귀무가설 채택: 독립변수와 타겟 변수는 연관성이 없다.", "\n","-"*30)



df2['heating_furnace'] = df2['heating_furnace'].fillna('unknown')
countbar(df2[['heating_furnace']], palette='dark', alpha=0.5)


df['tryshot_signal'] = df['tryshot_signal'].fillna('unknown')
hue_countbar(df[['tryshot_signal','passorfail']],'passorfail', palette='dark', alpha=0.5)


trygroup = df.groupby('hour',as_index=False).agg(Dcount=('tryshot_signal','count'))
df = pd.merge(df, trygroup, on='hour')

timeline(df[['day_of_year']])
timeline(df,'hour','Dcount',x_time_n=24)


df.info()


hue_box(df2[df2['month']==1], 'passorfail', palette='coolwarm')
hue_box(df2[df2['month']==2], 'passorfail', palette='coolwarm')
hue_box(df2[df2['month']==3], 'passorfail', palette='coolwarm')