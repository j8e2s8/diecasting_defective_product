
# 전/후 공통 중요 변수 : 
# sleeve_temperature, lower_mold_temp2, upper_mold_temp2, biscuit_thickness, upper_mold_temp1, lower_mold_temp1

# cast_pressure, low_section_speed_g, low_section_speed, high_section_speed 생김
# minute, physical_strength 없어짐



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


df2.select_dtypes(include=['object']).columns
num_df2 = df2.select_dtypes(include=['number','bool']).iloc[:,:-7]

corr_m = num_df2.select_dtypes(include=['number','bool']).corr(method='spearman')
# 히트맵 시각화
plt.figure(figsize=(11, 11))  # 그림 크기 설정
sns.heatmap(corr_m, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})  # annot_kws로 글씨 크기 조정
plt.title('Correlation Matrix', fontsize=14)  # 제목 설정 (작게)
plt.xticks(rotation=45, ha='right', fontsize=10)  # x축 라벨 회전 및 크기 조정
plt.yticks(rotation=0, fontsize=10)  # y축 라벨 수평 및 크기 조정
plt.tight_layout()  # 레이아웃 최적화
plt.show()



corr_train1 = df2[df2['registration_time']<'2019-02-15'].select_dtypes(include=['number','bool']).iloc[:,:-7]
corr_train2 = df2[(df2['registration_time']<'2019-03-25')&(df2['registration_time']>='2019-02-15')].select_dtypes(include=['number','bool']).iloc[:,:-7]
corr_train3 = df2[df2['registration_time']<'2019-03-25'].select_dtypes(include=['number','bool']).iloc[:,:-7]

corr_test = df2[df2['registration_time']>='2019-03-25'].select_dtypes(include=['number','bool']).iloc[:,:-7]


train1_corr_m = corr_train1.corr(method='spearman')
# 히트맵 시각화
plt.figure(figsize=(11, 11))  # 그림 크기 설정
sns.heatmap(train1_corr_m, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})  # annot_kws로 글씨 크기 조정
plt.title('Correlation Matrix', fontsize=14)  # 제목 설정 (작게)
plt.xticks(rotation=45, ha='right', fontsize=10)  # x축 라벨 회전 및 크기 조정
plt.yticks(rotation=0, fontsize=10)  # y축 라벨 수평 및 크기 조정
plt.tight_layout()  # 레이아웃 최적화
plt.show()


train2_corr_m = corr_train2.corr(method='spearman')
# 히트맵 시각화
plt.figure(figsize=(11, 11))  # 그림 크기 설정
sns.heatmap(train2_corr_m, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})  # annot_kws로 글씨 크기 조정
plt.title('Correlation Matrix', fontsize=14)  # 제목 설정 (작게)
plt.xticks(rotation=45, ha='right', fontsize=10)  # x축 라벨 회전 및 크기 조정
plt.yticks(rotation=0, fontsize=10)  # y축 라벨 수평 및 크기 조정
plt.tight_layout()  # 레이아웃 최적화
plt.show()




# 중요변수 : cast_pressure, lower_mold_temp1, lower_mold_temp2, upper_mold_temp1, upper_mold_temp2, sleeve_temperature,high_section_speed, low_section_speed, biscuit_thickness

imp_df2 = df2[['cast_pressure', 'lower_mold_temp1', 'lower_mold_temp2', 'upper_mold_temp1', 'upper_mold_temp2', 'sleeve_temperature','high_section_speed', 'low_section_speed', 'biscuit_thickness']]
imp_df2_corr_m = imp_df2.corr(method='spearman')
# 히트맵 시각화
plt.figure(figsize=(7, 8))  # 그림 크기 설정
sns.heatmap(imp_df2_corr_m, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})  # annot_kws로 글씨 크기 조정
plt.title('Correlation Matrix', fontsize=14)  # 제목 설정 (작게)
plt.xticks(rotation=45, ha='right', fontsize=10)  # x축 라벨 회전 및 크기 조정
plt.yticks(rotation=0, fontsize=10)  # y축 라벨 수평 및 크기 조정
plt.tight_layout()  # 레이아웃 최적화
plt.show()












info = dict( df = old , df2 =new2, col = 'lower_mold_temp2', palette = 'dark', alpha=1.0)   # 얘만 바꾸면 됨.
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.histplot(data = info['df'] , x=info['col'], stat='density', palette=info['palette'], alpha = info['alpha'], bins=10)
sns.histplot(data = info['df2'] , x=info['col'], stat='density', palette=info['palette'], alpha = info['alpha'], bins=10)

plt.title(f'{info['col']}의 히스토그램 분포', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌




info = dict(df=old, df2=new2, col='lower_mold_temp2', palette='dark', alpha=0.5)
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터의 최소값과 최대값을 계산
min_value = min(old[info['col']].min(), new2[info['col']].min())
max_value = max(old[info['col']].max(), new2[info['col']].max())

# 동일한 구간을 설정
bins = np.linspace(min_value, max_value, 31)  # 10개 구간으로 설정

# 첫 번째 데이터프레임의 히스토그램
sns.histplot(data=info['df'], x=info['col'], stat='density', bins=bins, color='blue', alpha=info['alpha'], label='Old')

# 두 번째 데이터프레임의 히스토그램
sns.histplot(data=info['df2'], x=info['col'], stat='density', bins=bins, color='orange', alpha=info['alpha'], label='New')

plt.title(f"{info['col']}의 히스토그램 분포", fontsize=20)
plt.legend()  # 범례 추가
plt.tight_layout()  # plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌






info = dict(df=old[old['upper_mold_temp2']<=500], df2=new2[new2['upper_mold_temp2']<=500], col='upper_mold_temp2', palette='dark', alpha=0.5)
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터의 최소값과 최대값을 계산
min_value = min(info['df'][info['col']].min(), info['df2'][info['col']].min())
max_value = max(info['df'][info['col']].max(), info['df2'][info['col']].max())

# 동일한 구간을 설정
bins = np.linspace(min_value, max_value, 31)  # 10개 구간으로 설정

# 첫 번째 데이터프레임의 히스토그램
sns.histplot(data=info['df'], x=info['col'], stat='density', bins=bins, color='blue', alpha=info['alpha'], label='Old')

# 두 번째 데이터프레임의 히스토그램
sns.histplot(data=info['df2'], x=info['col'], stat='density', bins=bins, color='orange', alpha=info['alpha'], label='New')

plt.title(f"{info['col']}의 히스토그램 분포", fontsize=20)
plt.legend()  # 범례 추가
plt.tight_layout()  # plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌





info = dict(df=old, df2=new2, col='cast_pressure', palette='dark', alpha=0.5)
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터의 최소값과 최대값을 계산
min_value = min(info['df'][info['col']].min(), info['df2'][info['col']].min())
max_value = max(info['df'][info['col']].max(), info['df2'][info['col']].max())

# 동일한 구간을 설정
bins = np.linspace(min_value, max_value, 31)  # 10개 구간으로 설정

# 첫 번째 데이터프레임의 히스토그램
sns.histplot(data=info['df'], x=info['col'], stat='density', bins=bins, color='blue', alpha=info['alpha'], label='Old')

# 두 번째 데이터프레임의 히스토그램
sns.histplot(data=info['df2'], x=info['col'], stat='density', bins=bins, color='orange', alpha=info['alpha'], label='New')

plt.title(f"{info['col']}의 히스토그램 분포", fontsize=20)
plt.legend()  # 범례 추가
plt.tight_layout()  # plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌




new3 = new2[new2['upper_mold_temp2']<=500]



for index, i in enumerate(old.columns):
    try:
        psi = calculate_psi(old, new2, i)
        #print(f"PSI for {i}컬럼: {psi:.4f}")
        psi_df.loc[index ,'old_new3_psi'] = psi
    except Exception as e:
        #print(f"Error calculating PSI for {i}컬럼: {e}")
        psi_df.loc[index ,'old_new3_psi'] = np.nan


for i in old.select_dtypes(include=['bool','boolean']).columns:
    psi = calculate_psi_boolean(old, new2, i)
    # print(f"PSI for {i}컬럼: {psi:.4f}")
    psi_df.loc[psi_df['columns'] == i, 'old_new3_psi'] = psi
    

psi_df[psi_df['old_new3_psi']>0.1].sort_values('old_new3_psi', ascending=False)














plt.figure(figsize=(5*2, 4*2))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(2, 2, 1)
#sns.violinplot(data=old, x='passorfail', y='low_section_speed', inner='quartile', palette='dark', alpha=0.5)
sns.boxenplot(data=old, x='passorfail', y='low_section_speed')
#plt.ylim([0,400])
plt.title(f'이전 low_section_speed', fontsize=20)

plt.subplot(2, 2, 2)
#sns.violinplot(data=new2, x='passorfail', y='low_section_speed', inner='quartile', palette='dark', alpha=0.5)
sns.boxenplot(data=new2, x='passorfail', y='low_section_speed')
#plt.ylim([0,400])
plt.title(f'이후 low_section_speed', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌




