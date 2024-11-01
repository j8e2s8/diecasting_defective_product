import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from scipy.stats import chi2_contingency

# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project4/data_week4.csv


# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

df2 = df[df['tryshot_signal'].isna()]

df2.columns
df2 = df2.drop('tryshot_signal', axis=1)
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000) & (df2['lower_mold_temp3'] < 60000)]

df[df['lower_mold_temp3'] > 60000]

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
	

kde(df)
kde(df2)

len(df2)
group_df2 = df2.groupby('passorfail').agg(개수=('passorfail', 'count'))
group_df2['비율'] = np.round(group_df2['개수']/len(df2),3)




# 범주별로 모든 수치 컬럼의 kde subplot 그래프	
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

hue_kde(df, 'passorfail')
hue_kde(df2, 'passorfail')



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
	
box(df)
box(df2)

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

hue_box(df, 'passorfail', palette='coolwarm')
hue_box(df[df['mold_code']=='8722'], 'passorfail', palette='coolwarm')
hue_box(df2, 'passorfail', palette='coolwarm')


pd.set_option('display.max_columns', None)
df2[df2['physical_strength']>60000]


for i in df.columns:
	print(f'{i}컬럼의 unique 개수 :',len(df[i].unique()))
	

for i in df.columns:
	print(f'{i}컬럼의 결측치 개수 :',df[i].isna().sum())    
      

cols = ['line', 'name', 'mold_name', 'working', 'emergency_stop', 'EMS_operation_time', 'passorfail', 'tryshot_signal', 'mold_code', 'heating_furnace']
for i in df.columns:
	if len(df[i].unique()) <= 15:
		print(f'{i}컬럼의 unique :', df[i].unique())


pd.set_option('display.max_columns', None)
df[df['passorfail'].isna()]

df.isna().sum()




def timeline(df, date_col , y , x_time_n=None, x_time_s = 'MS' ,palette='blue'):
    col = df.select_dtypes(include=['boolean','object']).columns
    w_n = sum([len(df[i].unique()) for i in col])
    
    # n 값이 너무 커지는 것을 방지하기 위해 제한 설정 (최대 10줄로 제한)
    n = min(10, int(np.ceil(w_n/4)))
    
    u = []
    for i in col:
        for v in df[i].unique():
            u.append([i, v])

    # 적절한 figsize로 변경
    plt.clf()
    plt.figure(figsize=(18, 4*n))  # 가로 18인치, 세로는 n에 비례

    for index, col_u in enumerate(u, 1):
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 서브플롯 설정
        plt.subplot(n, 4, index)
        
        # 범주에 해당하는 데이터 필터링
        df2 = df[df[col_u[0]] == col_u[1]]
        
        # 라인 플롯 생성
        sns.lineplot(data=df2, x=date_col, y=y, color=palette)

        # x축 눈금 설정
        if x_time_n is not None:
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=x_time_n))  # 눈금 개수 조정
        else:
            ticks_term = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=x_time_s)
            plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'), rotation=45)
        
        # 제목 설정
        plt.title(f'{col_u[0]}의 {col_u[1]} 범주에 대한 확률밀도', fontsize=15)

    # 레이아웃 조정 및 출력
    plt.tight_layout()  
    plt.show()
	

def hue_timeline(df, date_col , y , hue, x_time_n=None, x_time_s = 'MS' ,palette='blue'):
    col = df.select_dtypes(include=['boolean','object']).columns
    w_n = sum([len(df[i].unique()) for i in col])
    
    # n 값이 너무 커지는 것을 방지하기 위해 제한 설정 (최대 10줄로 제한)
    n = min(10, int(np.ceil(w_n/4)))
    
    u = []
    for i in col:
        for v in df[i].unique():
            u.append([i, v])

    # 적절한 figsize로 변경
    plt.clf()
    plt.figure(figsize=(18, 4*n))  # 가로 18인치, 세로는 n에 비례

    for index, col_u in enumerate(u, 1):
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 서브플롯 설정
        plt.subplot(n, 4, index)
        
        # 범주에 해당하는 데이터 필터링
        df2 = df[df[col_u[0]] == col_u[1]]
        
        # 라인 플롯 생성
        sns.lineplot(data=df2, x=date_col, y=y, hue=hue, color=palette)

        # x축 눈금 설정
        if x_time_n is not None:
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=x_time_n))  # 눈금 개수 조정
        else:
            ticks_term = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=x_time_s)
            plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'), rotation=45)
        
        # 제목 설정
        plt.title(f'{col_u[0]}의 {col_u[1]} 범주에 대한 확률밀도', fontsize=15)

    # 레이아웃 조정 및 출력
    plt.tight_layout()  
    plt.show()



timeline(df,'registration_time','count')
timeline(df2,'registration_time','molten_temp')
timeline(df2,'registration_time','facility_operation_cycleTime')
timeline(df2,'registration_time','production_cycletime')
df.select_dtypes(include='number').columns


hue_timeline(df2, 'registration_time', 'production_cycletime', 'passorfail' )
hue_timeline(df2, 'registration_time', 'physical_strength', 'passorfail' )


df3 = df2.copy()
df3['hour'] = df3['registration_time'].dt.hour
hue_timeline(df3, 'hour', 'low_section_speed', 'passorfail' )
hue_timeline(df3, 'hour', 'high_section_speed', 'passorfail' )
hue_timeline(df3, 'hour', 'molten_volume', 'passorfail' )

len(df)
	
df.columns
df.info()
	
df.query('count<20')


df.query('mold_code == 8917').sort_values('count')


df2 = df[(df['mold_code']==8917)&(df['registration_time']>='2019-03-01')]
sns.barplot(data=df2, x='registration_time', y='count')  # barplot으로 보는게 더 눈에 잘 들어옴


sns.barplot(data=df[df['mold_code']==8917], x='registration_time', y='count')  # barplot으로 보는게 더 눈에 잘 들어옴


df.query('mold_code == 8722')[(~df['tryshot_signal'].isna())&(df['registration_time']>'2019-03-01')]['registration_time']
df.query('mold_code == 8917')[(~df['tryshot_signal'].isna())&(df['registration_time']>'2019-03-01')]['registration_time']
df.query('mold_code == 8917')[(~df['tryshot_signal'].isna())&(df['registration_time']>'2019-03-01')]['registration_time']
df.query('mold_code == 8917')[(~df['tryshot_signal'].isna())&(df['registration_time']>'2019-03-01')]['registration_time']


df.query('mold_code == 8722')[(~df['tryshot_signal'].isna())]['registration_time']
df.query('mold_code == 8412')[(~df['tryshot_signal'].isna())]['registration_time']
df.query('mold_code == 8573')[(~df['tryshot_signal'].isna())]['registration_time']  # tryshot_signal가 다 결측치임
df.query('mold_code == 8917')[(~df['tryshot_signal'].isna())]['registration_time']
df.query('mold_code == 8600')[(~df['tryshot_signal'].isna())]['registration_time']  # tryshot_signal가 다 결측치임
df.query('mold_code == 8413')[(~df['tryshot_signal'].isna())]['registration_time'] # <- 모든 데이터 날짜 범위에서 tryshot_signal이 결측치임.
df.query('mold_code == 8576')[(~df['tryshot_signal'].isna())]['registration_time'] # <- 모든 데이터 날짜 범위에서 tryshot_signal이 결측치임.

from matplotlib.dates import date2num
sns.barplot(data=df[df['mold_code']==8576], x='registration_time', y='count')  # barplot으로 보는게 더 눈에 잘 들어옴
for i in df[(df['mold_code'] == 8576)&(~df['tryshot_signal'].isna())]['registration_time']:
	plt.axvline(x=date2num(i), color='red')

highlight1 = df[df['mold_code'] == 8576][['registration_time']].isin(df[(df['mold_code'] == 8576)&(~df['tryshot_signal'].isna())]['registration_time'])
plt.fill_between(df['registration_time'], df['count'], where=highlight1, color='red', alpha=0.5, label='이상치 구간')


sns.lineplot(data=df[df['mold_code']==8576], x='registration_time', y='count')  # barplot으로 보는게 더 눈에 잘 들어옴
for i in df[(df['mold_code'] == 8576)&(~df['tryshot_signal'].isna())]['registration_time']:
	plt.axvline(x=i, color='red')
# 원하는 날짜 리스트 예시
desired_ticks = df[(df['mold_code'] == 8576)&(~df['tryshot_signal'].isna())]['registration_time'].to_list()
desired_ticks = pd.to_datetime(desired_ticks)  # 문자열을 datetime으로 변환

plt.xticks(ticks=desired_ticks, labels=desired_ticks.strftime('%Y-%m-%d'), rotation=45)



# tryshot_signal 그게 D일 때가 count가 1일때?
# D 일 때 1.0 인듯?
# 불량일 때의 분포가 다 이쁜 종모양임
# lower_mold_temp3가 특정 값만 가지나?
# molten volume 같이 여러 구간이 있는데 특정 구간에서만 불량이 나오는경우, 그룹화 변수 만들기
# lower mold temp1 같은 애를 보면 불량일 때의 분포가 넓은데, 그러면 특정한 특징 없이 lower mold temp1 상관 없이 고르게 분량이 나온다는 것인듯

df['tryshot_signal'] = df['tryshot_signal'].fillna('NAN')
df['tryshot_signal'].unique()

hue_countbar(df, 'tryshot_signal')
hue_kde(df, 'tryshot_signal')

df.columns
df[~df['tryshot_signal'].isna()]



len(df[df['mold_code'] == 8576]['registration_time'])
len(df[(df['mold_code'] == 8576)&(~df['tryshot_signal'].isna())]['registration_time'])



df[df['sleeve_temperature']>800]
len(df[df['sleeve_temperature']>800])

df['sleeve_temperature'].head(20)




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
        plt.title(f'{col}의 범주별 빈도 그래프', fontsize=20)
    plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
    plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌

countbar(df)
countbar(df2)



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
        plt.title(f'{col}의 범주별 빈도 그래프', fontsize=20)
    plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
    plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌

df['passorfail'] = df['passorfail'].astype('bool')
hue_countbar(df, 'passorfail')
df.info()

df.columns
sns.countplot(x=df['passorfail'])
hue_countbar(df2, 'passorfail')


facility_operation_cycleTime
production_cycletime




# --------------------------------이상치 파생변수
# train, test 80%, 20% split          # X, y, X_train, X_test, y_train, y_test 명 통일하기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

print("훈련 데이터 크기:", X_train.shape, y_train.shape)
print("테스트 데이터 크기:", X_test.shape, y_test.shape)


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


num_df = df.select_dtypes(include=('number'))
variable = num_df.columns

for col in variable:
	X_train[f'{col}_outlier'] = np.where((X_train[col]<IQR_outlier(num_X_train).loc[0,col])|(X_train[col]>IQR_outlier(num_X_train).loc[1,col]),True,False)
	X_test[f'{col}_outlier'] = np.where((X_test[col]<IQR_outlier(num_X_train).loc[0,col])|(X_test[col]>IQR_outlier(num_X_train).loc[1,col]),True,False)






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
    

def hue_scatter(df, hue ,palette='dark', alpha=0.5):
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
        plt.title(f'범주별 {col[0]},{col[1]}', fontsize=20)
        plt.legend()
    plt.tight_layout()
    plt.show()
    
scatter(df2)
hue_scatter(df2, 'passorfail')





# 모든 범주 컬럼에 대해서 검정
# 교차표 생성
# 카이제곱 검정
cat_X_train = X_train.select_dtypes(include=('object','bool'))
for col in cat_X_train.columns:
	contingency_table = pd.crosstab(X_train[col], y_train)
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




8722 8412 8573 8917 8600 8413 8576
df2[df2['mold_code']==8722].groupby('heating_furnace').agg(개수 = ('heating_furnace','count'))

df2[df2['mold_code']==8722]['heating_furnace'].isna().sum()

df2[df2['mold_code']==8722].groupby(['registration_time','heating_furnace']).agg(개수 = ('heating_furnace','count'))
df2[df2['mold_code']==8412].groupby(['month','day','heating_furnace']).agg(개수 = ('heating_furnace','count'))



# train 1월 2일~ 3월 24일
# train 2월 15일 ~ 3월 24일
# test 3월 25일~31일
len(df2[df2['registration_time']<='2019-03-24'])  # 데이터 수 85850
df2[df2['registration_time']<='2019-03-24']['passorfail'].sum()  # 불량 수 2073 개
len(df2[(df2['registration_time']<='2019-03-24')&(df2['registration_time']>='2019-02-15')])  # 데이터 수 45421
df2[(df2['registration_time']<='2019-03-24')&(df2['registration_time']>='2019-02-15')]['passorfail'].sum()  # 불량 수 308 개
len(df2[df2['registration_time']>='2019-03-25'])  # 데이터 수 3709 개
df2[df2['registration_time']>='2019-03-25']['passorfail'].sum()  # 불량 수 17 개

data = pd.DataFrame({'데이터 수':[85850,45421,3709]
			  , '불량 수' : [2073, 308, 17]})
data.index = ['train1 0102-0324','trian2 0215-0324', 'test 0325-0331']
data['불량 비율'] = data['불량 수']/data['데이터 수']