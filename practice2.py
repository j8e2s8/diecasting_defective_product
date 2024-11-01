import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

# 시간 변수 추가
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second


# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

# 불필요한 컬럼 제거2  :  heating_furnace 도 제거할지 고려
df = df.drop(['upper_mold_temp3', 'lower_mold_temp3', 'count'], axis=1)  


# -------------------------------------------- 새로운 데이터 프레임.
# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# --------------------------------------------- 이상치 제거
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000)]

# ------------------------------------------- 파생변수
df2['molten_temp_g'] = np.where(df2['molten_temp']<600, True,False)
df2['cast_pressure_g'] = np.where(df2['cast_pressure'] <= 270, True,False)
df2['biscuit_thickness_g'] = np.where((df2['biscuit_thickness']>60) |(df2['biscuit_thickness']<20), True,False)
df2['physical_strength_g'] = np.where(df2['physical_strength'] < 500, True,False)
df2['low_section_speed_g'] = np.where(df2['low_section_speed'] < 75, True,False)
df2['high_section_speed_g'] = np.where((df2['high_section_speed'] >200)|(df2['high_section_speed'] < 70), True,False)


# 결측치 확인
df.isna().sum()

# ---------------------------------------------
# 종속 변수와 독립 변수 설정
X = df2.drop(columns=['passorfail'])
y = df2['passorfail']

X = pd.get_dummies(X, drop_first=True)



# ------------------------------------ 패스
for i in X.select_dtypes(include='object').columns:
	X[f'{i}_original'] = X[i]

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X, columns=X.select_dtypes(include='object').columns.drop(['working_original', \
       'mold_code_original', 'heating_furnace_original']), drop_first=True)

# ---------------------------------------

# 데이터셋 분할 (훈련셋 80%, 테스트셋 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 시계열 분석 train test 분할 (훈련셋 80%, 테스트셋 20%) 3월 15일부터 31까지 가 20%
test = X[X['registration_time'] >= '2019-03-15']
train = X[X['registration_time'] < '2019-03-15']




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

num_X_train = X_train.select_dtypes(include=('number'))

for col in num_X_train.columns:
	X_train[f'{col}_outlier'] = np.where((X_train[col]<IQR_outlier(num_X_train).loc['하한',col])|(X_train[col]>IQR_outlier(num_X_train).loc['상한',col]),True,False)
	X_test[f'{col}_outlier'] = np.where((X_test[col]<IQR_outlier(num_X_train).loc['하한',col])|(X_test[col]>IQR_outlier(num_X_train).loc['상한',col]),True,False)


X_train = X_train.drop(['hour_outlier', 'minute_outlier', 'second_outlier'], axis=1)
X_test = X_test.drop(['hour_outlier', 'minute_outlier', 'second_outlier'], axis=1)


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







import scipy.stats as stats
for col in ['biscuit_thickness_g', 'physical_strength_g', 'high_section_speed_g','working_original', 'lower_mold_temp1_outlier']:
	contingency_table = pd.crosstab(X_train[col], y_train)

	# 피셔의 정확 검정 실행
	oddsratio, p_value = stats.fisher_exact(contingency_table)

	print(f"오즈비(odds ratio): {oddsratio}")
	print(f"{col}의 p-값 : {p_value}")

	# 귀무가설 채택 여부 판단
	if p_value < 0.05:
		print(f"{col}의 귀무가설 기각: 독립변수와 타겟 변수는 연관성이 있다.", "\n","-"*30)
	else:
		print(f"{col}의 귀무가설 채택: 독립변수와 타겟 변수는 연관성이 없다.", "\n","-"*30)



# 타켓변수와 연관성이 없는 변수 : physical_strength_outlier
X_train = X_train.drop(['molten_temp_outlier', 'physical_strength_outlier', 'working_original','mold_code_original', 'heating_furnace_original'], axis=1)
X_test = X_test.drop(['molten_temp_outlier', 'physical_strength_outlier', 'working_original','mold_code_original', 'heating_furnace_original'], axis=1)





# 테스트 데이터에 대한 예측 수행 (이전 모델 학습 코드와 연계됨)

# Random Forest 모델 초기화
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]  # ROC AUC를 위해 확률값을 예측 (positive class)

# 성능 계산
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # Confusion Matrix에서 각 항목 추출
g_mean = sqrt((tp / (tp + fn)) * (tn / (tn + fp)))  # G-Mean 공식

# 결과 출력
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"G-Mean: {g_mean:.4f}")


# 혼동 행렬 계산
cm = confusion_matrix(y_test, y_pred)

# 혼동 행렬을 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['False_정상', 'True_불량'],
            yticklabels=['False_정상', 'True_불량'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.show()



# 랜덤 포레스트 모델 학습 후 중요 변수 추출
importances = rf_model.feature_importances_
feature_names = X_train.columns  # 피처 이름을 가져옵니다.

# 중요도를 데이터프레임으로 변환
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 상위 10개 중요 변수 선택
top_n = 10
top_importance_df = importance_df.head(top_n)

# 중요 변수 시각화
plt.figure(figsize=(10, 6))
plt.barh(top_importance_df['Feature'], top_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()  # 중요도에 따라 순서 정렬
plt.show()























