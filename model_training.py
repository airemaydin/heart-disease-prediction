import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. Veriyi Yükleme
data = pd.read_csv('heart.csv')

# Veri Seti Hakkında Temel Bilgiler
print(f"Özellik sayısı: {data.shape[1]}")
print(f"Gözlem sayısı: {data.shape[0]}")
print("Eksik veri durumu:\n", data.isnull().sum())

# 2. Keşifsel Veri Analizi (EDA) Görselleştirmeleri
# Not: Çok fazla grafik çıkıp ekranı dondurmaması için, dilerseniz bu blokları yoruma alabilirsiniz.
for col in data.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(8,6))
    sns.countplot(x=col, data=data)
    plt.title(f'{col} Dağılımı')
    plt.show()

for col in data.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(8,6))
    sns.countplot(x=col, hue='HeartDisease', data=data)
    plt.title(f'{col} ve Kalp Hastalığı Dağılımı')
    plt.show()

# 3. Veri Temizleme ve Ön İşleme
# Sıfır olan mantıksız değerleri silmek yerine medyan (ortanca) ile doldurma
data['RestingBP'] = data['RestingBP'].replace(0, data['RestingBP'].median())
data['Cholesterol'] = data['Cholesterol'].replace(0, data['Cholesterol'].median())

# Kategorik (metin) verileri sayısal verilere dönüştürme (Encoding)
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['ChestPainType'] = le.fit_transform(data['ChestPainType'])

# Kullanılacak özelliklerin (features) ve hedefin (target) seçimi
features = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS']
X = data[features]
y = data['HeartDisease']

# 4. Veri Sızıntısını (Data Leakage) Önlemek İçin Önce Veriyi Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bölme İşleminden Sonra Ölçeklendirme (Scaling) Yapma
scaler = MinMaxScaler()
# Scaler sadece eğitim verisinin dağılımını öğrenir (fit) ve dönüştürür (transform)
X_train_scaled = scaler.fit_transform(X_train)
# Test verisi kopya çekmemesi için sadece dönüştürülür (transform)
X_test_scaled = scaler.transform(X_test) 

# 5. Model Eğitimi (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
print(f"\nStandart Model doğruluğu: {model.score(X_test_scaled, y_test):.4f}")

# 6. Hiperparametre Optimizasyonu (Grid Search)
param_grid = {'n_neighbors': [3, 5, 7, 9], 'metric': ['minkowski', 'euclidean']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print(f"GridSearch En iyi doğruluk oranı: {grid_search.best_score_:.4f}")
print(f"GridSearch En iyi parametreler: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
print(f"Optimize Edilmiş Test setindeki doğruluk: {best_model.score(X_test_scaled, y_test):.4f}")
