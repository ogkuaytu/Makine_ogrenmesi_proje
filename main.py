import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Veri seti
file_path = "/Users/oguzkaanyalcin/Desktop/veri-seti2.txt"
data = pd.read_csv(file_path, delimiter="\t")

# Sütun adlarını belirle
data.columns = ["hamilelik_sayisi", "glukoz", "kan_basinci", "cilt_kalınlığı", "insülin", "bki", "diabetes_pedigree", "yas", "sınıf"]

# Eksik değerleri 0 olanları NaN ile değiştir
cols_with_zeroes = ["glukoz", "kan_basinci", "cilt_kalınlığı", "insülin", "bki"]
data[cols_with_zeroes] = data[cols_with_zeroes].replace(0, np.nan)

# Eksik değerleri median ile doldur
imputer = SimpleImputer(strategy="median")
data[cols_with_zeroes] = imputer.fit_transform(data[cols_with_zeroes])

# Özellikleri standartlaştır
scaler = StandardScaler()
features = data.drop("sınıf", axis=1)
scaled_features = scaler.fit_transform(features)
data_scaled = pd.DataFrame(scaled_features, columns=features.columns)
data_scaled["sınıf"] = data["sınıf"]

# Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(data_scaled.drop("sınıf", axis=1), data_scaled["sınıf"], test_size=0.3, random_state=42)

# Naive Bayes sınıflandırıcıyı eğit
nb = GaussianNB()
nb.fit(X_train, y_train)

# Naive Bayes Tahminleri
y_pred_nb = nb.predict(X_test)

# Naive Bayes Değerlendirmesi
print("Naive Bayes Sonuçları:")
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
print("ROC AUC Skoru:", roc_auc_score(y_test, y_pred_nb))

# ROC Eğrisi
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes (area = %0.2f)' % roc_auc_score(y_test, y_pred_nb))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# En iyi k değerini bulmak için parametre arama
param_grid = {'n_neighbors': np.arange(1, 21)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)

# En iyi k ile KNN sınıflandırıcıyı eğitin
best_k = knn_cv.best_params_['n_neighbors']
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# KNN Tahminler
y_pred_knn = knn.predict(X_test)

# KNN Değerlendirme
print("\nKNN Sonuçları:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print("ROC AUC Skoru:", roc_auc_score(y_test, y_pred_knn))

# ROC Eğrisi
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_knn, tpr_knn, label='KNN (area = %0.2f)' % roc_auc_score(y_test, y_pred_knn))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# MLP sınıflandırıcıyı eğit (daha fazla iterasyon ve farklı solver kullanarak)
mlp = MLPClassifier(max_iter=500, solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# MLP Tahminleri
y_pred_mlp = mlp.predict(X_test)

# MLP Değerlendirmesi
print("\nMLP Sonuçları:")
print(confusion_matrix(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))
print("ROC AUC Skoru:", roc_auc_score(y_test, y_pred_mlp))

# ROC Eğrisi
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_mlp, tpr_mlp, label='MLP (area = %0.2f)' % roc_auc_score(y_test, y_pred_mlp))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# SVM sınıflandırıcıyı eğit
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

# SVM Tahminleri
y_pred_svm = svm.predict(X_test)

# SVM Değerlendirmesi
print("\nSVM Sonuçları:")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print("ROC AUC Skoru:", roc_auc_score(y_test, y_pred_svm))

# ROC Eğrisi
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_svm, tpr_svm, label='SVM (area = %0.2f)' % roc_auc_score(y_test, y_pred_svm))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
