import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Generazione del dataset fittizio con logica per il contesto
np.random.seed(42)
n_samples = 200
NDVI = np.random.normal(loc=0.6, scale=0.1, size=n_samples)
WSI = np.random.normal(loc=0.4, scale=0.1, size=n_samples)
CCI = np.random.normal(loc=0.5, scale=0.1, size=n_samples)
SRI = np.random.normal(loc=0.3, scale=0.1, size=n_samples)
LFI = np.random.normal(loc=0.7, scale=0.1, size=n_samples)

# Creazione delle etichette di classe basate su una semplice logica di soglia
y_class = (NDVI < 0.5) | (WSI > 0.5) | (CCI < 0.4)
y_class = y_class.astype(int)

# Aggiunta di dati mancanti
X = np.vstack((NDVI, WSI, CCI, SRI, LFI)).T
X[::10] = np.nan

# Creazione del DataFrame e riempimento dei valori mancanti con la media
X_df = pd.DataFrame(X, columns=['NDVI', 'WSI', 'CCI', 'SRI', 'LFI'])
X_df.fillna(X_df.mean(), inplace=True)
X = X_df.values

# Suddivisione del dataset in training e test set
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.3, random_state=42)

# Visualizzazione del dataset
sns.pairplot(pd.DataFrame(np.hstack((X, y_class.reshape(-1, 1))), columns=['NDVI', 'WSI', 'CCI', 'SRI', 'LFI', 'Class']), hue='Class')
plt.suptitle('Scatterplot Matrix of Features Colored by Class', y=1.02)
plt.show()

# Heatmap di correlazione
plt.figure(figsize=(10, 8))
sns.heatmap(X_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Heatmap of Feature Correlations')
plt.show()

# Definizione dei modelli
models = {
    "FLDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Neural Network": MLPClassifier(max_iter=1000),
}

results = []
plt.figure(figsize=(15, 10))

# Addestramento, predizione e valutazione di ciascun modello
for i, (name, model) in enumerate(models.items()):
    start_time = time.time()
    model.fit(X_train, y_class_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    end_time = time.time()
    
    accuracy = accuracy_score(y_class_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_class_test, y_pred, average='binary')
    cm = confusion_matrix(y_class_test, y_pred)
    exec_time = end_time - start_time
    
    results.append((name, round(accuracy, 6), round(exec_time, 6), precision, recall, f1))
    
    plt.subplot(2, 3, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted', labelpad=-10)
    plt.ylabel('Actual')
    plt.subplots_adjust(hspace=0.5)
    
    # Confronto tra i dati di input e i risultati dei modelli
    initial_counts = pd.Series(y_class_train).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    print(f"\nModel: {name}")
    print(f"Initial Data Distribution (Training set):")
    print(initial_counts)
    print(f"Predicted Data Distribution (Test set):")
    print(pred_counts)
    print(f"Predicted Probabilities (first 5 samples): \n{y_prob[:5] if y_prob is not None else 'Not available'}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Execution Time: {exec_time} seconds")
    print(classification_report(y_class_test, y_pred))

plt.tight_layout()
plt.show()

# Tabella riassuntiva dei risultati
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Execution Time', 'Precision', 'Recall', 'F1-Score'])
print(results_df)

# Scatter plot per comparare tutti i modelli
plt.figure(figsize=(10, 6))
for name, accuracy, exec_time, precision, recall, f1 in results:
    plt.scatter(exec_time, accuracy, label=name)
    
plt.xlabel('Execution Time (s)')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Performance')
plt.legend()
plt.grid(True)
plt.show()
