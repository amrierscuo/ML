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
from scipy.stats import linregress

# Funzione per eseguire i modelli e raccogliere i risultati
def run_models(n_runs, models, X_train, y_class_train, X_test, y_class_test):
    results = []
    all_results = {name: [] for name in models.keys()}
    
    for _ in range(n_runs):
        for name, model in models.items():
            start_time = time.time()
            model.fit(X_train, y_class_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            end_time = time.time()
            
            accuracy = accuracy_score(y_class_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_class_test, y_pred, average='binary')
            exec_time = end_time - start_time
            
            all_results[name].append((accuracy, exec_time, precision, recall, f1))
            
            if _ == 0:
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

    return all_results

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

# Definizione dei modelli
models = {
    "FLDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Neural Network": MLPClassifier(max_iter=1000),
}

# Eseguire i modelli e raccogliere i risultati
n_runs = 100
all_results = run_models(n_runs, models, X_train, y_class_train, X_test, y_class_test)

# Calcolo della media e deviazione standard per ogni metrica
results = []
for name, values in all_results.items():
    accuracies, exec_times, precisions, recalls, f1_scores = zip(*values)
    results.append((
        name, 
        np.mean(accuracies), np.std(accuracies),
        np.mean(exec_times), np.std(exec_times),
        np.mean(precisions), np.std(precisions),
        np.mean(recalls), np.std(recalls),
        np.mean(f1_scores), np.std(f1_scores),
    ))

results_df = pd.DataFrame(results, columns=[
    'Model', 'Accuracy Mean', 'Accuracy Std', 'Execution Time Mean', 'Execution Time Std', 
    'Precision Mean', 'Precision Std', 'Recall Mean', 'Recall Std', 'F1-Score Mean', 'F1-Score Std'
])
print(results_df)

# Scatter plot per comparare tutti i modelli
plt.figure(figsize=(10, 6))
for name, accuracy_mean, accuracy_std, exec_time_mean, exec_time_std, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std in results:
    plt.errorbar(exec_time_mean, accuracy_mean, xerr=exec_time_std, yerr=accuracy_std, fmt='o', label=name)
    
# Retta dei minimi quadrati
x = results_df['Execution Time Mean']
y = results_df['Accuracy Mean']
slope, intercept, r_value, p_value, std_err = linregress(x, y)
plt.plot(x, intercept + slope * x, 'r', label='Least Squares Regression Line')
plt.xlabel('Execution Time (s)')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Performance')
plt.legend()
plt.grid(True)
plt.show()

# Analisi dell'errore
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f"P-value: {p_value}")
print(f"Standard error: {std_err}")

# Dettagli su errori di roundoff, deviazione standard e medie
for index, row in results_df.iterrows():
    print(f"Model: {row['Model']}")
    print(f"Accuracy Mean: {row['Accuracy Mean']:.6f} ± {row['Accuracy Std']:.6f}")
    print(f"Execution Time Mean: {row['Execution Time Mean']:.6f} ± {row['Execution Time Std']:.6f}")
    print(f"Precision Mean: {row['Precision Mean']:.6f} ± {row['Precision Std']:.6f}")
    print(f"Recall Mean: {row['Recall Mean']:.6f} ± {row['Recall Std']:.6f}")
    print(f"F1-Score Mean: {row['F1-Score Mean']:.6f} ± {row['F1-Score Std']:.6f}")
    print("\n")
