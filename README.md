# Machine Learning Comparative Analysis

Questo repository contiene script di esempio per l'analisi comparativa delle prestazioni di vari modelli di machine learning su dati agricoli sintetici.

## Script

1. **_Comparative_Performance_Analysis_of_Various_Machine_Learning_Models_on_Synthetic_Agricultural_Data_**
   - Descrizione: Questo script confronta le prestazioni di diversi modelli di machine learning (tra cui FLDA, QDA, Regression Logistica, Random Forest, SVM e Neural Network) su un dataset sintetico rappresentativo di dati agricoli. Include la visualizzazione delle matrici di confusione e l'analisi delle prestazioni con metriche come accuratezza, precisione, richiamo e F1-score. La run viene fatta 1 volta.
   - File: `1_Comparative_Performance_Analysis_of_Various_Machine_Learning_Models_on_Synthetic_Agricultural_Data.py`

2. **_Repeated_Evaluation_of_Machine_Learning_Models_for_Consistency_Analysis_on_Synthetic_Agricultural_Data_**
   - Descrizione: Questo script esegue una valutazione ripetuta di vari modelli di machine learning per analizzare la consistenza delle loro prestazioni. Vengono eseguiti più cicli di addestramento e valutazione per calcolare le medie e le deviazioni standard delle metriche di prestazione, nonché per analizzare la relazione tra il tempo di esecuzione e l'accuratezza. La run viene fatta 100 volte.
   - File: `2_Repeated_100_Evaluation_of_Machine_Learning_Models_for_Consistency_Analysis_on_Synthetic_Agricultural_Data.py`
   
I dataset sintetici utilizzati negli script sono generati utilizzando distribuzioni normali per rappresentare vari indici agricoli (NDVI, WSI, CCI, SRI, LFI). Le etichette di classe sono create in base a semplici logiche di soglia, e i dati mancanti sono riempiti con la media delle rispettive colonne.

Se desideri contribuire a questo repository, sentiti libero di fare fork, modifiche e pull request. Ogni contributo è benvenuto!
