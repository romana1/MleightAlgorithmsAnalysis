ffrom data_loaders import load_processed_data

hogwarts_df = load_processed_data()

hogwarts_df.head()
data_full = hogwarts_df.drop(columns=[
    'name', 
    'surname',
    'is_griffindor',
    'is_hufflpuff',
    'is_ravenclaw'
    ]).copy()

X_data = data_full.drop(columns=['is_slitherin'])
# В качестве целевой будет колонка, которая содержит 1 для учеников Слизерина
y = data_full.is_slitherin

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_evaluation import score_testing_dataset
 

# Фиксируем сид для воспроизводимоси результата, random state
seed = 7
# Пропорции разделения датасета
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=test_size, random_state=seed)

rfc = RandomForestClassifier()
rfc_model = rfc.fit(X_train, y_train)

# взяли из model_evaluation файла метод оценку тестового набора имен, котрая там зашита, она вызывает на модели,
# котрую туда передает, rfc_model сделали predict_proba(encoded_person)[0]
score_testing_dataset(rfc_model)

""" 
то есть что мы сделали?

1. загрузили данные, убрали названия строк - фамилии, оставили только да ли нет Слизерин
2. спрасили имена из каждой строки посчитали
3. обработали и сделали на модели predict 

model.predict_proba(encoded_person)[0] 

Predict will give either 0 or 1 as output
Predict_proba will give the only probability of 1.

model.predict_proba(test)[:,1]

test is the dataset i made predictions for (change it according to your dataset)
Using [:,1] in the code will give you the probabilities of getting the output as 1. 
If you replace 1 with 0 in the above code, you will only get the probabilities of getting the output as 0.

"""

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
predictions = rfc_model.predict(X_test)
print("Classification report: ")
print(classification_report(y_test, predictions))
print("Accuracy for Random Forest Model: %.2f" 
          % (accuracy_score(y_test, predictions) * 100))
print("ROC AUC from first Random Forest Model: %.2f"
             % (roc_auc_score(y_test, predictions)))

#%%
from model_training import train_classifiers
from data_loaders import load_processed_data
import warnings
warnings.filterwarnings('ignore')

# Загружаем данные
hogwarts_df = load_processed_data()

# Оставляем только нужные колонки
data_full = hogwarts_df.drop(
    [
    'name', 
    'surname',
    'is_griffindor',
    'is_hufflpuff',
    'is_ravenclaw'
    ], 
    axis=1).copy()
X_data = data_full.drop('is_slitherin', axis=1)
y = data_full.is_slitherin

# Проводим исследование моделей
"""
Trains several classifiers and reporting model quality.
:param X_data:
:param y:
:return: trained models

:return: model with default parameters and tuned one.

перед этим делаем report_quality и делаем
with help of Trains one model type, tuning hyperparameters with GridSearchCV

return svm_model, svm_grid, \
           train_model1, xgb_grid, \
           rfc_model, rfc_grid, \
           ext_model, ext_grid, \
           lgbm_model, lgbm_grid, \
           rgf_model, rgf_grid, \
           frgf_model, frgf_grid

"""
slitherin_models = train_classifiers(X_data, y)
print(slitherin_models.__sizeof__())

"""
    Scores models against pre-defined names
    :param models:
    :return: pd.DataFrame with probablities for each faculty

    сюда мы возвращаем rfc_grid предсказания для 5 срок котрые прописаны в методе в 
    модуле module_evaluation 

    Errors ---> 57     return model.predict_proba(encoded_person)[0]
    AttributeError: 'tuple' object has no attribute 'predict_proba'
"""
score_testing_dataset(slitherin_models[1])
#%%

