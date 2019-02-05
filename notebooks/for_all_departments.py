from model_training import train_all_models

# Обучаем модели для каждого факультета
"""
    Trains models for each faculty.
    :return: list of 4 models lists with trained models
    
    get_faculty_models
    
    Searches for best model among data.
    :param hogwarts_df: pd.DataFrame
    :param target_column: string name of target column
    :return: list of models
    
    prepare_data_for_training - 
    Cleans data of extra columns, making data usable for sinle faculty training.
    :param hogwarts_df: whole df
    :param target_column: string name of target column to leave in dataset
    :return: X_train, y
    X_data, y = prepare_data_for_training(hogwarts_df, target_column)
    
    
    faculty_models = train_classifiers(X_data, y)
    return faculty_models
    
"""
slitherin_models, griffindor_models, ravenclaw_models, hufflpuff_models = train_all_models()

'''

precision, recall, f1-score and support

precision - можно интерпретировать как долю объектов, названных классификатором положительными и при этом действительно являющимися положительными
то есть отношение выдананного алгоритмом к правильному

recall - показывает, какую долю объектов положительного класса из всех объектов положительного класса нашел алгоритм
то есть сколько нашел от действительного

f1 - среднее гармоническое F-мера достигает максимума при полноте и точности, равными единице, и близка к нулю, если один из аргументов близок к нулю.

В этой статье речь пойдет о задачи бинарной классификации объектов и ее реализации в одном из наиболее 
производительных пакетов машинного обучения "R" — "XGboost" (Extreme Gradient Boosting).
В реальной жизни мы довольно часто сталкиваемся с классом задач, где объектом предсказания является номинативная 
переменная с двумя градациями, когда нам необходимо предсказать результат некого события или принять решения в бинарном 
выражении на основании модели данных. Например, если мы оцениваем ситуацию на рынке и нашей целью является принятие однозначного
решения, имеет ли смысл инвестировать в определенный инструмент в данный момент времени, купит ли покупатель исследуемый продукт
 или нет, расплатится ли заемщик по кредиту или уволится ли сотрудник из компании в ближайшее время и.т.д.


В общем случае бинарная классификация применяется для предсказания вероятности возникновения некоторого события по значениям 
множества признаков. Для этого вводится так называемая зависимая переменная (исход события), принимающая лишь одно из двух 
значений (0 или 1), и множество независимых переменных (также называемых признаками, предикторами или регрессорами).


Сразу оговорюсь, что в "R" существует несколько линейных функций для решения подобных задач, таких как "glm" из
 стандартного пакета функций, но здесь мы рассмотрим более продвинутый вариант бинарной классификации, имплементированный
  в пакете "XGboost". Эта модель, многократный победитель соревнований Kaggle, основана на построении бинарных деревьев
   решений способна поддерживать многопоточную обработку данных.  "Gradient Boosting" 
'''

from model_training import train_production_models
from model_evaluation import score_testing_dataset_full
from xgboost import XGBClassifier

best_models = []
for i in range (0,4):
    best_models.append(XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=0.7, gamma=0, learning_rate=0.05, max_delta_step=0,
           max_depth=6, min_child_weight=11, missing=-999, n_estimators=1000,
           n_jobs=1, nthread=4, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=1337, silent=1,
           subsample=0.8))

slitherin_model, griffindor_model, ravenclaw_model, hufflpuff_model = \
    train_production_models(best_models)

top_models = slitherin_model, griffindor_model, ravenclaw_model, hufflpuff_model
score_testing_dataset_full(top_models)

# сохраняем все в модели

import pickle

pickle.dump(slitherin_model, open("../output/slitherin.xgbm", "wb"))
pickle.dump(griffindor_model, open("../output/griffindor.xgbm", "wb"))
pickle.dump(ravenclaw_model, open("../output/ravenclaw.xgbm", "wb"))
pickle.dump(hufflpuff_model, open("../output/hufflpuff.xgbm", "wb"))
