# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression(random_state=0, solver='lbfgs',  multi_class='multinomial') 
# hogwarts_df = load_processed_data_multi()

# # Оставляем только нужные колонки
# data_full = hogwarts_df.drop(
#     [
#     'name', 
#     'surname',
#     ], 
#     axis=1).copy()
# X_data = data_full.drop('faculty', axis=1)
# y = data_full.faculty

# clf.fit(X_data, y)
# score_testing_dataset(clf)
                                                
# указанное выше названо результатами мультиномиальной регрессии при этом метода load_processed_data_multi() нет и fuculty еолнки тоже нет

# --------------------------------------------------------------------------------------------------------

# from model_training import train_classifiers
# from data_loaders import load_processed_data
# import warnings
# warnings.filterwarnings('ignore')

# # Загружаем данные
# hogwarts_df = load_processed_data()

# # Оставляем только нужные колонки
# data_full = hogwarts_df.drop(
#     [
#     'name', 
#     'surname',
#     'is_griffindor',
#     'is_hufflpuff',
#     'is_ravenclaw'
#     ], 
#     axis=1).copy()
# X_data = data_full.drop('is_slitherin', axis=1)
# y = data_full.is_slitherin

# # Проводим исследование моделей, выбрана random forests оптимизированная модель
# slitherin_models = train_classifiers(data_full, X_data, y)
# score_testing_dataset(slitherin_models[5])

# --------------------------------------------------------------------------------------------------------

