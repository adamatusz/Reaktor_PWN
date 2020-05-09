import pandas as pd
import seaborn
import sklearn
from numpy import nan
from pandas import DataFrame, pivot_table, get_dummies
from seaborn import load_dataset
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class MLWarmup:

    def homework(self):
        self.titanic = seaborn.load_dataset("titanic")
        print(self.titanic)
    # 1. Oczyszczenie danych i podział na zbiór testowy i treningowy ???
    # 2. Trenowanie na podstawie kNN(???)
    # 3. Testowanie
    # 4. Ocena klasyfikacji i testowania

    #       print(self.titanic.describe())
        print(self.titanic.describe(include=[pd.np.number]))
        print(self.titanic.describe(include=[pd.np.object]))
        print(self.titanic.describe(include=['category']))
        print(self.titanic.describe(include={'boolean'}))
    #   print(self.titanic.info(null_counts=True))
        print(self.titanic.shape)

        male_famele = self.titanic.sex.value_counts()
        survived_2 = self.titanic.survived.value_counts()
        embarked = self.titanic.embark_town.value_counts()
        alone_2 = self.titanic.alone.value_counts()
        who_2 = self.titanic.who.value_counts()
        adult_male_2 = self.titanic.adult_male.value_counts()
        sibsp_2 = self.titanic.sibsp.value_counts()
        parch_2 = self.titanic.parch.value_counts()
        print(male_famele)  # mężczyżni vs. kobiety
        print(survived_2)  # liczba uratowanych
        print(embarked)  # porty pochodzenia
        print(alone_2)  # samotni vs. reszta
        print(who_2)  # płeć oraz dzieci
        print(adult_male_2)  # dorośli mężczyźni i reszta(kobiety+dzieci)
        print(sibsp_2)  # rodziny z liczbą rodzeństwa
        print(parch_2)  # rodziny z liczbą dzieci

        print(self.titanic['age'].describe())

        print(self.titanic.isnull().sum())  # deck, embarked pozniej usunę, dla wieku NaN zastapie mediana,
        # embark_town Southampton najbardziej prawdopodobny
        self.titanic['age'] = self.titanic['age'].fillna(self.titanic['age'].median())
        self.titanic['embark_town'] = self.titanic['embark_town'].fillna('Southampton')

        print(self.titanic.isnull().sum())

        # zamiana zmiennych jakościowych na numeryczne
        self.titanic_2 = self.titanic
        self.titanic_2 = get_dummies(self.titanic_2, columns=['embark_town', 'class', 'who', 'alone',
                                                          'sibsp', 'parch'], drop_first=True)
        print(self.titanic_2.info(null_counts=True))
        # usuwanie kolumn
        self.titanic_2.drop(self.titanic_2.columns[[1, 2, 4, 5, 6, 7, 8]], axis=1, inplace=True)
        print(self.titanic_2.info(null_counts=True))

        print(self.titanic_2.isnull().sum())

        # tworze 2 zbiory testowe . Jakko, że chce sprawdzic szanse przezycia pozbywam sie
        # "survived"
        titanic_train = self.titanic_2.filter(["age", "embark_town_Queenstown", "embark_town_Southampton",
                                                 "class_Second", "class_Third", "who_man", "who_woman",
                                                 "alone_True", "sibsp_1", "sibsp_2", "sibsp_3",
                                                 "sibsp_4", "sibsp_5", "sibsp_8", "parch_1", "parch_1",
                                                 "parch_1", "parch_1", "parch_1", "parch_1"], axis=1)
        X = titanic_train
        print(X)

        titanic_test = self.titanic_2["survived"]
        y = titanic_test
        print(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # metoda najbliszych sasiadów

        knn = KNeighborsClassifier()
        k_range = list(range(1, 10))
        weights_options = ['distance', 'uniform']
        k_grid = dict(n_neighbors=k_range, weights=weights_options)
        grid = sklearn.model_selection.GridSearchCV(knn, k_grid, cv=10, scoring='precision')
        print(grid.fit(X_train, y_train))
        print(grid.cv_results_)
        print("BEST SCORES", str(grid.best_score_))
        print("BEST PARAMETERS: ", str(grid.best_params_))
        print("BEST ESTIMATOR: ", str(grid.best_params_))

        # WYNIK PREDYKCJI
        label_pred = grid.predict(X_test)
        acc_clf = sklearn.metrics.accuracy_score(y_test, label_pred)
        print("classifier's accuracy: ", str(acc_clf))

        # obliczanie dokładności modelu


ml = MLWarmup()
#ml.getIrisDataset()
#ml.splitDataset()
#ml.trainModel()
#ml.testModel()
ml.homework()
# ml.getPlanetsDataset()