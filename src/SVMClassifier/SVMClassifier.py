import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class SVMClassifier:

    def __init__(self):
        self.trained = False

    def train(self, data, target, C=1, gamma='scale', grid_search=True):
        """
        Train the model
        parameters:
            data: data, not normalized!
            target: target
            C: C parameter for SVM (default: 1)
            gamma: gamma parameter for SVM (default: 'scale')
            grid_search: whether to perform grid search and use the best parameters (default: True)
        """
        #data are not meant to be normalized
        #normalize data
        data = data / 255

        # if grid_search, choose the best parameters
        if grid_search:
            parameters = {'C': [0.1, 1, 10, 100],'gamma':[0.001, 0.01, 0.1,1]}
            model = SVC(kernel = 'rbf')
            clf = GridSearchCV(model, parameters, scoring='accuracy')
            clf.fit(data, target)
            self.model = SVC(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
        else:
            self.model = SVC(kernel='rbf', C=C, gamma=gamma)
        self.model.fit(data, target)
        self.trained = True
        self.save('SVMmodel.pkl')

    def predict(self, data):
        """
        Predict the labels of the data
        parameters:
            data: data, not normalized!
        """
        if not self.trained: raise Exception("Model is not trained!")
        #normalize data
        data = data / 255
        return self.model.predict(data)


    def evaluate(self, data, target):
        """
        Evaluate the model
        parameters:
            data: data, not normalized!
            target: target
        """
        if not self.trained: raise Exception("Model is not trained!")
        #normalize data
        data = data / 255
        return self.model.score(data, target)

    def evaluate_confusion_matrix(self, data, target):
        """
        Evaluate the model
        parameters:
            data: data, not normalized!
            target: target
        """
        if not self.trained: raise Exception("Model is not trained!")
        #normalize data
        data = data / 255
        return confusion_matrix(target, self.model.predict(data))
    
    def evaluate_classification_report(self, data, target):
        """
        Evaluate the model
        parameters:
            data: data, not normalized!
            target: target
        """
        if not self.trained: raise Exception("Model is not trained!")
        #normalize data
        data = data / 255
        return classification_report(target, self.model.predict(data))

    def evaluate_accuracy(self, data, target):
        """
        Evaluate the model
        parameters:
            data: data, not normalized!
            target: target
        """
        if not self.trained: raise Exception("Model is not trained!")
        #normalize data
        data = data / 255
        return accuracy_score(target, self.model.predict(data))


    def save(self, filename='SVMModel.pkl'):
        """
        Save the model to a file
        parameters:
            filename: filename
        """
        if not self.trained: raise Exception("Model is not trained!")
        import pickle
        with open(filename,'wb') as f:
            pickle.dump(self.model,f)

    def load (self, filename='SVMModel.pkl'):
        import pickle
        with open('SVMmodel.pkl', 'rb') as f:
            self.model = pickle.load(f)
        self.trained = True