from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# multivariate bernoulli event model algorithm
class BernoulliNB:

    # class constructor
    def __init__(self):
        self.classes = None
    
    # method to fit model to training set
    def fit(self, X, Y):

        # isolate classes of label column
        self.classes = np.unique(Y)
        self.num_classes = len(self.classes)

        # get list and number of features
        self.vocabulary = list(X.columns)
        self.num_vocabulary = len(X.columns)

        # get class priors
        class_counts = Y.value_counts(normalize=True)
        self.priors = {}
        for i in self.classes:
            self.priors.update({i: np.log(class_counts.values[i])})

        # get number of class instances
        combined_df = pd.concat([Y, X], axis=1)
        class_df_dict = {}
        for i in self.classes:
            class_df_dict.update({i: combined_df.loc[combined_df[Y.name] == i]})

        self.num_class_instances = {}
        for i in self.classes:
            self.num_class_instances.update({i: len(class_df_dict[i])})
        
        # get conditional probabilities for all features
        self.class_word_prob_dict = {}
        for i in self.classes:
            temp_word_dict = {word: 0 for word in self.vocabulary}
            self.class_word_prob_dict.update({i: temp_word_dict})
        
        for i in self.classes:
            temp_df = class_df_dict[i]
            for word in self.vocabulary:
                num_word_instance_given_class = temp_df[word].sum()
                prob_word_instance_given_class = (num_word_instance_given_class + 1) / (self.num_class_instances[i] + self.num_classes)
                self.class_word_prob_dict[i][word] = prob_word_instance_given_class

    # method to predict model accuracy based on test set
    def predict(self, input, key):

        # convert one-hot encoded df to instance list
        if input._typ == 'dataframe':
            input_list = input.values.tolist()
            input_name_list = []
            for i, j in enumerate(input_list):
                col_list = [x for x in range(len(j)) if j[x] == 1]
                input_name_list.append([input.columns[y] for y in col_list])
            input = input_name_list

        # get predictions for test set
        prediction_list = [] 
        for line in input:
            classification_prob = {}
            for i in self.classes:
                sum_log_word_prob = 0
                for word in line:
                    if word in self.class_word_prob_dict[i]:
                        sum_log_word_prob += np.log(self.class_word_prob_dict[i][word])
                classification_prob.update({i: self.priors[i] + sum_log_word_prob})
            prediction_list.append(max(classification_prob, key=classification_prob.get))

        # calculate model accuracy
        total_correct = 0
        total_instances = len(key)
        for i in range(len(prediction_list)):
            if prediction_list[i] == key.iloc[i]:
                total_correct += 1

        # returns 2 element list containing model accuracy (%) and a list of predictions
        return [(total_correct/total_instances), prediction_list]


# multinomial event model algorithm
class MultinomialNB:

    # class constructor
    def __init__(self):
        self.classes = None

    # method to fit model to training set
    def fit(self, X, Y, alpha):
        
        # isolate classes of label column
        self.classes = np.unique(Y)
        self.num_classes = len(self.classes)

        # get list and number of features
        self.vocabulary = list(X.columns)
        self.num_vocabulary = len(X.columns)

        # get class priors
        class_counts = Y.value_counts(normalize=True)
        self.priors = {}
        for i in self.classes:
            self.priors.update({i: np.log(class_counts.values[i])})

        # get total number of words in each class
        combined_df = pd.concat([Y, X], axis=1)
        class_df_dict = {}
        for i in self.classes:
            class_df_dict.update({i: combined_df.loc[combined_df[Y.name] == i]})
        
        self.num_class_words = {}
        for i in self.classes:
            temp_df = class_df_dict[i].drop(class_df_dict[i].columns[0], axis=1)
            self.num_class_words.update({i: temp_df.to_numpy().sum()})

        # get conditional probabilities for all features
        self.class_word_prob_dict = {}
        for i in self.classes:
            temp_word_dict = {word: 0 for word in self.vocabulary}
            self.class_word_prob_dict.update({i: temp_word_dict})
        
        for i in self.classes:
            temp_df = class_df_dict[i]
            for word in self.vocabulary:
                num_word_given_class = temp_df[word].sum()
                prob_word_given_class = (num_word_given_class + alpha) / (self.num_class_words[i] + alpha*self.num_vocabulary)
                self.class_word_prob_dict[i][word] = prob_word_given_class

    # method to predict model accuracy based on test set
    def predict(self, input, key):

        # convert one-hot encoded df to instance list
        if input._typ == 'dataframe':
            input_list = input.values.tolist()
            input_name_list = []
            for i, j in enumerate(input_list):
                col_list = [x for x in range(len(j)) if j[x] == 1]
                input_name_list.append([input.columns[y] for y in col_list])
            input = input_name_list

        # get predictions for test set
        prediction_list = []
        for line in input:
            classification_prob = {}
            for i in self.classes:
                sum_log_word_prob = 0
                for word in line:
                    if word in self.class_word_prob_dict[i]:
                        sum_log_word_prob += np.log(self.class_word_prob_dict[i][word])
                classification_prob.update({i: self.priors[i] + sum_log_word_prob})
            prediction_list.append(max(classification_prob, key=classification_prob.get)) 

        # calculate model accuracy
        total_correct = 0
        total_instances = len(key)
        for i in range(len(prediction_list)):
            if prediction_list[i] == key.iloc[i]:
                total_correct += 1

        # returns 2 element list containing model accuracy (%) and a list of predictions
        return [(total_correct/total_instances), prediction_list]

# method to create dataframe of word counts
def create_count_df(input_dataset, model_type):

    # get list of all unique words
    index_list = input_dataset._stat_axis.values
    word_list = []
    for input in input_dataset:
        for word in input:
            word_list.append(word)
    word_list = list(set(word_list))

    # create dataframe of word counts (both bernoulli and multinomial)
    word_count_dict = {}
    for i in range(len(word_list)):
        word_count_dict.update({word_list[i]: [0]*len(input_dataset)})

    input_dataset.reset_index()
    for index, line in enumerate(input_dataset):
        for word in line:
            if model_type == "Bernoulli":
                if word_count_dict[word][index] == 0:
                    word_count_dict[word][index] += 1
            elif model_type == "Multinomial":
                word_count_dict[word][index] += 1
    output_df = pd.DataFrame(word_count_dict)
    output_df = output_df.set_index(index_list)

    return output_df

# method to cross validate model training for hyperparameter tuning
def cross_validation(model, X_train_set, Y_train_set, param_grid, alpha):
    full_data = pd.concat([Y_train_set, X_train_set], axis=1)
    index_list = full_data.index.tolist()
    remaining_indexes = index_list
    data_dict = {}
    for i in range(alpha):
        test_indexes = random.sample(remaining_indexes, int(len(full_data)/alpha))
        train_indexes = set(index_list)-set(test_indexes)        
        data_dict.update({i: {'train': full_data[full_data.index.isin(train_indexes)], 'test': full_data[full_data.index.isin(test_indexes)]}})
        remaining_indexes = list(set(remaining_indexes)-set(test_indexes))

    results_dict = {}
    for key in data_dict:
        temp_X_train = data_dict[key]['train'].drop(data_dict[key]['train'].columns[0], axis=1)
        temp_Y_train = data_dict[key]['train'][Y_train_set.name]
        temp_X_test = data_dict[key]['test'].drop(data_dict[key]['test'].columns[0], axis=1)
        temp_Y_test = data_dict[key]['test'][Y_train_set.name]

        temp_model = model
        temp_model.fit(temp_X_train, temp_Y_train, param_grid['alpha'][key])
        results_dict.update({param_grid['alpha'][key]: temp_model.predict(temp_X_test, temp_Y_test)})
    
    return max(results_dict, key=results_dict.get)

# method to obtain confusion matrix from predictions and key
def get_confusion_matrix(predictions, key):
    unique_classes = list(key.unique())
    unique_classes.sort()
    plot_dict = {}
    for i in unique_classes:
        plot_dict.update({i: {'Correct': 0, 'Incorrect': 0}})
    for i in range(len(predictions)):
        for j in unique_classes:
            if key.iloc[i] == j and predictions[i] == j:
                plot_dict[j]['Correct'] += 1
            elif key.iloc[i] == j and predictions[i] != j:
                plot_dict[j]['Incorrect'] += 1

    plot_list = []
    for i in unique_classes:
        plot_list.append(plot_dict[i]['Correct'])
        plot_list.append(plot_dict[i]['Incorrect'])
    plot_list = np.array([[plot_list[0], plot_list[1]],[plot_list[3], plot_list[2]]])
    fig, ax = plot_confusion_matrix(conf_mat=plot_list, show_absolute=True, show_normed=True, colorbar=True)
    plt.show()

# method to obtain confusion matrix values for multiple predictions
def get_confusion_values(predictions, key):
    unique_classes = list(key.unique())
    unique_classes.sort()
    plot_dict = {}
    for i in unique_classes:
        plot_dict.update({i: {'Correct': 0, 'Incorrect': 0}})
    for i in range(len(predictions)):
        for j in unique_classes:
            if key.iloc[i] == j and predictions[i] == j:
                plot_dict[j]['Correct'] += 1
            elif key.iloc[i] == j and predictions[i] != j:
                plot_dict[j]['Incorrect'] += 1

    plot_list = []
    for i in unique_classes:
        plot_list.append(plot_dict[i]['Correct'])
        plot_list.append(plot_dict[i]['Incorrect'])
    
    return plot_list

# method to plot average confusion matrix
def plot_avg_confusion_matrix(conf_list):
    plot_list = np.array([[int(conf_list[0]), int(conf_list[1])],[int(conf_list[3]), int(conf_list[2])]])
    fig, ax = plot_confusion_matrix(conf_mat=plot_list, show_absolute=True, show_normed=True, colorbar=True)
    plt.show()

# method to test model over multiple iterations
def iterative_test(dataset, X_data_train, Y_data_train, model, iterative_range, cv=None, alpha=None):

    if cv is not None and alpha is not None:
        X_train, X_test, Y_train, Y_test = train_test_split(X_data_train, Y_data_train, test_size=0.2)
        best_alpha = cross_validation(MultinomialNB(), X_train, Y_train, cv, alpha)

    result_list = []
    for i in range(iterative_range):
        X_train, X_test, Y_train, Y_test = train_test_split(X_data_train, Y_data_train, test_size=0.2)
        if model == "Bernoulli":
            temp_NB = BernoulliNB()
            temp_NB.fit(X_train, Y_train)
        elif model == "Multinomial":
            temp_NB = MultinomialNB()
            temp_NB.fit(X_train, Y_train, best_alpha)
        temp_result_list = temp_NB.predict(X_test, Y_test)
        result_list.append(temp_result_list[0])
        print(i)

    if model == "Bernoulli":
        with open(f"C:\\Visual_Studio\\CS_5333\\Project 1\\bernoulli_{dataset}_results.txt", "w") as output:
            for item in result_list:
                output.write(str(item)+'\n')
    elif model == "Multinomial":
        with open(f"C:\\Visual_Studio\\CS_5333\\Project 1\\multinomial_{dataset}_results.txt", "w") as output:
            output.write("Best Alpha: "+ str(best_alpha) + '\n')
            for item in result_list:
                output.write(str(item)+'\n')


if __name__ == "__main__":

    # obtain review data and place into dataframes
    amazon_data = pd.read_csv('C:\\Visual_Studio\\CS_5333\\Project 1\\amazon_reviews.txt', sep='\t', names=['Review', 'Label'])
    yelp_data = pd.read_csv('C:\\Visual_Studio\\CS_5333\\Project 1\\yelp_reviews.txt', sep='\t', names=['Review', 'Label'])
    imdb_data = pd.read_csv('C:\\Visual_Studio\\CS_5333\\Project 1\\imdb_reviews.txt', sep='\t', names=['Review', 'Label'])

    # combine data and clean dataframe 
    review_data = amazon_data.append(yelp_data)
    review_data = review_data.append(imdb_data)
    review_data = review_data[review_data.columns[[-1, 0]]]
    review_data['Review'] = review_data['Review'].str.replace('\W', ' ')
    review_data['Review'] = review_data['Review'].str.lower()
    review_data['Review'] = review_data['Review'].str.split()
    review_data = review_data.reset_index(drop=True)

    # create training set and separate X and Y columns
    training_X = review_data['Review']
    training_Y = review_data['Label']

    # get count df for X column
    X_train_bernoulli = create_count_df(training_X, "Bernoulli")
    X_train_multinomial = create_count_df(training_X, "Multinomial")

    # fit bernoulli model and get accuracy/confusion matrix (iterative test)
    iterative_test("text", X_train_bernoulli, training_Y, "Bernoulli", 100)

    # fit bernoulli model and get accuracy/confusion matrix (single test)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_bernoulli, training_Y, test_size=0.2)
    NB = BernoulliNB()
    NB.fit(X_train_bernoulli, Y_train)
    bernoulli_results_list = NB.predict(X_test, Y_test)
    print("Text Bernoulli Accuracy:", bernoulli_results_list[0])
    get_confusion_matrix(bernoulli_results_list[1], Y_test)

    # fit multinomial model and get accuracy/confusion matrix (iterative test)
    alpha_params = {'alpha': [0.1, 0.5, 1.0, 2.0, 3.0]}
    iterative_test("text", X_train_multinomial, training_Y, "Multinomial", 100, alpha_params, 5)

    # fit multinomial model and get accuracy/confusion matrix (single test)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_multinomial, training_Y, test_size=0.2)
    alpha_params = {'alpha': [0.1, 0.5, 1.0, 2.0, 3.0]}
    best_alpha = cross_validation(MultinomialNB(), X_train_multinomial, Y_train, alpha_params, 8)
    print('Best Alpha:', best_alpha)

    NB = MultinomialNB()
    NB.fit(X_train_multinomial, Y_train, 1)
    multinomial_results_list = NB.predict(X_test, Y_test)
    print("Text Multinomial Accuracy:", multinomial_results_list[0])
    get_confusion_matrix(multinomial_results_list[1], Y_test)


    # obtain digit image data and generate training/test sets
    digit_data = pd.read_csv('C:\\Visual_Studio\\CS_5333\\Project 1\\digit_data.csv')
    training_Y = digit_data['label']
    training_X = digit_data.drop(digit_data.columns[0], axis=1)
    
    # fit Bernoulli models and get results (iterative test)
    iterative_test("digit", training_X, training_Y, "Bernoulli", 100)

    # fit Bernoulli models and get results (single test)
    X_train, X_test, Y_train, Y_test = train_test_split(training_X, training_Y, test_size = 0.2)
    NB = BernoulliNB()
    NB.fit(X_train, Y_train)
    bernoulli_results_list = NB.predict(X_test, Y_test)
    print("Digit Bernoulli Accuracy:", bernoulli_results_list[0])

    # fit multinomial model and get results (iterative test)
    alpha_params = {'alpha': [0.1, 0.5, 1.0, 2.0, 3.0]}
    iterative_test("digit", training_X, training_Y, "Multinomial", 100, alpha_params, 5)

    # fit multinomial model and get results (single test)
    X_train, X_test, Y_train, Y_test = train_test_split(training_X, training_Y, test_size = 0.2)
    alpha_params = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]}
    best_alpha = cross_validation(MultinomialNB(), X_train, Y_train, alpha_params, 8)
    print('Best Alpha:', best_alpha)

    NB = MultinomialNB()
    NB.fit(X_train, Y_train, best_alpha)
    multinomial_results_list = NB.predict(X_test, Y_test)
    print("Digit Multinomial Accuracy:", multinomial_results_list[0])
