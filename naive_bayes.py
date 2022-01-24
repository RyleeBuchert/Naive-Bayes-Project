from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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
        classification_prob = {}
        for line in input:
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

        return total_correct / total_instances

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
        classification_prob = {}
        for line in input:
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
        
        return total_correct / total_instances

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

    # create training set and separate X and Y columns
    training_Y = review_data['Label']
    training_X = review_data['Review']
    X_train, X_test, Y_train, Y_test = train_test_split(training_X, training_Y, test_size = 0.2)

    # get count df for X column
    X_train_bernoulli = create_count_df(X_train, "Bernoulli")
    X_train_multinomial = create_count_df(X_train, "Multinomial")

    # fit models and get results
    NB = BernoulliNB()
    NB.fit(X_train_bernoulli, Y_train)
    print("Text Bernoulli Accuracy:", NB.predict(X_test, Y_test))

    NB = MultinomialNB()
    NB.fit(X_train_multinomial, Y_train, 1)
    print("Text Multinomial Accuracy:", NB.predict(X_test, Y_test))


    # obtain digit image data and generate training/test sets
    digit_data = pd.read_csv('C:\\Visual_Studio\\CS_5333\\Project 1\\digit_data.csv')
    training_Y = digit_data['label']
    training_X = digit_data.drop(digit_data.columns[0], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(training_X, training_Y, test_size = 0.2)

    # fit models and get results
    NB = BernoulliNB()
    NB.fit(X_train, Y_train)
    print("Digit Bernoulli Accuracy:", NB.predict(X_test, Y_test))

    NB = MultinomialNB()
    NB.fit(X_train, Y_train, 1)
    print("Digit Multinomial Accuracy:", NB.predict(X_test, Y_test))