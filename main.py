import pandas as pd
import matplotlib.pyplot as plt
import automate_the_model_selection as auto
from implements_the_modeling import *
from sklearn.ensemble import RandomForestClassifier
from form_coalitions import *
from prepare_the_data import *
from sklearn.linear_model import Perceptron

def main():
    # Transform the raw data into something usable
    prepare_data()
    # Split into sets
    x_test, x_train, x_validation, x_pred, y_test, y_train, y_validation, y_pred = load_prepared_data()

    # In the previous homeworks we determined that the following model gives the best results:
    best_model_clf =  RandomForestClassifier(n_jobs=-1, random_state=2, criterion='entropy')
    x_train_and_validation = x_train.append(x_validation).reset_index(drop=True)
    y_train_and_validation = y_train.append(y_validation).reset_index(drop=True)
    best_model_clf.fit(x_train_and_validation, y_train_and_validation)

    # Task 1: Predict which party would win the majority of votes    
    winner_party(best_model_clf, x_test)
    
    # Task 2: Predict the division of votes between the various parties
    res = best_model_clf.predict(x_test)
    vote_division(res, y_test)

    calculate_overall_test_error(res, y_test)

    # Task 3: Predict the vote of each voter in the new sample
    y_new_test_pred = best_model_clf.predict(x_pred)
    y_new_test_pred = [from_num_to_label[x] for x in y_new_test_pred]
    res = pd.DataFrame({'IdentityCard_Num': y_pred.values, 'PredictVote': y_new_test_pred})
    res.to_csv('new_test_voting_predictions.csv', index=False)

    # Task 4: What will be a steady coalition
    get_clustering_coalition(x_test, x_train, x_validation, y_test, y_train, y_validation)


if __name__ == '__main__':
    main()
