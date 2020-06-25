from implements_the_modeling import *


def main():
    print()
    # from hw 3:
    # task 1: Predict which party would win the majority of votes
    # task 2: Predict the division of votes between the various parties
    # Load the prepared training set
    df_prepared_train = pd.read_csv("prepared_train.csv")
    # shuffle
    df_prepared_train = df_prepared_train.sample(frac=1).reset_index(drop=True)
    x_train = df_prepared_train.drop("Vote", 1)
    y_train = df_prepared_train["Vote"]

    # Load the prepared validation set
    df_prepared_validation = pd.read_csv("prepared_validation.csv")
    # shuffle
    df_prepared_validation = df_prepared_validation.sample(frac=1).reset_index(drop=True)
    x_validation = df_prepared_validation.drop("Vote", 1)
    y_validation = df_prepared_validation["Vote"]

    # Train and evaluate performances of multiple models
    models = train_some_models(x_train, y_train, x_validation, y_validation)

    # Select the best model for the prediction tasks
    best_model_clf = auto.find_best_model(models)

    # Load prepared test set
    df_prepared_test = pd.read_csv("prepared_test.csv")
    # shuffle
    df_prepared_test = df_prepared_test.sample(frac=1).reset_index(drop=True)
    x_test = df_prepared_test.drop("Vote", 1)
    y_test = df_prepared_test["Vote"]

    x_train_and_validation = x_train.append(x_validation).reset_index(drop=True)
    y_train_and_validation = y_train.append(y_validation).reset_index(drop=True)

    print("the best score from random forest on train + validation is:")
    print_cross_val_accuracy(best_model_clf, x_train_and_validation, y_train_and_validation)

    best_model_clf.fit(x_train_and_validation, y_train_and_validation)
    y_test_pred = best_model_clf.predict(x_test)

    # Use the selected model to provide the following:
    # vote division
    vote_division(y_test_pred, y_test)

    # the party that wins the elections is:
    print()
    winner_party(best_model_clf, x_test)
    print()

    # save
    save_voting_predictions(y_test_pred)

    # test confusion matrix
    plot_confusion_matrix(best_model_clf, x_test, y_test)
    plt.show()

    # overall test error
    calculate_overall_test_error(y_test, y_test_pred)

    # each party would like to suggest transportation services for its voters
    estimate_transportation(best_model_clf, x_test)

    # task 3: Predict the vote of each voter in the new sample

    # task 4: What will be a steady coalition – show why
    # coalition will be more stable than other coalitions


if __name__ == '__main__':
    main()
