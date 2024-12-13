from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score,accuracy_score,precision_score,recall_score



def hyper_parameters(model,param_grid,X_train, Y_train):

    f1_scorer = make_scorer(f1_score)
    accuracy_scorer = make_scorer(accuracy_score)
    precision_scorer = make_scorer(precision_score)
    recall_scorer = make_scorer(recall_score)

    scorers=[f1_scorer,accuracy_scorer,precision_scorer,recall_scorer]

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=accuracy_scorer,#scoring="accuracy"
        cv=5,
        verbose=2,
        n_jobs=-1
        )

    grid_search.fit(X_train, Y_train)
   
    print("-----------------------")
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: {:.4f}".format(grid_search.best_score_))

    return grid_search.best_params_,grid_search.best_score_
