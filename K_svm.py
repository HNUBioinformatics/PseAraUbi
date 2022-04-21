import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.metrics as sm



def get_train_set(interaction_matrix):
    cv = "n"
    (rows_index, cols_index) = np.where(interaction_matrix == 1)
    link_num = np.sum(np.sum(interaction_matrix))
    random_index=random.sample(range(0,link_num),link_num)
    size_of_cv=round(link_num//cv)
    print('cv=%d,size_of_cv=%d'%(cv,size_of_cv))

    result=[]
    for k in range(cv):
        print('Cross validation',k+1)
        if (k!=cv):
            test_row_index = rows_index[random_index[size_of_cv*k:size_of_cv*(k+1)]] 
            test_col_index = cols_index[random_index[size_of_cv*k:size_of_cv*(k+1)]]
        else:
            test_row_index=rows_index[random_index[size_of_cv*k:]]
            test_col_index=cols_index[random_index[size_of_cv*k:]]

        train_set=np.copy(interaction_matrix)
        for i in range(test_row_index.shape[0]):
            train_set[test_row_index[i],test_col_index[i]]


        predict_matrix=SVMClassifier(train_set) 
        test_index = np.where(train_set ) 
        real_score=interaction_matrix[test_index]
        predict_score=predict_matrix[test_index]

        num_auc=l5_model_evaluate(real_score,predict_score)
        result.append(num_auc)

    print("last five auc：",result)
    print(" average auc为：\n",np.sum(np.array(result),axis=0)/cv)

def cos_similarity(inteartion_matrix):
    m1 = np.mat(inteartion_matrix) 
    cos_similarity = cosine_similarity(m1)  
    return cos_similarity



model = svm.SVC(kernel="poly", degree=2)
model.fit(train_x, train_y)

prd_test_y = model.predict(test_x)
print(sm.classification_report(test_y, prd_test_y))


left, right = x[:, 0].min() - 1, x[:, 0].max() + 1
bottom, top = x[:, 1].min() - 1, x[:, 1].max() + 1


n = 
grid_x, grid_y = np.meshgrid(np.linspace(left, right, n), np.linspace(bottom, top, n))
mesh_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
mesh_z = model.predict(mesh_x)
grid_z = mesh_z.reshape(grid_x.shape)




def l5_model_evaluate(interaction_matrix, predict_matrix):

    real_score = np.matrix(np.array(interaction_matrix).flatten())
    predict_score = np.matrix(np.array(predict_matrix).flatten())
    metrics = get_metrics(real_score, predict_score)
    return metrics

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))  
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.array(range(1, 1000)) / 1000)]
    thresholds = np.mat(thresholds)

    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1


    TP = predict_score_matrix * real_score.T
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]

    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])


    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    print("aupr:%f,auc:%f,f1_score:%f,acuuracy:%f,recall:%f,specificity:%f,precision:%f" % (
    aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]





