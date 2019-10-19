import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def show_auroc(y, yhat) : 
    
    plt.figure(figsize=(6, 6))
    auroc = roc_auc_score(y, yhat)
    fpr, tpr, threshold = roc_curve(y, yhat)

    plt.grid()
    plt.plot([0, 1], [0,1], linestyle='--')
    plt.plot(fpr, tpr, marker='.', label='AUROC=%.4f'%auroc)
    plt.legend()
    plt.show()
    
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,threshold))
    return j_ordered[-1][1]

def save_auroc(y, yhat, random_state, epochs, experiment) : 
    
    plt.figure(figsize=(6, 6))
    auroc = roc_auc_score(y, yhat)
    fpr, tpr, threshold = roc_curve(y, yhat)

    plt.grid()
    plt.plot([0, 1], [0,1], linestyle='--')
    plt.plot(fpr, tpr, marker='.', label='AUROC=%.4f'%auroc)
    plt.legend()
    plt.savefig('training_history/%s/randomstate_%s_epochs_%s_auroc' % (experiment, random_state, epochs))
    #plt.show()
    
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,threshold))
    return j_ordered[-1][1]