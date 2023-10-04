from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def evaluate_classifier(classifier, X_val, y_val):
    y_val_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    
    return accuracy, f1

def plot_roc_curve(y_val, y_val_proba):
    fpr, tpr, _ = roc_curve(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_val, y_val_proba):
    precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
    average_precision = average_precision_score(y_val, y_val_proba)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve (AP = {:.2f})'.format(average_precision))
    plt.show()
