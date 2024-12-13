# Karışıklık matrisi oluşturmak için bir fonksiyon
def confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    # Gerçek ve tahmin edilen değerlere dayalı bir karışıklık matrisi oluştur
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    print(confusion_matrix_result)
    # True Negative (TN), True Positive (TP), False Positive (FP), False Negative (FN) değerlerini al
    tn, fp, fn, tp = confusion_matrix_result.ravel()
    print('TN: %0.2f' % tn)
    print('TP: %0.2f' % tp)
    print('FP: %0.2f' % fp)
    print('FN: %0.2f' % fn)

# ROC eğrisi oluşturmak için bir fonksiyon
def roc_curve(logreg, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    # ROC alanını hesapla
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    # FPR (False Positive Rate), TPR (True Positive Rate) ve eşik değerlerini al
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict(X_test))
    # Grafik alanını ayarla
    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # En kötü durumu gösteren bir çizgi çiz
    plt.plot([0, 1], [0, 1], 'b--')
    # Oluşturduğumuz lojistik regresyonu çiz
    plt.plot(fpr, tpr, color='darkorange', label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    # Etiketleri ve diğer detayları ekle
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    #plt.savefig('Log_ROC')
    #plt.show()


# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt

# # Örnek tahmin ve gerçek sınıflar
# y_true = [0, 1, 0, 1, 1, 1, 0, 0]
# y_scores = [0.1, 0.4, 0.2, 0.7, 0.8, 0.9, 0.3, 0.5]

# # ROC eğrisini hesapla
# fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# # AUC hesapla
# roc_auc = auc(fpr, tpr)

# # ROC eğrisini çiz
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
