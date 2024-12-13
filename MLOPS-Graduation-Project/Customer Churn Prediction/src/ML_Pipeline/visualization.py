import matplotlib.pyplot as plt
import seaborn as sns


def correlation(df):
    corr=df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    return corr


def pairplot(df):
    sns.pairplot(df)


def visualization_performances(metric_df):

    # Görselleştirme
    plt.figure(figsize=(16, 8))

    # Accuracy
    df_combined = metric_df.sort_values(by='Accuracy', ascending=False)
    plt.subplot(2, 2, 1)
    sns.barplot(data=df_combined, x='Model', y='Accuracy', palette='Blues')
    plt.title('Accuracy by Model')

    # Precision
    df_combined = metric_df.sort_values(by='Precision', ascending=False)
    plt.subplot(2, 2, 2)
    sns.barplot(data=df_combined, x='Model', y='Precision', palette='Greens')
    plt.title('Precision by Model')

    # Recall
    df_combined = metric_df.sort_values(by='Recall', ascending=False)
    plt.subplot(2, 2, 3)
    sns.barplot(data=df_combined, x='Model', y='Recall', palette='Oranges')
    plt.title('Recall by Model')

    # F1 Score
    df_combined = metric_df.sort_values(by='F1 Score', ascending=False)
    plt.subplot(2, 2, 4)
    sns.barplot(data=df_combined, x='Model', y='F1 Score', palette='Reds')
    plt.title('F1 Score by Model')

    plt.tight_layout()
    plt.show()    