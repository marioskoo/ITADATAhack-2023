import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def view(y_train, name="Class Distribution"):
    # Visualizzazione della distribuzione delle classi
    count = [0] * 20  # Inizializza una lista di zeri con lunghezza pari al numero di classi
    for i in y_train:
        for j in range(20):
            if i[j] != 0:
                count[i[j]-1] += 1
    print(count)
    class_counts = pd.DataFrame({'count': y_train.sum(axis=0)})
    sns.barplot(x=class_counts.index, y=count, data=class_counts)
    plt.xticks(rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(name)
    plt.show()