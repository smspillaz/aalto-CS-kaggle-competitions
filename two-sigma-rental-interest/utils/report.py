import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display

def generate_classification_report_from_preds(preds, orig_data, labels, classes, columns=None):
    report_columns = (columns or ['description', 'price', 'bedrooms', 'bathrooms', 'photos_count'])
    for outer in classes:
        label_indices = frozenset([i for i, e in enumerate(labels) if e == outer])
        for inner in classes:
            classification_indices = [
                i for i, e in enumerate(preds) if e == inner
                and i in label_indices
            ]

            print('Label was {}, classified as {}'.format(outer, inner))
            display(pd.DataFrame(orig_data.iloc[classification_indices][report_columns]).head(5))

    display(sn.heatmap(confusion_matrix(preds, labels),
                       annot=True))
    print(classification_report(preds, labels))


def generate_classification_report(model, orig_data, data, labels, classes):
    preds = model.predict(data)
    return generate_classification_report_from_preds(preds, orig_data, labels, classes)
