import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class AprioriModel:
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = None

    def fit(self, df_onehot: pd.DataFrame):
        # Generate frequent itemsets
        frequent_itemsets = apriori(df_onehot, min_support=self.min_support, use_colnames=True)
        # Generate rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        self.rules = rules
        return rules

    def save_rules(self, filepath):
        self.rules.to_csv(filepath, index=False)

    def load_rules(self, filepath):
        self.rules = pd.read_csv(filepath)

    def get_rules(self):
        return self.rules
