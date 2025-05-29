import pandas as pd
from typing import List

class Recommender:
    def __init__(self, rules: pd.DataFrame):
        self.rules = rules
        

    def recommend(self, items: List[str], top_n=5) -> List[str]:
        """
        Given a list of items, recommend complementary items based on rules.
        """
        items_set = set(items)
        recommended = {}

        for _, row in self.rules.iterrows():
            antecedents = row['antecedents']
            consequents = row['consequents']
            print("@@@@", antecedents, "\n", consequents, "\n",items_set)

            # If the rule applies
            if antecedents.issubset(items_set) and not consequents.issubset(items_set):
                for item in consequents:
                    recommended[item] = max(recommended.get(item, 0), row['confidence'])

        # Sort and return top N
        recommended_sorted = sorted(recommended.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in recommended_sorted[:top_n]]