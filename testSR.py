import pandas as pd
import sys
import contentBased as cB
sys.path.append('./Scripts')

data = pd.read_csv("recipes.csv", error_bad_lines=False)
data.head(3)

content_based = cB.ContentBased()
content_based.fit(data, 'recipe_str')

print(content_based.predict(['Mayonesa patata']))
