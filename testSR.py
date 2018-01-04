import pandas as pd
import sys
sys.path.append('./Scripts')
import contentBased as CB

datos = pd.read_csv("recipes.csv", error_bad_lines=False)
datos.head(3)

content_based = CB.ContentBased()
content_based.fit(datos, 'recipe_str')

print(content_based.predict(['Mayonesa patata']))