import pandas as pd
import sys
sys.path.append('./Scripts')
import contentBased as CB

datos = pd.read_csv("people_wiki.csv")
datos.head(3)


print(datos)

content_based = CB.ContentBased()
content_based.fit(datos, 'text')

print(content_based.predict(['Award Actor Oscar']))