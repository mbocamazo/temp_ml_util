# http://timhomelab.blogspot.com/2014/01/how-to-read-xml-file-into-dataframe.html
from lxml import objectify
import pandas as pd

path = 'file_path'
xml = objectify.parse(open(path))
root = xml.getroot()
root.getchildren()[0].getchildren()
df = pd.DataFrame(columns=('id', 'name'))

for i in range(0,4):
    obj = root.getchildren()[i].getchildren()
    row = dict(zip(['id', 'name'], [obj[0].text, obj[1].text]))
    row_s = pd.Series(row)
    row_s.name = i
    df = df.append(row_s)