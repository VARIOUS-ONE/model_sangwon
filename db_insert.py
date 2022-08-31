import sqlite3
import pandas as pd
import datetime

review = pd.read_csv('./models/dummy.csv')
con = sqlite3.connect("./models/db.sqlite3", detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
cur = con.cursor()
review['createdAt'] = datetime.datetime.now()
print(type(datetime.datetime.now()))
for i in range(len(review)):
    cur.execute('insert into base_review VALUES(?,?,?,?,?,?,?,?);', \
    (review['name'].iloc[i], review['rating'].iloc[i], review['comment'].iloc[i], datetime.datetime.now(), \
    review['_id'].iloc[i], review['product_id'].iloc[i], review['user_id'].iloc[i], review['is_positive'].iloc[i]))
con.commit()

cur.execute("select * from base_review")
data = cur.fetchone()
print(pd.Dataframe(data).shape)
cur.close()
con.close()
