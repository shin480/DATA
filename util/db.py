from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
id = 'web_user'
pw = 'pass'
host = '192.168.0.23:3306'
db = 'data_platform'
url = f'mysql+pymysql://{id}:{pw}@{host}/{db}'

engine = create_engine(url=url, echo=True, pool_size=1)

# 세션생성
session = sessionmaker(bind=engine)

def get_engine():
    return session()