from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# 1. .env 파일 로드
load_dotenv()

# 2. 환경변수에서 값 쏙쏙 뽑아오기
id = os.getenv("DB_USER")
pw = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")  # 테일스케일 IP가 들어옵니다
db = os.getenv("DB_NAME")
url = f'mysql+pymysql://{id}:{pw}@{host}/{db}'

engine = create_engine(url=url, echo=True, pool_size=1)

# 세션생성
session = sessionmaker(bind=engine)

def get_engine():
    return session()