from fastapi import FastAPI
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
import pandas as pd
from util.db import get_engine
app = FastAPI()
app.mount("/view", StaticFiles(directory="view"))

@app.get("/")
def main():
    return RedirectResponse("view/index.html")

# 테이블 내용 가져오기 테스트
@app.get('/read_table')
def read_table():
    df = pd.read_sql_table(table_name='users', con=get_engine())
    print(df.head())
    return df.to_dict() # http://127.0.0.1:8000/read_table
