from passlib.context import CryptContext

# pwd_context 설정 (bcrypt 알고리즘 사용)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """비밀번호를 해싱하여 반환"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """입력받은 평문 비밀번호와 DB의 해시값을 비교"""
    return pwd_context.verify(plain_password, hashed_password)