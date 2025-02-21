from sqlalchemy import Column, BigInteger
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta

from pgvector.sqlalchemy import Vector

Base: DeclarativeMeta = declarative_base()

class Embeddings(Base):
    __tablename__ = 'embeddings'
    id = Column(BigInteger, primary_key=True, index = True)
    vector= Column(Vector(1536))