try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

class Message(pc.BaseModel):
    """ChatCompletion message"""

    role: str
    content: str
