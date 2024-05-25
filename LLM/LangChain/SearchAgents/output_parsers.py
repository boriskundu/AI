from typing import List, Dict, Any
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Summary(BaseModel):
    summary: str = Field(descrooption = "summary")
    facts: List[str] = Field(descrooption="interesting facts about them")

    def to_dic(self):
        return {"summary": self.summary, "facts": self.facts}

summary_parser = PydanticOutputParser(pydantic_object = Summary)