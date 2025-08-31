from pydantic import BaseModel
from typing import List, Optional, Dict, Any
class TestCase(BaseModel):
    case_id:str; query:str; expected_intent:str; action_expected:Optional[str]=''; domain:str
    test_type:str; context:Optional[str]=''; group_id:Optional[str]=''; difficulty:int=2
    design_logic:str=''; tags:List[str]=[]; meta:Dict[str,Any]={}
