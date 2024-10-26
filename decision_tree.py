import pickle
from typing import Callable, Any, Optional, Dict

class Node:
    def __init__(
        self, 
        question: Optional[str] = None,
        question_func: Optional[Callable] = None,
        true_branch: Optional['Node'] = None, 
        false_branch: Optional['Node'] = None, 
        result: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        id: Optional[str] = None
    ):
        self.question = question
        self.question_func = question_func
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.result = result
        self.prompt_template = prompt_template or "Please answer yes or no: {question}"
        self.id = id or question

class DecisionTree:
    def __init__(self):
        self.root = None
        self.current_node = None
        self.collected_data = {}
    
    def fit(self, root_node: Node):
        self.root = root_node
        self.reset()
    
    def reset(self):
        self.current_node = self.root
        self.collected_data = {}
    
    def get_next_prompt(self) -> Optional[str]:
        if self.current_node is None or self.current_node.result is not None:
            return None
        
        return self.current_node.prompt_template.format(
            question=self.current_node.question
        )
    
    def predict(self, data: Dict[str, bool]) -> Any:
        """Make a prediction using a complete set of data"""
        node = self.root
        
        while node.result is None:
            if node.id in data:
                result = data[node.id]
            elif node.question in data:
                result = data[node.question]
            elif node.question_func:
                result = node.question_func(data)
            else:
                raise KeyError(f"No answer found for question: {node.question} (id: {node.id})")
                
            node = node.true_branch if result else node.false_branch
                
        return node.result
    
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename: str) -> 'DecisionTree':
        with open(filename, 'rb') as f:
            return pickle.load(f)