from decision_tree import *


bird_leaf = Node(result="Bird")
reptile_leaf = Node(result="Reptile")
mammal_leaf = Node(result="Mammal")

feathers_node = Node(
    question="Does the animal have feathers?",
    prompt_template="Please ask the user: {question} (Respond with yes/no)",
    true_branch=bird_leaf,
    false_branch=reptile_leaf,
    id="feathers_node"
)

root = Node(
    question="Does the animal give live birth?",
    prompt_template="To classify the animal, I need to know: {question} (Please respond yes/no)",
    true_branch=mammal_leaf,
    false_branch=feathers_node,
    id="root"
)

tree = DecisionTree()
tree.fit(root)

animal = {
    'root': False,
    'feathers_node': True
}
prediction = tree.predict(animal)

print(prediction)