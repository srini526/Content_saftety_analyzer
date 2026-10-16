import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)

def merge_socre_dicts(existing:dict, newupdated:dict) -> dict:
    if existing is None:
        return newupdated
    return {**existing, **newupdated}

# create state

class AnalyserState(TypedDict):
    raw_input : str
    safety_scores : Annotated[dict[str, int], merge_socre_dicts]

# Create nodes

def toxicity_node(state: AnalyserState) -> dict:
    print ("\n [Branch 1] Analysing Toxicity and Hate Speech\n")

    prompt = (
        'You are an expert content moderation analyst. Analyze the given text for toxicity, hate speech, harassment, and abusive language.'
        "Classify the text as toxic or non-toxic and assign a severity score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic."
        "Return only plain Integer number nothing else"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_scores":{"toxicity_level":score}}

def copyright_node(state: AnalyserState) -> dict:
    print ("\n [Branch 2] Analysing Copyright\n")

    prompt = (
        "You are an expert copyright detection analyst. Analyze the given text for potential copyright infringement, unauthorized reproduction, or substantial similarity to protected content. "
        "Assign a copyright risk score from 0 to 100, where 0 means low risk and 100 means high risk. "
        "Return only a plain integer number, nothing else.\n"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_scores":{"copyright_risk":score}}


def cultural_node(state: AnalyserState) -> dict:
    print ("\n [Branch 3] Analysing Cultural\n")

    prompt = (
        "You are an expert cultural sensitivity analyst. Analyze the given text for cultural insensitivity, stereotypes, disrespect toward traditions, and offensive cultural references. "
        "Assign a cultural sensitivity risk score from 0 to 100, where 0 means culturally respectful and 100 means highly offensive or discriminatory. "
        "Return only a plain integer number, nothing else.\n"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_scores":{"cultural_insensitivity":score}}

builder = StateGraph(AnalyserState)

builder.add_node("toxicity_node",toxicity_node)
builder.add_node("copyright_node",copyright_node)
builder.add_node("cultural_node",cultural_node)

builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_node")
builder.add_edge(START, "cultural_node")

builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_node", END)
builder.add_edge("cultural_node", END)

app = builder.compile()

# sample_script = """A creator makes a video using several paragraphs copied directly from a popular author's book without permission or attribution. In the video, the creator also mocks a religious community's traditional clothing, calling their customs backward and uncivilized. The creator then tells members of that community that they are stupid and do not belong in modern society."""

# initial_state = {
#     "raw_input": sample_script,
#     "safety_scores" : {}
# }

# final_state = app.invoke(initial_state)

#print(final_state["safety_scores"])
