# Importing the required Packages
import os 
import pandas as pd
import json
# LangChain related Packages 
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
# Prepare the LangChain Dataframe Agent 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# For OpenAPI 
from helper import get_openai_api_key

# For Visualization and User Interface 
import streamlit as st

def setup_openai():
    # Setup the OpenAI Chat Model 
    oai_model = ChatOpenAI(
        api_key=get_openai_api_key()
    )
    return oai_model

def create_agent(csv_file, llm_model):
    if not csv_file:
        print("CSV File is missing")

    # Loading the CSV Data 
    df = pd.read_csv(csv_file).fillna(value = 0)
    
    agent = create_pandas_dataframe_agent(llm=llm_model,
                                          df=df,verbose=True, 
                                          allow_dangerous_code=True)
    
    return agent

def create_prompt(query):

    # Creating a Prompt 
    CSV_PROMPT_PREFIX = """
    First set the pandas display options to show all the columns,
    get the column names, then answer the question.
    """

    CSV_PROMPT_SUFFIX = """
    - **ALWAYS** before giving the Final Answer, try another method.
    Then reflect on the answers of the two methods you did and ask yourself
    if it answers correctly the original question.
    If you are not sure, try another method.
    - If the methods tried do not give the same result,reflect and
    try again until you have two methods that have the same result.
    - If you still cannot arrive to a consistent result, say that
    you are not sure of the answer.
    - If you are sure of the correct answer, create a beautiful
    and thorough response using Markdown.
    - If the query requires a table, format your answer like this:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

    - For a bar chart, respond like this:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

    - If a line chart is more appropriate, your reply should look like this:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
    - For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}
    - If the answer is not known or available, respond with:
            {"answer": "I do not know."}

    - Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
    For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

    - **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
    ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
    """

    return CSV_PROMPT_PREFIX + query + CSV_PROMPT_SUFFIX

def get_agent_response(agent, query):

    response = agent.invoke(query)
    return response


def decode_response(response):
    print(response['output'])
    print("\n")
    try:
        response_dict = json.loads(response['output'])
        
        # Check if the response is an answer.
        if "answer" in response_dict:
            st.write(response_dict["answer"])

        elif "bar" in response_dict:
            print("Result is a bar chart")
            data = response_dict["bar"]
            try:
                df_data = {
                        col: [x[i] if isinstance(x, list) else x for x in data['data']]
                        for i, col in enumerate(data['columns'])
                    }       
                df = pd.DataFrame(df_data)
                df.set_index("Grievances", inplace=True)
                st.bar_chart(df)
            except ValueError:
                print(f"Couldn't create DataFrame from data: {data}")

        # Check if the response is a line chart.
        elif "line" in response_dict:
            print("Result is line chart")
            data = response_dict["line"]
            try:
                df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
                df = pd.DataFrame(df_data)
                df.set_index("Products", inplace=True)
                st.line_chart(df)
            except ValueError:
                print(f"Couldn't create DataFrame from data: {data}")

        # Check if the response is a table.
        elif "table" in response_dict:
            print("Result is a Table")
            data = response_dict["table"]
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.table(df)
        
        else:
            st.write(response_dict)
    
    except Exception:
        st.write(response['output'])


if __name__ == "__main__":

    # Create the OpenAI Model 
    llm_model = setup_openai()

    st.set_page_config(page_title="Visualizing CSV Data using LangChain")
    st.title("Visualize CSV Data")

    st.write("Please upload your CSV file below.")

    csv_file = st.file_uploader("Upload a CSV" , type="csv")

    query = st.text_area("Send a Query")

    if st.button("Submit Query", type="primary"):
        # Create an agent from the CSV file.
        agent = create_agent(csv_file=csv_file, llm_model=llm_model)

        query_to_prompt = create_prompt(query)

        # Query the agent.
        response = get_agent_response(agent, query_to_prompt)

        # Decode the response.
        decoded_response = decode_response(response)        