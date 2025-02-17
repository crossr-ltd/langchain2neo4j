from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from cypher_database_tool import LLMCypherGraphChain
from keyword_neo4j_tool import LLMKeywordGraphChain
from vector_neo4j_tool import LLMNeo4jVectorChain
from question_generation_tool import QuestionGenerationTool


class MovieAgent(AgentExecutor):
    """Movie agent"""

    @staticmethod
    def function_name():
        return "MovieAgent"
    
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return ["output"]

    @classmethod
    def initialize(cls, movie_graph, model_name, *args, **kwargs):
        if model_name in ['gpt-3.5-turbo', 'gpt-4']:
            llm = ChatOpenAI(temperature=0, model_name=model_name)
        else:
            raise Exception(f"Model {model_name} is currently not supported")

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k = 1)
        readonlymemory = ReadOnlySharedMemory(memory=memory)

        cypher_tool = LLMCypherGraphChain(
            llm=llm, graph=movie_graph, verbose=True, memory=readonlymemory)
        fulltext_tool = LLMKeywordGraphChain(
            llm=llm, graph=movie_graph, verbose=True)
        vector_tool = LLMNeo4jVectorChain(
            llm=llm, verbose=True, graph=movie_graph
        )
        question_tool = QuestionGenerationTool(llm=llm)
    

        # Load the tool configs that are needed.
        tools = [
            Tool(
                name="Cypher search",
                func=cypher_tool.run,
                description="""
                Utilize this tool to search within a gene knowledge graph database, specifically designed to answer gene, disease and drug-related questions.
                This specialized tool offers streamlined search capabilities to help you find the gene information you need with ease.
                Input should be full question.""",
            ),
            
            # Tool(
            #     name="Vector search",
            #     func=vector_tool.run,
            #     description="Utilize this tool when explicity told to use vector search.Input should be full question.Do not include agent instructions.",
            # ),

            # Tool(
            #     name="Keyword search",
            #     func=fulltext_tool.run,
            #     description="""Utilize this tool when explicitly told to use keyword search.
            #     Input should be a list of relevant genes inferred from the question.
            #     """,
            # ),


            Tool(
                name="Question generation",
                func=question_tool.run,
                description="""Utlize the tool when the question given by the user is too general. This tool provide more detailed question 
                relevant to the user question.""",
            ),
            
        ]

        agent_chain = initialize_agent(
            tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

        return agent_chain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)

        try:
            result = super().run(*args, **kwargs)
        except ValueError as e:
            result = str(e)
        if not result.startswith("Could not parse LLM output: `"):
            raise e
        result = result.removeprefix("Could not parse LLM output: `").removesuffix("`")
            
        return result
