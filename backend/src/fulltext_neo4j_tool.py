from __future__ import annotations

from typing import Any, Dict, List

from pydantic import Field

from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import ENTITY_EXTRACTION_PROMPT, PROMPT
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate

from database import Neo4jDatabase

fulltext_search = """
CALL db.index.fulltext.queryNodes("movie", $query) 
YIELD node, score
WITH node, score LIMIT 5
CALL {
  WITH node
  MATCH (node)-[r:!RATED]->(target)
  RETURN coalesce(node.name, node.title) + " " + type(r) + " " + coalesce(target.name, target.title) AS result
  UNION
  WITH node
  MATCH (node)<-[r:!RATED]-(target)
  RETURN coalesce(target.name, target.title) + " " + type(r) + " " + coalesce(node.name, node.title) AS result
}
RETURN result LIMIT 100
"""


def generate_params(input_str):
    movie_titles = [title.strip() for title in input_str.split(',')]
    # Enclose each movie title in double quotes
    movie_titles = ['"' + title + '"' for title in movie_titles]
    # Join the movie titles with ' OR ' in between
    transformed_str = ' OR '.join(movie_titles)
    # Return the transformed string
    return transformed_str


class LLMFulltextGraphChain(Chain):
    """Chain for question-answering against a graph."""

    graph: Neo4jDatabase = Field(exclude=True)
    entity_extraction_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.
        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        qa_prompt: BasePromptTemplate = PROMPT,
        entity_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT,
        **kwargs: Any,
    ) -> GraphQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        entity_chain = LLMChain(llm=llm, prompt=entity_prompt)

        return cls(qa_chain=qa_chain, entity_extraction_chain=entity_chain, **kwargs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Extract entities, look up info and answer question."""
        question = inputs[self.input_key]

        entity_string = self.entity_extraction_chain.run(question)

        self.callback_manager.on_text(
            "Entities Extracted:", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            entity_string, color="green", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            "Query parameters:", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            generate_params(entity_string), color="green", end="\n", verbose=self.verbose
        )
        context = self.graph.query(
            fulltext_search, {'query': generate_params(entity_string)})
        self.callback_manager.on_text(
            "Full Context:", end="\n", verbose=self.verbose)
        self.callback_manager.on_text(
            context, color="green", end="\n", verbose=self.verbose
        )
        result = self.qa_chain({"question": question, "context": context})
        return {self.output_key: result[self.qa_chain.output_key]}


if __name__ == '__main__':
    from langchain.llms import OpenAI

    llm = OpenAI(temperature=0.0)
    database = Neo4jDatabase(host="bolt://100.27.33.83:7687",
                             user="neo4j", password="room-loans-transmissions")
    chain = GraphQAChain.from_llm(llm=llm, verbose=True, graph=database)

    output = chain.run(
        "What type of movie is Top Gun and Matrix?"
    )

    print(output)
