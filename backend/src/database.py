from typing import List, Optional, Tuple, Dict

from neo4j import GraphDatabase
import os

from logger import logger

neo4j_url = os.environ.get('NEO4J_URL')
neo4j_user = os.environ.get('NEO4J_USER')
neo4j_pass = os.environ.get('NEO4J_PASS')

class Neo4jDatabase:
    def __init__(self, host: str = neo4j_url,
                 user: str = neo4j_user,
                 password: str = neo4j_pass):
        """Initialize the graph database"""

        self.driver = GraphDatabase.driver(host, auth=(user, password))

    def query(
        self,
        cypher_query: str,
        params: Optional[Dict] = {}
    ) -> List[Dict[str, str]]:
        logger.debug(cypher_query)
        with self.driver.session() as session:
            result = session.run(cypher_query, params)
            # Limit to at most 50 results
            return [r.values()[0] for r in result][:50]


if __name__ == "__main__":
    database = Neo4jDatabase(host=neo4j_url,
                             user=neo4j_user, password=neo4j_pass)

    a = database.query("""
    MATCH (n) RETURN {count: count(*)} AS count
    """)

    print(a)
