import re
from langchain import hub
from langchain_core.runnables import RunanablePassthrough
from langchain_core.output_parsers import StrOutputParser

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self,
                       text_response: str, 
                       pattern: str = r"Answer:\s*(.*)"
                       ) -> str:

        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text_response

class OfflineRAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.promt = hub.pull("rlm/rag-prompts")
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunanablePassthrough()
        }
        rag_chain = input_data | self.promt | self.llm | self.str_parser
        return rag_chain
    
    def format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])