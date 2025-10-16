from typing import List, Union, Literal
import glob
from tqdm import tqdm
import os
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def remove_non_utf8_characters(text: str) -> str:
    return "".join([char if ord(char) < 128 else " " for char in text])

def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self) -> None:
        self.num_process = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass

class PDFLoader(BaseLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __call__(self, files: List[str], **kwargs):
        num_multiprocess = min(kwargs["num_workers"], self.num_process)
        all_docs = []
        total_files = len(files)
        with multiprocessing.Pool(processes=num_multiprocess) as pool:
            for docs in tqdm(pool.imap(load_pdf, files), total=total_files):
                all_docs.extend(docs)
        return all_docs

class TextSplitter:
    def __init__(
        self, 
        separators: List[str] = ["\n\n", "\n", " ", ""],
        chunk_size: int = 300,
        chunk_overlap: int = 10
    ) -> None:
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def __call__(self, docs: List, **kwargs):
        return self.splitter.split_documents(docs)

class Loader:
    def __init__(
        self, 
        file_type: Literal["pdf"] = "pdf",
        split_kwargs: dict = {
            "chunk_size": 300,
            "chunk_overlap": 10
        },
        text_splitter: Union[TextSplitter, None] = None, **kwargs) -> None:
        
        assert file_type in ["pdf"], f"File type {file_type} not supported"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader(**kwargs)
        else:
            raise ValueError(f"File type {self.file_type} not supported")

        self.text_splitter = TextSplitter(**split_kwargs)

    def load(self, files: Union[str, List[str]], **kwargs, workers: int=1):
        if isinstance(files, str):
            files = [files]

        doc_loaded = self.doc_loader(files, workers=workers)
        doc_loaded = self.text_splitter(doc_loaded)
        return doc_loaded
    
    def load_dir(self, dir_path: str, workers: int=1):
        if self.file_type == "pdf":
            files = glob.glob(os.path.join(dir_path))
            assert len(files) > 0, f"No files found in {dir_path}"
        else:
            raise ValueError(f"File type {self.file_type} must be pdf")
    
        return self.load(files, workers=workers)
    
