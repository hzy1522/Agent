import os

from google.genai.types import Retrieval
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loader, pdf_loader
from utils.file_handler import listdir_with_allowed_type
from utils.file_handler import get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService(object):
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"]
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separator"],
            length_function=len
        )

    def get_retrieve(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的md5做去重
        Returns: None
        """

        def check_md5(md5_for_check: str):
            """"检查传入的md5字符串是否已经被处理过了"""
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                md5_hex_list = f.readlines()
                for line in md5_hex_list:
                    line = line.strip()
                    if line == md5_for_check:
                        return True #文件已经处理过了
                return  False   #文件没有处理过

        def save_md5(md5_for_check: str):
            """保存md5字符串"""
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            elif read_path.endswith("pdf"):
                return pdf_loader(read_path)
            else:
                raise []

        allowed_files_path : list[str] = listdir_with_allowed_type(
           get_abs_path( chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取md5
            md5_hex = get_file_md5_hex(path)
            if check_md5(md5_hex):
                logger.info(f"[加载知识库]文件{path}已经存在于知识库内，跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]文件{path}为空，跳过")
                    continue
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]文件{path}已经分片为空，跳过")
                    continue

                # 将内容存入向量库中
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免重复处理
                save_md5(md5_hex)
                logger.info(f"[加载知识库]文件{path}处理完毕")
            except Exception as e:
                # 记录处理失败的文件 exc_info=True, 会记录详细的异常信息
                logger.error(f"[加载知识库]文件{path}处理失败，{str(e)}", exc_info=True)



if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retrieval = vs.get_retrieve()
    res = retrieval.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("*"*20)

