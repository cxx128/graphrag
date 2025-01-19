# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Query Engine package root."""

import streamlit as st
import json
import argparse
from enum import Enum

import sys
sys.path.append('/mnt/afs/chenxiaoxuan/RAG_folder/GraphRAG/ragtest')
print(sys.path)
import ast

from graphrag.utils.cli import dir_exist, file_exist

from graphrag.query.cli import run_global_search, run_local_search

INVALID_METHOD_ERROR = "Invalid method"


class SearchType(Enum):
    """The type of search to run."""

    LOCAL = "local"
    GLOBAL = "global"

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m graphrag.query",
        description="The graphrag query engine",
    )
    parser.add_argument(
        "--config",
        help="The configuration yaml file to use when running the query",
        required=False,
        type=file_exist,
    )
    parser.add_argument(
        "--data",
        help="The path with the output data from the pipeline",
        type=dir_exist,
    )
    parser.add_argument(
        "--root",
        help="The data project root. Default value: the current directory",
        default="/mnt/afs/chenxiaoxuan/RAG_folder/GraphRAG/ragtest",
        type=dir_exist,
    )
    parser.add_argument(
        "--method",
        help="The method to run",
        default="local",
        type=SearchType,
        choices=list(SearchType),
    )
    parser.add_argument(
        "--community_level",
        help="Community level in the Leiden community hierarchy from which we will load the community reports. A higher value means we will use reports from smaller communities. Default: 2",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--response_type",
        help="Free form text describing the response type and format, can be anything, e.g. Multiple Paragraphs, Single Paragraph, Single Sentence, List of 3-7 Points, Single Page, Multi-Page Report. Default: Multiple Paragraphs",
        type=str,
        default="Multiple Paragraphs",
    )
    parser.add_argument(
        "--streaming",
        help="Print response in a streaming manner",
        action="store_true",
    )
    #parser.add_argument(
    #    "query",
    #    nargs=1,
    #    help="The query to run",
    #    type=str,
    #)

    args = parser.parse_args()

    
    st.set_page_config(
        page_title="GraphRAG-PowerChat",
        page_icon=":robot:",
        layout="wide"
    )
    st.title("GraphRAG-PowerChat")
    st.subheader("使用步骤")
    st.markdown("""
    1、输入一个powerchat模型能回答的问题，我会基于文档检索给出一个单轮对话的答复。\n
    (可以打开侧边栏选一个问题示例，代替手动输入问题)
    """
    )    
        
    if "history" not in st.session_state:
        st.session_state.history = []
    
    with open('/mnt/afs/chenxiaoxuan/RAG_folder/GraphRAG/examples_powerchat.json','r') as jsonfile:
        
        json_data=[json.loads(l) for l in open('/mnt/afs/chenxiaoxuan/RAG_folder/GraphRAG/examples_powerchat.json')][::10]
    ac_list=["None"]+[x["instruction"] for x in json_data]
    
    # 初始化会话状态中的选择，如果不存在的话
    if 'selection' not in st.session_state:
        st.session_state.selection = 'None'  # 设置默认选择

    # 创建单选按钮，并绑定到会话状态
    st.session_state.selection = st.sidebar.radio(
        "可以选择以下一个选项作为问题输入，代替手动输入问题：",
        ac_list
    )
    

    
    prompt_text = st.chat_input("请输入您的问题")
    if str(st.session_state.selection) !="None":
        prompt_text=str(st.session_state.selection)
    
    buttonClean = st.button("清理会话历史", key="clean")
    if buttonClean:
        st.session_state.selection="None"
        st.session_state.history = []
        st.session_state.past_key_values = None
        #st.rerun()
        
    
    if prompt_text:
        
        graphrag_response=''
        # 5. 一个输出，包含两张图片和一段文本
        st.subheader("系统输出")
        status_text = st.empty()  # 用于显示状态文本
        status_text.text(f"正在推理...")
        
        history = st.session_state.history
        history.append({"role":"user","content":prompt_text})

        st.session_state.history = history
        input_query=prompt_text
    
        
        match args.method:
            case SearchType.LOCAL:
                graphrag_response,_=run_local_search(
                    args.config,
                    args.data,
                    args.root,
                    args.community_level,
                    args.response_type,
                    args.streaming,
                    input_query,
                )
            case SearchType.GLOBAL:
                run_global_search(
                    args.config,
                    args.data,
                    args.root,
                    args.community_level,
                    args.response_type,
                    args.streaming,
                    input_query,
                )
            case _:
                raise ValueError(INVALID_METHOD_ERROR)
        
        #graphrag_response='ok'
        history.append({"role":"assitant","content":graphrag_response})
        
        if len(graphrag_response)>0:
            status_text.text(f"")
        for i, message in enumerate(st.session_state.history):
            if message["role"] == "user":
                with st.chat_message(name="user", avatar="user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message(name="assistant", avatar="assistant"):
                    st.markdown(message["content"])
    else:
        st.write("请先输入一个问题，然后我会提供输出。\n")
    
