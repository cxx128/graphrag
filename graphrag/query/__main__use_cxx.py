# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Query Engine package root."""

import streamlit as st

import argparse
from enum import Enum

import sys
sys.path.append('/data/chenxiaoxuan/cxxpythonfiles/GraphRAG/ragtest')
print(sys.path)
from graphrag.query.cli_use_cxx import run_global_search, run_local_search

INVALID_METHOD_ERROR = "Invalid method"


class SearchType(Enum):
    """The type of search to run."""

    LOCAL = "local"
    GLOBAL = "global"

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        help="The path with the output data from the pipeline",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--root",
        help="The data project root. Default value: the current directory",
        required=False,
        default="/data/chenxiaoxuan/cxxpythonfiles/GraphRAG/ragtest",
        type=str,
    )

    parser.add_argument(
        "--method",
        help="The method to run, one of: local or global",
        required=False,
        type=SearchType,
        default='local'
    )

    parser.add_argument(
        "--community_level",
        help="Community level in the Leiden community hierarchy from which we will load the community reports higher value means we use reports on smaller communities",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--response_type",
        help="Free form text describing the response type and format, can be anything, e.g. Multiple Paragraphs, Single Paragraph, Single Sentence, List of 3-7 Points, Single Page, Multi-Page Report",
        type=str,
        default="Multiple Paragraphs",
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
        
        
    if "history" not in st.session_state:
        st.session_state.history = []
    

    

    
    prompt_text = st.chat_input("请输入您的问题")
    
    buttonClean = st.button("清理会话历史", key="clean")
    if buttonClean:
        st.session_state.history = []
        st.session_state.past_key_values = None
        st.rerun()
    

    
    if prompt_text:

        history = st.session_state.history
        history.append({"role":"user","content":prompt_text})

        st.session_state.history = history
        input_query=prompt_text
    
        
        match args.method:
            case SearchType.LOCAL:
                graphrag_response=run_local_search(
                    args.data,
                    args.root,
                    args.community_level,
                    args.response_type,
                    input_query,
                )
            case SearchType.GLOBAL:
                run_global_search(
                    args.data,
                    args.root,
                    args.community_level,
                    args.response_type,
                    input_query,
                )
            case _:
                raise ValueError(INVALID_METHOD_ERROR)
        
        #graphrag_response='ok'
        history.append({"role":"assitant","content":graphrag_response})
        
        
        for i, message in enumerate(st.session_state.history):
            if message["role"] == "user":
                with st.chat_message(name="user", avatar="user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message(name="assistant", avatar="assistant"):
                    st.markdown(message["content"])
    
    
