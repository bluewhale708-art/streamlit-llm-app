import os
import streamlit as st
from dotenv import load_dotenv

# LangChain（OpenAI用）
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ================================
# 1. APIキーの読み込み
# ================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ================================
# 2. LLMに問い合わせる関数
# ================================
def ask_llm(user_text: str, expert_type: str) -> str:
    """
    入力テキスト（user_text）と専門家タイプ（expert_type）を受け取り、
    LLM からの回答テキストを返す関数
    """

    # 専門家の種類ごとにプロンプトを切り替える
    if expert_type == "健康アドバイザー":
        system_prompt = "あなたは健康に関するアドバイザーです。安全でやさしいアドバイスを提供してください。"
    elif expert_type == "料理研究家":
        system_prompt = "あなたは料理研究家です。料理や食材について、丁寧で分かりやすく説明してください。"
    else:
        system_prompt = "あなたは丁寧に回答するアシスタントです。"

    # LangChain の LLM モデルを準備
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.5
    )

    # LLM に渡すメッセージ（system + user）
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text)
    ]

    # LLM に問い合わせて回答を取得
    response = llm.invoke(messages)
    return response.content


# ================================
# 3. Streamlit Web アプリ
# ================================
def main():
    st.title("専門家アドバイス アプリ（LangChain × Streamlit）")

    st.write("""
    ### 📘 アプリの概要
    このアプリでは、入力した内容に対して  
    **健康アドバイザー** または **料理研究家** の立場から  
    AI がアドバイスを返してくれます。

    **使い方：**
    1. 専門家をラジオボタンで選ぶ  
    2. 下の入力欄に質問や相談内容を入力  
    3. 「送信」ボタンを押すと AI の回答が表示されます
    """)

    # 専門家選択
    expert = st.radio(
        "専門家を選んでください：",
        ("健康アドバイザー", "料理研究家")
    )

    # ユーザー入力欄
    user_input = st.text_area("質問や相談内容を入力してください")

    # 送信ボタン
    if st.button("送信"):
        if user_input.strip() == "":
            st.warning("入力欄が空です。何か入力してください。")
        else:
            answer = ask_llm(user_input, expert)
            st.write("### ▼ 回答内容")
            st.write(answer)


# メイン処理
if __name__ == "__main__":
    main()
