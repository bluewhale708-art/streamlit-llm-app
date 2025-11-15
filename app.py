import os
import streamlit as st

# LangChain（OpenAI用）
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ================================
# 1. APIキーの読み込み
# ================================
def get_api_key():
    """
    環境変数またはStreamlit SecretsからAPIキーを取得
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            pass
    return api_key

# ================================
# 2. LLMに問い合わせる関数
# ================================
def ask_llm(user_text: str, expert_type: str) -> str:
    """
    入力テキスト（user_text）と専門家タイプ（expert_type）を受け取り、
    LLM からの回答テキストを返す関数
    """
    # APIキーの取得
    api_key = get_api_key()
    if not api_key:
        return "エラー: OpenAI APIキーが設定されていません。環境変数またはStreamlit Secretsで設定してください。"

    # 専門家の種類ごとにプロンプトを切り替える
    if expert_type == "健康アドバイザー":
        system_prompt = """あなたは健康に関する専門アドバイザーです。
以下のルールに従って回答してください：
1. 健康、医療、運動、メンタルヘルスなど健康に関する質問には、安全で優しいアドバイスを提供してください。
2. 料理、食材、レシピ、プログラミング、技術、その他健康以外の分野の質問には、「申し訳ございませんが、その質問は私の専門外です。健康に関することでしたらお答えできます。」と回答してください。
3. 専門分野かどうか判断が難しい場合は、健康との関連性を考慮して判断してください。"""
    elif expert_type == "料理研究家":
        system_prompt = """あなたは料理の専門家です。
以下のルールに従って回答してください：
1. 料理、食材、レシピ、調理法、栄養、食文化など料理に関する質問には、丁寧で分かりやすく専門的に説明してください。
2. 健康、医療、プログラミング、技術、その他料理以外の分野の質問には、「申し訳ございませんが、その質問は私の専門外です。料理に関することでしたらお答えできます。」と回答してください。
3. 専門分野かどうか判断が難しい場合は、料理との関連性を考慮して判断してください。"""
    else:
        system_prompt = "あなたは丁寧に回答するアシスタントです。"

    try:
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
    
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"


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
            # APIキーの確認
            if not get_api_key():
                st.error("⚠️ OpenAI APIキーが設定されていません。\n\nStreamlit Cloudの場合：\n1. アプリの設定画面を開く\n2. Secretsタブを選択\n3. 以下を追加:\n```\nOPENAI_API_KEY = \"your-api-key-here\"\n```")
            else:
                with st.spinner("回答を生成中..."):
                    answer = ask_llm(user_input, expert)
                st.write("### ▼ 回答内容")
                st.write(answer)


# メイン処理
if __name__ == "__main__":
    main()
