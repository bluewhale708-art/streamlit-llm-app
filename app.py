import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def get_llm_response(user_input, expert_type):
    """
    LLMに問い合わせを行い、回答を取得する関数
    
    Args:
        user_input (str): ユーザーの入力テキスト
        expert_type (str): 選択された専門家の種類
    
    Returns:
        str: LLMからの回答
    """
    
    expert_prompts = {
        "医療アドバイザー": "あなたは医療に関する専門知識を持つアドバイザーです。正確で安全な医療情報を提供し、必要に応じて医療機関への受診を勧めてください。",
        "ITエンジニア": "あなたはソフトウェア開発とIT技術に精通したエンジニアです。プログラミング、システム設計、技術的問題解決に関する専門的なアドバイスを提供してください。",
        "料理研究家": "あなたは料理と栄養に関する専門家です。おいしく健康的な料理の作り方、食材の選び方、栄養バランスについてアドバイスしてください。",
        "教育コンサルタント": "あなたは教育に関する専門家です。効果的な学習方法、教育手法、スキル向上のためのアドバイスを提供してください。"
    }
    
    try:
        api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key
        )
        
        
        messages = [
            SystemMessage(content=expert_prompts[expert_type]),
            HumanMessage(content=user_input)
        ]
        
        
        result = llm(messages)
        return result.content
        
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

def main():
    """メイン関数：Streamlitアプリケーションのレイアウトを構成"""
    
    
    st.title(" AI専門家相談アプリ")
    
    
    st.markdown("""
    ## アプリケーションの概要
    このアプリは、異なる分野の専門家AIに相談できるWebアプリケーションです。
    
    ## 操作方法
    1. **専門家を選択**: 下のラジオボタンから相談したい分野の専門家を選んでください
    2. **質問を入力**: テキストエリアに相談内容や質問を入力してください  
    3. **回答を取得**: 「回答を取得」ボタンをクリックして専門家AIの回答を表示します
    
    ## 注意事項
    - OpenAI APIキーが必要です（環境変数`OPENAI_API_KEY`で設定）
    - 医療関連の回答は参考程度とし、重要な健康問題は医療機関にご相談ください
    
    ---
    """)
    
    
    expert_options = ["医療アドバイザー", "ITエンジニア", "料理研究家", "教育コンサルタント"]
    selected_expert = st.radio(
        " 相談したい専門家を選択してください:",
        expert_options,
        index=0
    )
    
    
    expert_descriptions = {
        "医療アドバイザー": " 健康・医療に関する一般的なアドバイスを提供します",
        "ITエンジニア": " プログラミング・技術に関する専門的なサポートを提供します", 
        "料理研究家": " 料理・栄養に関するアドバイスとレシピを提案します",
        "教育コンサルタント": " 学習・教育に関する効果的な方法を提案します"
    }
    
    st.info(expert_descriptions[selected_expert])
    
    
    user_input = st.text_area(
        "ご質問・相談内容を入力してください:",
        height=150,
        placeholder="例：最近眠れなくて困っています。何か良い方法はありますか？"
    )
    
    
    if st.button("回答を取得", type="primary"):
        if not user_input.strip():
            st.warning("質問内容を入力してください。")
            return
            
        api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI APIキーが設定されていません。Streamlit Cloudの場合はSecretsで`OPENAI_API_KEY`を設定してください。")
            return
        
        
        with st.spinner(f" {selected_expert}が回答を考えています..."):
            response = get_llm_response(user_input, selected_expert)
        
        st.markdown("##  専門家からの回答")
        st.markdown(f"**担当: {selected_expert}**")
        st.markdown("---")
        st.markdown(response)

if __name__ == "__main__":
    main()
