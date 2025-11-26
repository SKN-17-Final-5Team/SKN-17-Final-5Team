import os
from typing import List, Dict, Any
import pandas as pd
from scipy.stats import pearsonr
from openai import OpenAI
import os
# -------------------------------------
# 0. OpenAI 클라이언트 설정
# -------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key= OPENAI_API_KEY)


# -------------------------------------
# 1. Judge 프롬프트 정의
# -------------------------------------

# 할 일 : user_question과 system_answer를 보고 1~4 점수로 평가
# 점수 기준 (1~4)에 대한 설명 포함

IMPROVED_JUDGE_PROMPT = """
당신에게는 user_question과 system_answer 한 쌍이 주어질 것입니다.
당신의 임무는 system_answer가 user_question에서 표현된 사용자의 우려를 얼마나 잘 해결했는지를 평가하는 ‘total rating’을 제공하는 것입니다.
평가는 1에서 4까지의 척도로 제공하며, 1은 system_answer가 전혀 도움이 되지 않는 경우를 의미하고, 4는 system_answer가 user_question을 완전히 그리고 유용하게 해결하는 경우를 의미합니다.

당신이 평가를 구성할 때 사용해야 할 기준은 다음과 같습니다:
1: 질문의 맥락을 파악하지 못했고, 답변에 사실적 오류 또는 부정확한 정보가 포함되어 있다.
     - 질문의 핵심 요소를 잘못 해석함
     - 실무 기준 또는 법적 기준과 어긋나는 잘못된 정보를 제공함

2: 질문의 맥락을 일부 잘못 해석하거나, 답변에 사실 오류가 일부 포함되어 있다.
     - 핵심 의도는 부분적으로 이해했으나 중요한 부분을 놓침
     - 일부 정보가 틀렸거나 부정확함

3: 질문의 맥락을 정확히 이해하고 사실 오류 없이 답변했으나, 답변이 너무 간단하거나 충분히 설명되지 않았다.
     - 핵심 개념은 맞지만 깊이가 부족함
     - 질문이 요구한 범위를 충분히 채우지 못함

4: 질문의 맥락을 정확히 이해하고, 충분하고 구체적이며 실무적으로 유효한 정보를 제시했다.
     - 사실 오류 없음
     - 질문의 핵심 요구사항을 모두 충족
     - 조건, 법적 근거, 위험요소 등 필요한 내용을 충분히 다룸

피드백은 아래 형식에 따라 제공하십시오:

Feedback:::
Evaluation: (평가 점수를 준 이유를 텍스트로 작성)
Total rating: (1~4 사이 숫자로 평가 점수 작성)

당신의 답변에는 반드시 'Evaluation:'과 'Total rating:'의 값이 포함되어야 합니다.

이제 아래에 질문과 답변이 주어집니다.

Question: {question}
Answer: {answer}

피드백을 제공하십시오.

Feedback:::
Evaluation: """


# -------------------------------------
# 2. LLM Judge 호출 함수
#    - question, answer를 넣으면
#    - {'evaluation': str, 'score': float} 를 반환
# -------------------------------------
def call_llm_judge(question: str, answer: str, model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    prompt = IMPROVED_JUDGE_PROMPT.format(question=question, answer=answer)

    # ChatCompletion 요청
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # responses API의 output 텍스트 추출
    text = response.output[0].content[0].text

    # text 예시 (이렇게 나온다고 가정):
    # Feedback:::
    # Evaluation: ~~ rationale text ~~
    # Total rating: 3
    #
    # 여기서 Evaluation과 Total rating을 파싱
    evaluation = ""
    score = None

    lines = text.splitlines()
    current_eval_lines = []
    for line in lines:
        # "Total rating:" 줄 찾기
        if line.strip().lower().startswith("total rating:"):
            # "Total rating: 3" → 3만 추출
            after_colon = line.split(":", 1)[1]
            try:
                score = float(after_colon.strip())
            except ValueError:
                # 파싱 실패시 None 유지
                pass
        # "Evaluation:" 줄 찾기
        elif line.strip().lower().startswith("evaluation:"):
            # 첫 줄에서 바로 내용이 시작되는 경우도 고려
            first_eval = line.split(":", 1)[1].strip()
            current_eval_lines.append(first_eval)
        else:
            # 이미 Evaluation: 영역을 지나고 있는 중이면 계속 붙이기
            if current_eval_lines:
                current_eval_lines.append(line)

    evaluation = "\n".join(current_eval_lines).strip()

    return {
        "raw_output": text,
        "evaluation": evaluation,
        "score": score,
    }


# -------------------------------------
# 3. 예시: 사람 레이블이 있는 작은 평가 데이터셋
#    - 실제로는 CSV에서 읽거나, 다른 코드에서 생성하면 됨
# -------------------------------------
def build_example_human_dataset() -> pd.DataFrame:
    # 예시용 데이터 (실제로는 더 다양하게 준비)
    data = [
        {
            "question": "What is the capital of France?",
            "system_answer": "The capital of France is Paris.",
            "human_score": 4,  # 사람이 매긴 점수(1~4)
        },
        {
            "question": "Explain what overfitting is in machine learning.",
            "system_answer": "Overfitting is when a model learns the training data too well, including noise, so it performs poorly on new data.",
            "human_score": 4,
        },
        {
            "question": "Explain what overfitting is in machine learning.",
            "system_answer": "Overfitting is when your computer overheats and the model stops working.",
            "human_score": 1,
        },
        {
            "question": "How do you make pasta?",
            "system_answer": "Boil water, add salt, cook pasta until al dente, then drain and add sauce.",
            "human_score": 3,
        },
        {
            "question": "Summarize the benefits of regular exercise.",
            "system_answer": "Regular exercise causes economic growth in your country.",
            "human_score": 1,
        },
    ]
    return pd.DataFrame(data)


# -------------------------------------
# 4. 데이터셋에 대해 LLM Judge 실행
# -------------------------------------
def evaluate_judge_on_dataset(df: pd.DataFrame, model: str = "gpt-4.1-mini") -> pd.DataFrame:
    llm_scores = []
    evaluations = []

    for i, row in df.iterrows():
        q = row["question"]
        a = row["system_answer"]
        result = call_llm_judge(q, a, model=model)
        llm_scores.append(result["score"])
        evaluations.append(result["evaluation"])

        print(f"[{i}] human_score={row['human_score']}, llm_score={result['score']}")
        print("  Evaluation:", result["evaluation"][:100].replace("\n", " "), "...")

    df["llm_score"] = llm_scores
    df["llm_evaluation"] = evaluations

    return df


# -------------------------------------
# 5. 사람 점수 vs LLM 점수 상관계수 계산
# -------------------------------------
def compute_correlation(df: pd.DataFrame) -> float:
    # NaN 제거
    valid_df = df.dropna(subset=["human_score", "llm_score"])
    if len(valid_df) < 2:
        print("Not enough data to compute correlation.")
        return float("nan")

    corr, p_value = pearsonr(valid_df["human_score"], valid_df["llm_score"])
    print(f"Pearson correlation: {corr:.4f} (p={p_value:.4f})")
    return corr


# -------------------------------------
# 6. 메인 실행 예시
# -------------------------------------
if __name__ == "__main__":
    # (1) 예시 사람 평가 데이터셋 생성
    df_human = build_example_human_dataset()

    # (2) LLM Judge 실행
    df_eval = evaluate_judge_on_dataset(df_human, model="gpt-4.1-mini")

    # (3) 상관계수 계산
    corr = compute_correlation(df_eval)

    # (4) 결과 저장 (옵션)
    df_eval.to_csv("llm_judge_results.csv", index=False)
    print("Saved results to llm_judge_results.csv")
