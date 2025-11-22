import os
import json
from typing import List, Dict, Any
import pandas as pd
from openai import OpenAI
import asyncio
from agents import Agent, Runner
from tools.search_tool import search_trade_documents
from my_agents.trade_agent import load_instructions
import config
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -------------------------------------
# 0. OpenAI 클라이언트 설정
# -------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------------------
# 1. Judge 프롬프트 정의
# -------------------------------------

# 할 일 : user_question과 system_answer를 보고 1~4 점수로 평가
# 점수 기준 (1~4)에 대한 설명 포함

IMPROVED_JUDGE_PROMPT = """
당신에게는 [question]과 [answer] 한 쌍이 주어질 것입니다.
당신의 임무는 system_answer가 user_question에서 표현된 사용자의 우려를 얼마나 잘 해결했는지를 평가하는 'total rating'을 제공하는 것입니다.
평가는 1에서 4까지의 척도로 제공하며, 1은 [answer]가 전혀 도움이 되지 않는 경우를 의미하고, 4는 [answer]가 [question]을 완전히 그리고 유용하게 해결하는 경우를 의미합니다.

당신이 평가를 구성할 때 사용해야 할 기준은 다음과 같습니다:
1: 질문의 맥락을 파악하지 못했고, 답변에 사실적 오류 또는 부정확한 정보가 포함되어 있다.
     - 질문의 핵심 요소를 잘못 해석함
     - 실무 기준 또는 법적 기준과 어긋나는 잘못된 정보를 제공함
     - 예시:
    {
     "question": "바이어가 계약할 때 알려준 사이트와 회사 로고가 픽셀이 살짝 깨져있는게 보이고 지금 대금 납부하기로 한 날짜가 지났는데 대금의 일부가 회수되지 않은 상황이거든. 사기를 당한건지 그렇다면 어떻게 해결할 수 있는지 알려줘",
     "answer": "바이어와 셀러의 비용과 위험이전은 CISG에서 규정하고 있습니다. 이를 통해 해결할 수 있습니다.",
     "score": 1,
     "reason": "비용이전과 위험이전에 관한 사항은 Incoterms에서 규정하고 있으며, CISG에서는 이에 대한 구체적인 해결책을 제공하지 않습니다. 따라서 답변이 부적절합니다."
    },
    {
    "question": "사우디아라비아로 CIP조건으로 식품 수출계약을 맺고 싶습니다. 인증받아야 하는 사항이 있는지, CIP조건에 대해 설명해주세요",
    "answer": "사우디아라비아는 남아메리카에 위치한 국가입니다. CIP조건은 Incoterms2020에 있는 조건입니다",
    "score": 1,
    "reason": "질문 맥락 파악을 전혀 하지 못했습니다. 또한 CIP 조건에 대한 설명이 부족하여 답변이 부적절합니다."
    },
    {
    "question": "국내 거래만 진행하다가 처음으로 수출을 하게 되어 오퍼상을 끼고 오퍼를 보내려고 하는데 오퍼가 정확히 언제 효력이 발생하는지, 오퍼상은 어떤 역할인건지 설명해줘. 수출 상대국도 cisg를 적용하는 국가야.",
    "answer": "CISG는 청약에 대해 설명하고 있습니다. 청약은 절대 철회할 수 없습니다.",
    "score": 1,
    "reason": "질문 맥락 파악을 전혀 하지 못했으며, 청약에 대한 설명이 틀린 정보입니다."
    },
    {
    "question": "마켓클레임과 정당한 클레임을 어떻게 구분해야 하는지, 준거법은 CISG로 계약서에 명시했는데, 매도인으로서 받을 수 있는 권리 보장은 무엇인지 알려줘.",
    "answer": "마켓클레임은 정당한 클레임으로, 분쟁해결절차를 통해 해결 가능합니다.",
    "score": 1,
    "reason": "질문 맥락 파악을 전혀 하지 못했으며, 마켓클레임에 대한 설명이 틀린 정보입니다."
    },
    {
    "question": "이번에 처음으로 연락이 닿은 대만의 정부기관 담당자와 거래를 진행할 예정인데 정부와 관련되어 있기 때문에 더 믿을만 하지 않을까 생각이 들어. DDP 조건으로 진행하는 대신 물품 가격은 우리 회사가 생각하는 것보다 조금 더 비싸게 진행하게 되었어. 이상하게 기분이 찜찜한데 조건이 잘못된건지 체크해줘.",
    "answer": "정부기관 담당자는 신뢰할 수 있습니다. 무역 거래에서는 신뢰를 기반으로 한 계약이 중요합니다.",
    "score": 1,
    "reason": "질문 맥락 파악을 전혀 하지 못했습니다. 정부기관 담당자라고 해서 무조건 신뢰할 수 있는 것은 아니며, 신뢰를 기반으로 사기가 발생하는 경우가 많아 답변 또한 틀린 정보입니다."
    }

2: 질문의 맥락을 일부 잘못 해석하거나, 답변에 사실 오류가 일부 포함되어 있다.
     - 핵심 의도는 부분적으로 이해했으나 중요한 부분을 놓침
     - 일부 정보가 틀렸거나 부정확함
     - 예시:
    {
    "question": "바이어가 계약할 때 알려준 사이트와 회사 로고가 픽셀이 살짝 깨져있는게 보이고 지금 대금 납부하기로 한 날짜가 지났는데 대금의 일부가 회수되지 않은 상황이거든. 사기를 당한건지 그렇다면 어떻게 해결할 수 있는지 알려줘",
    "answer": "계약 당시 바이어의 신뢰성을 조사했다면 충분히 믿을 수 있습니다. 픽셀이 깨지는 경우는 발생하기 쉬운 상황이며, 대금 납부가 어느정도 이루어졌다면 사기 가능성은 낮습니다. 우선 바이어에게 연락하여 상황을 확인하고, 추가 대금 납부 일정을 협의하는 것이 좋습니다.",
    "score": 2,
    "reason": "질문에 대해 제공한 답변에 오류가 있습니다. 픽셀이 깨진 경우, 대금 납부가 지연되는 상황은 충분히 무역사기 가능성이 높은 경우에 해당합니다."
    },
    {
    "question": "사우디아라비아로 CIP조건으로 식품 수출계약을 맺고 싶습니다. 인증받아야 하는 사항이 있는지, CIP조건에 대해 설명해주세요",
    "answer": "사우디아라비아로 의료기기를 수출할 때에는 SFDA(의료기기) 인증을 받아야 합니다. CIP조건은 매도인이 선적지에서 자신이 지정한 운송인에게 물품을 인도하되 지정된 목적지까지 운송계약과 보험계약을 체결하고 운송비와 보험료를 지급하는 조건입니다.",
    "score": 2,
    "reason": "질문 맥락을 잘못 파악했습니다. 사우디아라비아로 식품 수출 계약을 하기 위해 사용자가 질문한 것입니다."
    },
    {
    "question": "국내 거래만 진행하다가 처음으로 수출을 하게 되어 오퍼상을 끼고 오퍼를 보내려고 하는데 오퍼가 정확히 언제 효력이 발생하는지, 오퍼상은 어떤 역할인건지 설명해줘. 수출 상대국도 cisg를 적용하는 국가야.",
    "answer": "수입승인(I/L)은 수입물품이 수출입공고 등에서 수입이 제한되는 경우에 제한조치에 합당한 수입요령에 따라 받는 승인이다.",
    "score": 2,
    "reason": "질문 맥락을 전혀 파악하지 못했습니다. 질문은 수입승인에 대해 묻는 것이 아닙니다."
    },
    {
    "question": "마켓클레임과 정당한 클레임을 어떻게 구분해야 하는지, 준거법은 CISG로 계약서에 명시했는데, 매도인으로서 받을 수 있는 권리 보장은 무엇인지 알려줘.",
    "answer": "마켓클레임은 수입업자가 받은 피해에 대해 정당하게 시장에 제기하는 클레임입니다. 이와 관련하여 매도인이 가질 수 있는 권리는 CISG 제35조~제44조에 규정되어 있습니다.",
    "score": 2,
    "reason": "질문에 대해 제공한 답변에 오류가 있습니다. 마켓클레임은 부당한 클레임으로 이에 대한 설명이 잘못 되었습니다."
    },
    {
    "question": "이번에 처음으로 연락이 닿은 대만의 정부기관 담당자와 거래를 진행할 예정인데 정부와 관련되어 있기 때문에 더 믿을만 하지 않을까 생각이 들어. DDP 조건으로 진행하는 대신 물품 가격은 우리 회사가 생각하는 것보다 조금 더 비싸게 진행하게 되었어. 이상하게 기분이 찜찜한데 조건이 잘못된건지 체크해줘.",
    "answer": "신용장을 수령한 매수인은 신용장의 내용이 매매계약의 내용과 일치하는지 여부를 확인해야 합니다.",
    "score": 2,
    "reason": "질문 맥락을 전혀 하지 못했습니다."
    }

3: 질문의 맥락을 정확히 이해하고 사실 오류 없이 답변했으나, 답변이 너무 간단하거나 충분히 설명되지 않았다.
     - 핵심 개념은 맞지만 깊이가 부족함
     - 질문이 요구한 범위를 충분히 채우지 못함
     - 예시:
    {
    "question": "바이어가 계약할 때 알려준 사이트와 회사 로고가 픽셀이 살짝 깨져있는게 보이고 지금 대금 납부하기로 한 날짜가 지났는데 대금의 일부가 회수되지 않은 상황이거든. 사기를 당한건지 그렇다면 어떻게 해결할 수 있는지 알려줘",
    "answer": "상황을 보았을 때 무역사기의 대표적인 유형으로 있는 서류위조에 해당할 가능성이 있어 보입니다. 대한상사중재원에 대외알선을 신청하여 해결하세요.",
    "score": 3,
    "reason": "질문을 잘 파악했고 답변에 오류가 있지는 않으나 너무 간소하여 불충분하다고 판단됩니다."
    },
    {
    "question": "사우디아라비아로 CIP조건으로 식품 수출계약을 맺고 싶습니다. 인증받아야 하는 사항이 있는지, CIP조건에 대해 설명해주세요",
    "answer": "사우디아라비아로 식품을 수출하기 위해서는 SFDA(식품) 인증을 받아야 합니다.CIP조건은 매도인이 선적지에서 자신이 지정한 운송인에게 물품을 인도하되 지정된 목적지까지 운송계약과 보험계약을 체결하고 운송비와 보험료를 지급하는 조건입니다.",
    "score": 3,
    "reason": "질문을 잘 파악했고 답변에 오류가 있지는 않으나 너무 간소하여 불충분하다고 판단됩니다."
    },
    {
    "question": "국내 거래만 진행하다가 처음으로 수출을 하게 되어 오퍼상을 끼고 오퍼를 보내려고 하는데 오퍼가 정확히 언제 효력이 발생하는지, 오퍼상은 어떤 역할인건지 설명해줘. 수출 상대국도 cisg를 적용하는 국가야.",
    "answer": "CISG제 14조~24조는 오퍼(청약)에 대해 설명하고 있습니다. 청약은 상대방에게 도달한 때에 효력이 발생합니다. 오퍼상은 무역대리인으로서 매매계약의 당사자가 아니고 해당 거래로부터 발생하는 책임과 발생한 비용은 권한을 위임한 본인에게 귀속됩니다.",
    "score": 3,
    "reason": "질문을 잘 파악하였고 오류 없는 답변을 생성했으나, 다소 간소하여 불충분하다고 판단됩니다."
    },
    {
    "question": "마켓클레임과 정당한 클레임을 어떻게 구분해야 하는지, 준거법은 CISG로 계약서에 명시했는데, 매도인으로서 받을 수 있는 권리 보장은 무엇인지 알려줘.",
    "answer": "마켓 클레임은 부당한 클레임이기 때문에 증빙자료가 결여된 경우가 많으며, 따라서 수출업자는 수입업자가 마켓클레임을 제기하여 오는 경우 입증자료를 요구함은 물론 검사 보고서 등이 제시되었다고 하더라도 그 클레임의 진위여부를 가려야 합니다. CISG는 제35조~제44조에서 물품의 적합성과 제3자의 권리주장에 관해 정의하고 있습니다.",
    "score": 3,
    "reason": "질문을 잘 파악하였고 오류 없는 답변을 생성했으나, 다소 간소하여 불충분하다고 판단됩니다."
    },
    {
    "question": "이번에 처음으로 연락이 닿은 대만의 정부기관 담당자와 거래를 진행할 예정인데 정부와 관련되어 있기 때문에 더 믿을만 하지 않을까 생각이 들어. DDP 조건으로 진행하는 대신 물품 가격은 우리 회사가 생각하는 것보다 조금 더 비싸게 진행하게 되었어. 이상하게 기분이 찜찜한데 조건이 잘못된건지 체크해줘.",
    "answer": "또한, 첫거래에서 사기 발생 가능성이 매우 높기 때문에 조심할 필요가 있습니다. DDP 조건 또한, Incoterms 내에서 매도인의 최대 의무를 표방하고 있기에 주의해야 합니다.",
    "score": 3,
    "reason": "질문을 잘 파악했고 답변에 오류가 있지는 않으나 너무 간소하여 불충분하다고 판단됩니다."
    }

4: 질문의 맥락을 정확히 이해하고, 충분하고 구체적이며 실무적으로 유효한 정보를 제시했다.
     - 사실 오류 없음
     - 질문의 핵심 요구사항을 모두 충족
     - 조건, 법적 근거, 위험요소 등 필요한 내용을 충분히 다룸
     - 예시:
    {
    "question": "바이어가 계약할 때 알려준 사이트와 회사 로고가 픽셀이 살짝 깨져있는게 보이고 지금 대금 납부하기로 한 날짜가 지났는데 대금의 일부가 회수되지 않은 상황이거든. 사기를 당한건지 그렇다면 어떻게 해결할 수 있는지 알려줘",
    "answer": "상황을 보았을 때 무역사기의 대표적인 유형으로 있는 서류위조에 해당할 가능성이 있어 보입니다. 서류 위조는 위조한 서류를 보내 정상적인 거래인 것처럼 기업을 안심시킨 후 운송비, 제품 등을 갈취하는 유형으로 주로 사업자등록증, 송금증 등을 위조합니다. 정부기관 또는 실존하는 기업의 담당자를 사칭하거나 페이퍼컴퍼니, 임시로 개설한 웹사이트 정보를 제공해 제품 편취를 시도하기도 합니다. 이러한 서류위조 사기의 경우 다른 사기 유형(금품 갈취 등)과 결합하여 나타나는 경우가 많습니다. 이러한 분쟁 특히 사용자의 경우처럼 미수금이 존재하는 경우엔 시간이 지연되면 자금회전 악화로 어려움이 가중될 수 있습니다. 가장 좋은 방법은 예방으로 미수금 발생요인을 사전에 제거하는 것이지만, 그렇지 않은 경우 분쟁 해결 방안으로는 3가지 방법이 있습니다. 첫째, 대한상사중재원에 대외알선을 신청하는 방안 둘째, 국제채권추심회사에 의뢰하는 방안 셋째, 현지 법원에서 소송을 통해 해결하는 방안 (만약 계약서에 중재조항이 있으면 중재를 신청하여 쉽게 해결)",
    "score": 4,
    "reason": "질문을 잘 파악하고 그에 맞는 답변을 구체적으로 제시한 경우입니다. 무역사기 유형을 설명하고, 예방과 해결 방안을 상세히 안내하여 유용한 정보를 제공합니다."
    },
    {
    "question": "사우디아라비아로 CIP조건으로 식품 수출계약을 맺고 싶습니다. 인증받아야 하는 사항이 있는지, CIP조건에 대해 설명해주세요",
    "answer": "사우디아라비아로 식품을 수출하기 위해서는 SFDA(식품) 인증을 받아야 합니다.https://www.globalcerti.kr/nationCertiDetail?seqNo=27에서 더 자세한 정보를 확인할 수 있습니다. CIP조건은 매도인이 선적지에서 자신이 지정한 운송인에게 물품을 인도하되 지정된 목적지까지 운송계약과 보험계약을 체결하고 운송비와 보험료를 지급하는 조건입니다. 이것은 물품이 인도된 이후 발생하는 모든 위험과 비용(운송비 및 보험료 제외)은 매수인이 부담하는 것을 의미합니다. 또한, 매도인은 CIP조건에 따르면 ICC(A)조건이나 그와 유사한 약관에 따른 광범위한 담보 조건으로 부보할 의무가 있습니다. 그러나 당사자들이 더 낮은 수준의 담보조건으로 부보하기로 합의할 수 있습니다.",
    "score": 4,
    "reason": "질문을 잘 파악하고 그에 맞는 답변을 구체적으로 제시한 경우입니다. 인증 정보와 CIP 조건에 대한 상세한 설명을 제공하여 유용한 정보를 제공합니다."
    },
    {
    "question": "국내 거래만 진행하다가 처음으로 수출을 하게 되어 오퍼상을 끼고 오퍼를 보내려고 하는데 오퍼가 정확히 언제 효력이 발생하는지, 오퍼상은 어떤 역할인건지 설명해줘. 수출 상대국도 cisg를 적용하는 국가야.",
    "answer": "CISG제 14조~24조는 오퍼(청약)에 대해 설명하고 있습니다. 청약은 상대방에게 도달한 때에 효력이 발생합니다. 청약은 철회될 수 없는 것이더라도, 회수의 의사표시가 청약의 도달 전 또는 그와 동시에 상대방에게 도달하는 경우에는 회수될 수 있습니다. 청약에 대한 더 많은 내용과 예외사항에 관한 것은 CISG에서 확인할 수 있습니다. 오퍼상은 무역대리인으로서 매매계약의 당사자가 아니고 해당 거래로부터 발생하는 책임과 발생한 비용은 권한을 위임한 본인에게 귀속됩니다. 따라서 계약이행에 대한 책임도 권한을 위임한 본인에게 귀속되므로 대리인은 이에 대한 책임을 전혀 질 필요가 없으며, 상대방에게 책임을 묻고자 할 경우에는 본인으로부터 클레임에 대한 권한까지 위임받아야 합니다.",
    "score": 4,
    "reason": "질문을 잘 파악하고 그에 맞는 답변을 구체적으로 제시한 경우입니다. 오퍼의 효력 발생 시점과 오퍼상의 역할에 대해 상세히 안내하여 유용한 정보를 제공합니다."
    },
    {
    "question": "마켓클레임과 정당한 클레임을 어떻게 구분해야 하는지, 준거법은 CISG로 계약서에 명시했는데, 매도인으로서 받을 수 있는 권리 보장은 무엇인지 알려줘.",
    "answer": "마켓클레임은 수입업자가 물품의 수입과 관련하여 시황에 따른 손해를 보거나 또는 극히 미소한 손해를 보았음에도 수출업자의 과실을 구실로 가격인하 혹은 손해배상을 요구하는 것을 일컫습니다. 마켓 클레임은 부당한 클레임이기 때문에 증빙자료가 결여된 경우가 많으며, 따라서 수출업자는 수입업자가 마켓클레임을 제기하여 오는 경우 입증자료를 요구함은 물론 검사 보고서 등이 제시되었다고 하더라도 그 클레임의 진위여부를 가려야 합니다. 실무에서는 경기가 좋지 않은 경우에는 사소한 하자를 트집잡아 클레임으로 발전하는 경우도 있으니, 이를 유의해야 합니다. 또한, 무역계약 시 계약서 상에 클레임을 제기할 수 있는 기한을 약정해 둔 경우 이 기간이 가장 최우선적으로 적용됩니다. 또한, 물품이 변질되기 쉬운 경우에는 그 기간을 단축시키는 것이 수출업자의 입장에서는 유리합니다. CISG는 제35조~제44조에서 물품의 적합성과 제3자의 권리주장에 관해 정의하고 있습니다. 제38조, 제39조에 따르면 매수인은 실행가능한 단기간 내에 물품을 검사하거나 검사하게 하여야 하며, 매수인이 물품의 부적합을 발견하였거나 발견할 수 있었던 때로부터 합리적인 기간 내에 매도인에게 그 부적합한 성질을 특정하여 통지하지 아니한 경우에는, 매수인은 물품의 부적합을 주장할 권리를 상실한다고 명시하고 있습니다.  매수인은 물품이 매수인에게 현실로 교부된 날부터 늦어도 2년 내에 매도인에게 제1항의 통지를 하지 아니한 경우에는, 물품의 부적합을 주장할 권리를 상실하며, 이 기간제한이 계약상의 보증기간과 양립하지 아니하는 경우에는 그렇지 않다고 CISG가 매도인의 권리를 보장하고 있는 동시에 계약의 자율성을 인정하고 있기에 계약 체결 시 계약서에 물품의 성질을 고려하여 클레임 제기 기간을 명시하는 것이 중요합니다.",
    "score": 4,
    "reason": "질문을 잘 파악하고 그에 맞는 답변을 구체적으로 제시한 경우입니다. 마켓클레임과 정당한 클레임의 구분 방법과 매도인의 권리 보장에 대해 상세히 안내하여 유용한 정보를 제공합니다."
    },
    {
    "question": "이번에 처음으로 연락이 닿은 대만의 정부기관 담당자와 거래를 진행할 예정인데 정부와 관련되어 있기 때문에 더 믿을만 하지 않을까 생각이 들어. DDP 조건으로 진행하는 대신 물품 가격은 우리 회사가 생각하는 것보다 조금 더 비싸게 진행하게 되었어. 이상하게 기분이 찜찜한데 조건이 잘못된건지 체크해줘.",
    "answer": "기업이나 사업에 대한 상세한 정보 안내보다는 유대 관계를 강조하며 특별한 기회가 있는 것처럼 환심을 사는 것이라면 무역사기의 대표적인 경우이니 의심이 필요합니다. 또한, 첫거래에서 사기 발생 가능성이 매우 크며, 이때 바이어로부터 이메일로만 연락이 오는 경우 특히 이메일 도메인이 바이어 상호명과 미묘하게 유사하거나 Gmail 계정을 사용하는 경우에는 더욱 주의가 필요합니다. DDP 조건 또한, Incoterms 내에서 매도인의 최대 의무를 표방하고 있는 만큼 바이어에 대한 꼼꼼한 정보 확인이 필요합니다.",
    "score": 4,
    "reason": "질문을 잘 파악하고 그에 맞는 답변을 구체적으로 제시한 경우입니다. 무역사기 가능성에 대한 경고와 DDP 조건에 대한 주의사항을 상세히 안내하여 유용한 정보를 제공합니다."
    }

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
# 2. JSONL 파일 로더
# -------------------------------------
def load_eval_dataset_from_jsonl(file_path: str) -> pd.DataFrame:
    """
    JSONL 파일에서 평가 데이터셋 로드

    예상 JSONL 형식:
    {"question": "질문 내용", "answer": "참조 답변 (선택)"}

    Args:
        file_path: JSONL 파일 경로

    Returns:
        DataFrame with columns: question (필수), answer (선택)

    Note:
        - 평가 시에는 question 컬럼만 사용됨
        - answer 컬럼은 로드되지만 에이전트에 전달되지 않음 (reference-free 평가)
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"✓ {len(data)}개 평가 질문 로드됨: {file_path}\n")
    return pd.DataFrame(data)


# -------------------------------------
# 3. 테스트용 에이전트 생성 (웹서치 툴 제외)
# -------------------------------------
def create_test_agent(use_reranker: bool) -> Agent:
    """
    RAG 테스트 전용 에이전트 생성
    - 웹서치 툴 제외
    - Reranker 설정 변경 가능

    Args:
        use_reranker: Reranker 사용 여부

    Returns:
        Test Agent
    """
    # config.USE_RERANKER 설정 변경
    config.USE_RERANKER = use_reranker

    agent_name = f"Trade Agent ({'WITH' if use_reranker else 'WITHOUT'} Reranker)"

    # RAG 해줄 에이전트
    test_agent = Agent(
        name=agent_name,
        model="gpt-4o",
        instructions=load_instructions(),  # 기존 instructions 로드
        tools=[search_trade_documents]  # 웹서치 툴 제외, RAG만 사용
    )

    print(f"✓ 테스트 에이전트 생성: {agent_name}")
    print(f"  - Reranker: {'사용' if use_reranker else '미사용'}")
    print(f"  - Tools: search_trade_documents만 사용 (웹서치 제외)\n")

    return test_agent


# -------------------------------------
# 4. 에이전트 실행 함수
# -------------------------------------
async def run_agent_on_question(agent: Agent, question: str) -> str:
    """
    에이전트에게 질문하고 답변 받기

    Args:
        agent: Agent 인스턴스
        question: 사용자 질문

    Returns:
        Agent의 최종 답변
    """
    # 에이전트 실행 (OpenAI Agents SDK 표준 방식)
    result = await Runner.run(agent, input=question)

    # 최종 답변 추출
    answer = result.final_output

    return answer


# -------------------------------------
# 5. LLM Judge 호출 함수 (Chat Completions API 사용)
# -------------------------------------
def call_llm_judge(question: str, answer: str, model: str = "gpt-5.1") -> Dict[str, Any]:
    """
    LLM Judge 호출 (Chat Completions API 사용)

    Args:
        question: 사용자 질문
        answer: 에이전트 답변
        model: 평가에 사용할 모델

    Returns:
        {'evaluation': str, 'score': float, 'raw_output': str}
    """
    prompt = IMPROVED_JUDGE_PROMPT.format(question=question, answer=answer)

    # Chat Completions API 요청
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0  # 평가의 일관성을 위해 temperature=0
    )

    # 응답 텍스트 추출
    text = response.choices[0].message.content

    # 텍스트 파싱: Evaluation과 Total rating 추출
    evaluation = ""
    score = None

    lines = text.splitlines()
    current_eval_lines = []
    for line in lines:
        # "Total rating:" 줄 찾기
        if line.strip().lower().startswith("total rating:"):
            after_colon = line.split(":", 1)[1]
            try:
                score = float(after_colon.strip())
            except ValueError:
                pass
        # "Evaluation:" 줄 찾기
        elif line.strip().lower().startswith("evaluation:"):
            first_eval = line.split(":", 1)[1].strip()
            current_eval_lines.append(first_eval)
        else:
            if current_eval_lines:
                current_eval_lines.append(line)

    evaluation = "\n".join(current_eval_lines).strip()

    return {
        "raw_output": text,
        "evaluation": evaluation,
        "score": score,
    }


# -------------------------------------
# 6. 전체 평가 실행 함수
# -------------------------------------
async def evaluate_rag_system(
    eval_df: pd.DataFrame,
    agent: Agent,
    judge_model: str = "gpt-4o-mini"
) -> pd.DataFrame:
    """
    RAG 시스템 전체 평가

    Args:
        eval_df: 평가 데이터셋 (question 컬럼 필수)
        agent: 평가할 Agent
        judge_model: LLM Judge 모델

    Returns:
        평가 결과 DataFrame (question, judge_score, human_score 컬럼)
    """
    print(f"\n{'='*60}")
    print(f"평가 시작")
    print(f"총 {len(eval_df)}개 질문 평가")
    print(f"{'='*60}\n")

    questions = []
    judge_scores = []

    for i, row in eval_df.iterrows():
        question = row["question"]
        questions.append(question)

        print(f"\n[{i+1}/{len(eval_df)}] 질문: {question[:100]}...")

        # 1) 에이전트 실행
        try:
            answer = await run_agent_on_question(agent, question)
            print(f"✓ 답변 생성 완료 ({len(answer)} 글자)")
        except Exception as e:
            print(f"⚠️ 에이전트 실행 실패: {e}")
            answer = f"ERROR: {str(e)}"

        # 2) LLM Judge 평가
        try:
            judge_result = call_llm_judge(question, answer, model=judge_model)
            judge_scores.append(judge_result["score"])
            print(f"✓ LLM Judge 점수: {judge_result['score']}/4")
        except Exception as e:
            print(f"⚠️ Judge 평가 실패: {e}")
            judge_scores.append(None)

    # 결과 DataFrame 생성 (question, judge_score, human_score)
    result_df = pd.DataFrame({
        "question": questions,
        "judge_score": judge_scores,
        "human_score": [None] * len(questions)  # 전문가가 나중에 채울 빈 컬럼
    })

    # 평균 점수 계산
    valid_scores = [s for s in judge_scores if s is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\n{'='*60}")
        print(f"✓ 평가 완료")
        print(f"  평균 Judge 점수: {avg_score:.2f}/4.00")
        print(f"  유효 평가: {len(valid_scores)}/{len(judge_scores)}")
        print(f"{'='*60}\n")

    return result_df


# -------------------------------------
# 7. 점수 분포 시각화 (Pie Chart)
# -------------------------------------
def visualize_score_distribution(df: pd.DataFrame, score_column: str = "judge_score", output_path: str = None):
    """
    점수 분포를 Pie Chart로 시각화

    Args:
        df: 평가 결과 DataFrame
        score_column: 점수 컬럼 이름 (기본값: "judge_score")
        output_path: 저장 경로 (None이면 화면에만 표시)
    """
    if score_column not in df.columns:
        print(f"⚠️ {score_column} 컬럼이 없습니다.")
        return

    # 점수 분포 계산
    scores = df[score_column].dropna()
    if len(scores) == 0:
        print(f"⚠️ 유효한 점수가 없습니다.")
        return

    score_counts = scores.value_counts().sort_index()
    avg_score = scores.mean()

    # 한글 폰트 설정 (macOS)
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # Pie Chart 생성
    fig, ax = plt.subplots(figsize=(10, 8))

    # 색상 설정 (1점=빨강, 2점=주황, 3점=노랑, 4점=초록)
    colors = ['#ff6b6b', '#ffa500', '#ffd93d', '#51cf66']

    wedges, texts, autotexts = ax.pie(
        score_counts.values,
        labels=[f"{int(score)}점" for score in score_counts.index],
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors[int(score)-1] for score in score_counts.index],
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    # 제목 추가 (평균 점수 포함)
    title = "Judge 점수 분포" if score_column == "judge_score" else "전문가 점수 분포"
    ax.set_title(
        f"{title}\n평균 점수: {avg_score:.2f}/4.00",
        fontsize=16,
        weight='bold',
        pad=20
    )

    # 범례 추가
    ax.legend(
        wedges,
        [f"{int(score)}점: {count}개" for score, count in score_counts.items()],
        title="점수별 개수",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10
    )

    plt.tight_layout()

    # 저장 또는 표시
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 시각화 저장: {output_path}")
    else:
        plt.show()

    plt.close()


# -------------------------------------
# 8. 메인 실행
# -------------------------------------
async def main():
    """
    메인 실행 함수
    1. JSONL 파일에서 평가 데이터 로드
    2. 에이전트로 답변 생성 및 Judge 평가
    3. question, judge_score, human_score 컬럼을 가진 CSV 저장
    """

    # ===== 1. 평가 데이터 로드 =====
    eval_file = "complex_qa.jsonl"  # 프로젝트 루트에 있어야 함

    if not os.path.exists(eval_file):
        print(f"⚠️ 평가 데이터 파일이 없습니다: {eval_file}")
        print(f"프로젝트 루트 디렉토리에 '{eval_file}' 파일을 생성해주세요.\n")
        print("예시 형식:")
        print('{"question": "FOB와 CIF 조건의 차이는?"}')
        print('{"question": "무역사기 예방 방법은?"}')
        return

    eval_df = load_eval_dataset_from_jsonl(eval_file)

    # ===== 2. 에이전트 생성 및 평가 =====
    print("\n" + "="*60)
    print("RAG 에이전트 평가 시작")
    print("="*60)

    # 에이전트 생성 (use_reranker는 원하는 설정으로 변경 가능)
    agent = create_test_agent(use_reranker=True)

    # 평가 실행
    result_df = await evaluate_rag_system(
        eval_df=eval_df,
        agent=agent,
        judge_model="gpt-4o-mini"  # Judge 모델 선택
    )

    # ===== 3. 결과 저장 =====
    output_file = "evaluation_results.csv"
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n✓ 평가 결과 저장: {output_file}")
    print(f"  컬럼: question, judge_score, human_score")
    print(f"  전문가는 'human_score' 컬럼을 채워주세요.\n")

    # ===== 4. 점수 분포 출력 =====
    print("\n" + "="*60)
    print("Judge 점수 분포")
    print("="*60)

    scores = result_df["judge_score"].dropna()
    if len(scores) > 0:
        avg_score = scores.mean()
        score_dist = scores.value_counts().sort_index()

        print(f"\n평균 점수: {avg_score:.2f}/4.00")
        print(f"\n점수 분포:")
        for score, count in score_dist.items():
            print(f"  {int(score)}점: {count}개 ({count/len(scores)*100:.1f}%)")

    print("\n✓ 모든 평가 완료!")


# -------------------------------------
# 9. 실행
# -------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
