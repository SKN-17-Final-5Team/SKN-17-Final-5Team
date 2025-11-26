## Memory module 업데이트

### 수정된 파일
- memory_modules.repository.py
    - 채팅방 제목: 첫 질의 요약 & 10글자 이내
    - 워크플로우 제목: 초기 untitle로 & 10글자 이내

- my_agents.trade_agent.py
    - 히스토리 참조해서 대화하도록 추가함
    
- main.py -> start.py 로 수정함 
- model_memory.sql : ERD 기준으로 DB 구성 업데이트함