import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# 1. 환경 변수 불러오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# 2. 클라이언트 설정
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# 3. Flask 앱 설정
app = Flask(__name__)
CORS(app)

# 4. Pinecone 인덱스 초기화
index_name = "tqd-subjects"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        metadata_config={"indexed": ["text"]}
    )
index = pc.Index(index_name)

# ✅ 서버 상태 확인
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200

# ✅ 텍스트 임베딩 후 저장 (중복 방지 포함)
@app.route("/embed", methods=["POST"])
def embed():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "텍스트를 입력해주세요"}), 400

    # 1. 임베딩 생성
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-large"
    )
    embedding = response.data[0].embedding

    # 2. 유사한 텍스트가 이미 있는지 확인
    search_result = index.query(
        vector=embedding,
        top_k=1,
        include_metadata=True
    )

    if search_result.matches and search_result.matches[0].score >= 0.98:
        return jsonify({"message": "⚠️ 이미 유사한 문장이 저장되어 있습니다."}), 200

    # 3. 저장
    index.upsert([
        ("item-" + str(hash(text)), embedding, {"text": text})
    ])

    return jsonify({"message": "✅ 성공적으로 저장했습니다!"})

# ✅ 의미 기반 검색 후 GPT 응답 생성
@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json()
    question = data.get("question", "").strip()
    threshold = float(data.get("threshold", 0.75))

    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400

    try:
        # 1. 질문 임베딩 생성
        embed_response = client.embeddings.create(
            input=[question],
            model="text-embedding-3-large"
        )
        query_embedding = embed_response.data[0].embedding

        # 2. Pinecone에서 의미기반 검색
        search_result = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        # 3. 점수 기준 필터링
        relevant_texts = [
            match.metadata["text"]
            for match in search_result.matches
            if match.score >= threshold
        ]

        # ✅ 4. 중복 제거 (순서 유지)
        relevant_texts = list(dict.fromkeys(relevant_texts))

        if not relevant_texts:
            return jsonify({"answer": "관련 정보를 찾을 수 없습니다.", "sources": []})

        # 5. GPT 프롬프트 구성
        context = "\n".join(relevant_texts)
        prompt = f"""
        다음은 사용자 질문에 관련된 정보입니다:

        {context}

        위 정보를 참고하여 아래 질문에 답해주세요:

        질문: {question}
        """

        # 6. GPT 응답 생성
        chat_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )

        answer_text = chat_response.choices[0].message.content.strip()

        return jsonify({
            "answer": answer_text,
            "sources": relevant_texts
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # ✅ 의미 유사한 subject 목록 반환
@app.route("/similar_subjects", methods=["POST"])
def similar_subjects():
    data = request.get_json()
    question = data.get("question", "").strip()
    threshold = float(data.get("threshold", 0.75))

    if not question:
        return jsonify({"error": "질문을 입력해주세요"}), 400

    try:
        # 1. 입력된 질문 임베딩 생성
        embed_response = client.embeddings.create(
            input=[question],
            model="text-embedding-3-large"
        )
        query_embedding = embed_response.data[0].embedding

        # 2. Pinecone 유사 문장 검색
        search_result = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )

        # 3. 필터링 및 중복 제거
        matches = [
            match.metadata["text"]
            for match in search_result.matches
            if match.score >= threshold
        ]
        matches = list(dict.fromkeys(matches))  # 중복 제거

        return jsonify({"matches": matches})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ✅ 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
