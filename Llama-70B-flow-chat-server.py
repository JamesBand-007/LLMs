from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import TextIteratorStreamer
from sse_starlette.sse import EventSourceResponse
from threading import Thread


# ------------ 配置 ------------
MODEL_PATH = "/run/user/zky/models/llama3-70B_lora_sft_10W"  # 替换为你的模型路径
PORT = 8081
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------ 加载模型 ------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_4bit=True,  # 启用 4-bit 量化（显存不足时）
)
print("Model loaded!")

# ------------ FastAPI 应用 ------------
app = FastAPI()

# 历史对话存储（简易版，生产环境建议用数据库）
chat_history: Dict[str, List[Dict[str, str]]] = {}

class Message(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]  # 历史对话 + 新消息
    max_length: int = 4096
    temperature: float = 0.6
    stream: bool = False  # 是否启用流式响应

def build_prompt(messages: List[Message]) -> str:
    """将历史对话转换为模型输入的提示文本"""
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "
    return prompt

@app.post("/llama")
async def chat(request: ChatRequest, client_request: Request):
    try:
        # 构建完整提示
        prompt = build_prompt(request.messages)
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # 流式响应
        if request.stream:
            async def generate():
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

                generation_kwargs =  dict(
                    **inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    do_sample=True,
                    streamer=streamer,  # 启用流式生成
                )
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # 逐词返回
                full_response = ""
                for token in streamer:
                    full_response += token
                    yield token.strip()
                
                # 保存历史（非流式部分）
                if client_request.client.host not in chat_history:
                    chat_history[client_request.client.host] = []
                chat_history[client_request.client.host].extend([
                    {"role": "user", "content": request.messages[-1].content},
                    {"role": "assistant", "content": full_response},
                ])

            return EventSourceResponse(generate())

        # 非流式响应
        else:
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 保存历史
            if client_request.client.host not in chat_history:
                chat_history[client_request.client.host] = []
            chat_history[client_request.client.host].extend([
                {"role": "user", "content": request.messages[-1].content},
                {"role": "assistant", "content": response},
            ])
            
            return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(client_request: Request):
    """获取当前用户的历史对话"""
    return chat_history.get(client_request.client.host, [])

@app.delete("/history")
async def clear_history(client_request: Request):
    """清空当前用户的历史对话"""
    if client_request.client.host in chat_history:
        chat_history.pop(client_request.client.host)
    return {"status": "success"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)