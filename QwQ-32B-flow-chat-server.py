import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from transformers import AutoModel, AutoConfig, LogitsProcessorList, LogitsProcessor
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, json, datetime
import torch, os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import set_seed
from threading import Thread
from fastapi.responses import StreamingResponse


# main文件用于指定路由、配置访问，不进行业务逻辑处理，业务逻辑转移到controllers目录。
app = FastAPI()
# 配置允许域名
origins = [
    "https://319214k20i.goho.co",
    "http://192.168.0.104:8001",
    "http://localhost",
    "*",
    "http://localhost:8080",
]
# 配置允许域名列表、允许方法、请求头、cookie等
app.add_middleware(
    CORSMiddleware,  # 允许跨域
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 4096  # For chat.

    return model, tokenizer


# def chat(model, tokenizer, query, history, max_length=4096, num_beams=1,
#          top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
#
#     conversation = []
#     if history is not None:
#         for query_h, response_h in history:
#             conversation.append({"role": "user", "content": query_h})
#             conversation.append({"role": "assistant", "content": response_h})
#     else:
#         history = []
#     conversation.append({"role": "user", "content": query})
#     input_text = tokenizer.apply_chat_template(
#         conversation,
#         add_generation_prompt=True,
#         tokenize=False,
#     )
#     inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
#
#     if logits_processor is None:
#         logits_processor = LogitsProcessorList()
#     logits_processor.append(InvalidScoreLogitsProcessor())
#     gen_kwargs = {"max_length": max_length,
#                   "num_beams": num_beams,
#                   "top_p": top_p,
#                   "temperature": temperature,
#                   "logits_processor": logits_processor,
#                   "attention_mask": inputs['attention_mask'],
#                   **kwargs}
#     outputs = model.generate(inputs['input_ids'], **gen_kwargs, eos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
#     outputs = outputs.tolist()[0][len(inputs["input_ids"][0]): -1]
#     response = tokenizer.decode(outputs)
#     history.append((query, response))
#     return response, history


def _chat_stream(model, tokenizer, query, history):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

class Item(BaseModel):
    query: str
    history: list = []


@app.post("/qwq")
async def create_item(item: Item):
    global model, tokenizer  # 全局变量
    query = item.query
    print(query)
    history = item.history
    # max_length = json_post_list.get('max_length')
    # top_p = json_post_list.get('top_p')
    # temperature = json_post_list.get('temperature')
    return StreamingResponse(_chat_stream(model, tokenizer, query, history), media_type='text/event-stream')


if __name__ == '__main__':
    DEFAULT_CKPT_PATH = "/root/zky/models/QwQ-32B"
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Instruct command-line interactive chat demo."
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    args = parser.parse_args()

    history, response = [], ""

    model, tokenizer = _load_model_tokenizer(args)

    seed = args.seed
    set_seed(seed)

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
