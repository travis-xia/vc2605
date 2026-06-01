from transformers import AutoModel, AutoTokenizer
import torch

# model setting
model_path = './'
text_only = True  # True: no video, text QA only

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()

model.config.mm_llm_compress = False

generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)

question = "What is 1+1?"

if text_only:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_config)
    print(tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
else:
    video_path = "test.mp4"
    max_num_frames = 512
    output, _ = model.chat(
        video_path=video_path,
        tokenizer=tokenizer,
        user_prompt=question,
        return_history=True,
        max_num_frames=max_num_frames,
        generation_config=generation_config,
    )
    print(output)
