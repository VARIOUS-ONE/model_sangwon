import torch
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from transformers import BartForConditionalGeneration

def kobart_summary(text):
  tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
  model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
  text = text.replace('\n', ' ')

  raw_input_ids = tokenizer.encode(text)
  input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
  summary_ids = model.generate(torch.tensor([input_ids]),
                              length_penalty = 0.5,
                              num_beams=8,  
                              max_length=128,  
                              eos_token_id=1)
  output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
  print(output)
  return output

text = """
늘 사용하는 제품입니다!!
휴지는 달라붙으면 갖다버리라는 할머님의 말씀
항상 기억하며 안달라붙는 제품 찾느라
휴지 유목민 생활을 하던중에
핫딜이 떠서 한번 구매했다가 계속 이제품만
사용하게된 케이스구요~
동영상에도 나와있듯 달라붙지 않아요
도톰하고 은은한 향이 나서 사용할때마다
기분 좋아지는 제품입니다!!
간혹 같은 회사 같은 제품인데도 그 질이
약간씩 다를때가 있는데~
다행히 이번 제품은 우수한 품질이네요
배송도 빠르게 잘 왔구요~
다만 비닐포장 단 하나로 오는데
이게 장마철에 배송되면 자칫 습기가
먹을수도 있다고 생각됩니다
그런점은 판매자분의 개선사항이 아닐까 싶네요
제품에는 전혀 불만없는 아주 만점이에요^^
"""

kobart_summary(text)