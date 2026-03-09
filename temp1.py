from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Hàm giải mã các triple (bộ ba) từ kết quả đầu ra của mô hình
def extract_triplets_typed(text):
    """
    Parse văn bản được sinh ra và trích xuất các triple (thực thể - quan hệ - thực thể)
    kèm theo kiểu của thực thể.
    """
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '', '', '', '', ''

    # Xóa các token đặc biệt không cần thiết
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
        if token == "<triplet>" or token == "<relation>":
            current = 't'
            if relation != '':
                triplets.append({
                    'head': subject.strip(),
                    'head_type': subject_type,
                    'type': relation.strip(),
                    'tail': object_.strip(),
                    'tail_type': object_type
                })
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({
                        'head': subject.strip(),
                        'head_type': subject_type,
                        'type': relation.strip(),
                        'tail': object_.strip(),
                        'tail_type': object_type
                    })
                object_ = ''
                subject_type = token[1:-1]
            else:
                current = 'o'
                object_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    
    # Thêm triple cuối cùng nếu còn
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({
            'head': subject.strip(),
            'head_type': subject_type,
            'type': relation.strip(),
            'tail': object_.strip(),
            'tail_type': object_type
        })
    return triplets

# Tải model và tokenizer
# Lưu ý: src_lang được đặt là "vi_VN" cho tiếng Việt
tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", src_lang="vi_VN", tgt_lang="tp_XX")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")

# Cấu hình tham số sinh văn bản
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
    "forced_bos_token_id": None,
}

# Ví dụ câu tiếng Việt từ hợp đồng mua bán xe
text = "Bên bán là ông Nguyễn Văn A có CMND số 0123456789 và bên mua là bà Trần Thị B có CMND số 9876543210."

# Tokenize câu đầu vào
model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')

# Sinh kết quả
generated_tokens = model.generate(
    model_inputs["input_ids"],
    attention_mask=model_inputs["attention_mask"],
    decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
    **gen_kwargs,
)

# Giải mã kết quả
decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

# Trích xuất và in các triple
print("Câu đầu vào:", text)
print("\nKết quả trích xuất:")
for idx, sentence in enumerate(decoded_preds):
    print(f"\nDự đoán {idx + 1}:")
    triplets = extract_triplets_typed(sentence)
    if triplets:
        for t in triplets:
            print(f"  - {t['head']} ({t['head_type']}) --[{t['type']}]--> {t['tail']} ({t['tail_type']})")
    else:
        print("  (Không tìm thấy triple nào)")