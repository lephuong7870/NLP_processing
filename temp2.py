from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Kiểm tra có GPU không để chạy nhanh hơn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng: {device}")

# Tải model và tokenizer
model_name = "Babelscape/mrebel-large"
print(f"Đang tải mô hình {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="vi_VN", tgt_lang="tp_XX")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
print("Đã tải xong mô hình!\n")

# Hàm trích xuất triple đơn giản hóa
def extract_triplets_simple(text):
    """
    Phiên bản đơn giản hơn để trích xuất các triple từ output
    """
    triplets = []
    # Tách các triplet dựa trên token <triplet>
    parts = text.split('<triplet>')
    
    for part in parts[1:]:  # Bỏ qua phần đầu tiên (trước triplet đầu)
        if '<relation>' in part:
            # Tách head và relation+tail
            head_part, relation_tail = part.split('<relation>', 1)
            
            # Lấy head (bỏ qua token <...> nếu có)
            head_tokens = head_part.strip().split()
            head = ' '.join([t for t in head_tokens if not t.startswith('<')])
            
            # Tách relation và tail
            if '>' in relation_tail:
                # Lấy relation token (nằm trong <...>)
                relation_token = relation_tail.split('>')[0] + '>'
                relation_parts = relation_tail.split(relation_token, 1)
                if len(relation_parts) > 1:
                    # Relation là token giữa các dấu <>
                    relation = relation_token[1:-1]  # Bỏ dấu <>
                    # Tail là phần còn lại
                    tail_tokens = relation_parts[1].strip().split()
                    tail = ' '.join([t for t in tail_tokens if not t.startswith('<')])
                    
                    if head and relation and tail:
                        triplets.append({
                            'head': head.strip(),
                            'relation': relation.strip(),
                            'tail': tail.strip()
                        })
    return triplets

# Các câu ví dụ bằng tiếng Việt
test_sentences = [
    "Bên bán là ông Nguyễn Văn A có CMND số 0123456789 và bên mua là bà Trần Thị B có CMND số 9876543210.",
    
    "Hợp đồng mua bán xe giữa anh Phạm Văn C (CMND: 1122334455) và chị Lê Thị D (CMND: 5544332211).",
    
    "Tôi, Nguyễn Thị E, đồng ý bán chiếc xe máy cho anh Hoàng Văn F với giá 30 triệu đồng.",
    
    "Công ty TNHH ABC đại diện bởi ông Trần Văn G bán xe ô tô cho cá nhân bà Phạm Thị H."
]

# Chạy thử từng câu
for i, text in enumerate(test_sentences):
    print(f"\n{'='*60}")
    print(f"VÍ DỤ {i+1}:")
    print(f"Câu: {text}")
    
    # Tokenize
    inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # Sinh kết quả
    with torch.no_grad():
        generated_tokens = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
            max_length=256,
            num_beams=3,
            num_return_sequences=1,
            length_penalty=0,
        )
    
    # Giải mã
    decoded = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
    print(f"\nOutput thô: {decoded}")
    
    # Trích xuất triple
    triplets = extract_triplets_simple(decoded)
    
    # Hiển thị kết quả
    if triplets:
        print(f"\n📊 Kết quả trích xuất được ({len(triplets)} quan hệ):")
        for j, triplet in enumerate(triplets):
            print(f"  {j+1}. [{triplet['head']}] --({triplet['relation']})--> [{triplet['tail']}]")
    else:
        print("\n❌ Không trích xuất được quan hệ nào từ câu này.")
        
        # Thử cách 2: in output đã làm sạch
        cleaned = decoded.replace('<triplet>', ' [TRIPLET] ').replace('<relation>', ' [RELATION] ')
        cleaned = ' '.join([t for t in cleaned.split() if not (t.startswith('<') and t.endswith('>'))])
        print(f"   Output đã làm sạch: {cleaned}")

print("\n" + "="*60)
print("✅ Demo hoàn tất!")