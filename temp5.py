from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import time

start = time.time()

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# =========================
# INPUT DOCUMENT
# =========================
TEXT_DOC = """
THƯ MỜI NHẬN VIỆC Kính gửi: Anh /Chị : Lê Thị Thanh Huyền Địa chỉ: 495A Cách Mạng Tháng Tám, Phường 13, Quận 10, Thành phố Hồ Chí Minh. Điện thoại: 0927123456, 0923336666 Email: Huyen.ltt97@gmail.com Chúng tôi vui mừng thông báo tới Anh/Chị đã trúng tuyển trong đợt phỏng vấn vừa qua của Az group. Anh/Chị sẽ: Làm việc tại: 171 Điện Biên Phủ, Phường 15, Bình Thạnh, Thành phố Hồ Chí Minh Chức danh: Kỷ sư phần mềm – Ngày nhận việc: 08/10/2021 Thời gian thử việc: 3 tháng Thời gian làm việc: 08:30 đến 17:30 từ thứ 2 đến thứ 6 Lương và các chế độ khác: Lương: 10.000.000 Lương thử việc: 8.000.000 Các khoản phụ cấp khác(nếu có): Theo quy định của công ty Các chế độ khác: Theo Luật Lao động Việt Nam, Nội quy lao động và Quy định tài chính của công ty. Bạn có thể xem qua bản mềm hợp đồng lao động và công việc chi tiết hơn chúng tôi đã đính kèm trước khi quyết định. Sau khi nhận được thư mời, bạn vui lòng phản hồi lại cho chúng tôi trước ngày 25/09/2021 Nếu đồng ý, ngày ký hợp đồng của bạn sẽ vào ngày 10/10/2021 ,ngày bắt đầu làm việc của bạn sẽ vào 10/10/2022 Nếu bạn có bất kỳ thắc mắc gì liên quan đến những điều trên, bạn có thể liên lạc với chúng tôi qua số điện thoại: 0588888889 hoặc 0929179999 để được giải đáp. Địa chỉ ip: 167.71.18.92, 192.168.1.1 Hy vọng chúng ta sẽ cùng nhau hợp tác lâu bền và tốt đẹp trong tương lai sớm nhất. Trân trọng!
"""


# =========================
# STEP 1: META PROMPT - TỰ ĐỘNG XÁC ĐỊNH ENTITIES
# =========================
META_PROMPT = f"""
Bạn là chuyên gia NLP trích xuất thông tin từ văn bản.

Hãy đọc văn bản sau và phân tích:

1. Xác định DOMAIN của văn bản (ví dụ: hợp đồng, báo cáo, thư từ...)
2. Xác định các THỰC THỂ CHÍNH xuất hiện trong văn bản (thường là các đối tượng/ cá nhân/tổ chức được nhắc đến)
3. Với MỖI thực thể, hãy liệt kê các THUỘC TÍNH cần trích xuất (ví dụ: tên, tuổi, địa chỉ, số CMND...)
4. Xác định MỐI QUAN HỆ giữa các thực thể (nếu có)

Yêu cầu output JSON:
{{
  "domain": "string",
  "entities": [
    {{
      "entity_id": "A",  // hoặc "B", "C" dựa trên vai trò/vị trí trong văn bản
      "entity_label": "tên gợi nhớ (ví dụ: bên bán, bên mua, người liên quan)",
      "entity_type": "person/organization/object",
      "description": "mô tả ngắn về thực thể này",
      "attributes": ["tên", "năm sinh", "CMND", "địa chỉ", "số điện thoại", ...]
    }}
  ],
  "relationships": [
    {{
      "from_entity": "A",
      "to_entity": "B", 
      "relation_type": "mua_bán/thuê/tặng_cho/ủy_quyền/khác",
      "description": "mô tả chi tiết quan hệ"
    }}
  ]
}}

Văn bản đầu vào:
\"\"\"{TEXT_DOC}\"\"\"
"""

messages_meta = [
    {"role": "system", "content": "Bạn là chuyên gia NLP phân tích văn bản."},
    {"role": "user", "content": META_PROMPT}
]

chat_meta = tokenizer.apply_chat_template(
    messages_meta,
    tokenize=False,
    add_generation_prompt=True
)

inputs_meta = tokenizer(chat_meta, return_tensors="pt").to(model.device)

outputs_meta = model.generate(
    **inputs_meta,
    max_new_tokens=600,
    temperature=0,
    do_sample=False
)

meta_ids = outputs_meta[0][inputs_meta["input_ids"].shape[-1]:]
meta_text = tokenizer.decode(meta_ids, skip_special_tokens=True)

# =========================
# SAFE JSON PARSE
# =========================
def extract_json(text):
    # Thử tìm object JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    # Nếu không, thử tìm mảng JSON
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("Không tìm thấy JSON")

try:
    meta_json = extract_json(meta_text)
except:
    # Nếu parse lỗi, tạo cấu trúc mặc định
    meta_json = {
        "domain": "hợp_đồng_mua_bán_xe",
        "entities": [
            {
                "entity_id": "A",
                "entity_label": "bên_bán",
                "entity_type": "person",
                "description": "Người bán xe",
                "attributes": ["tên", "năm_sinh", "CMND", "ngày_cấp_CMND", "nơi_cấp_CMND", "hộ_khẩu", "địa_chỉ_hiện_tại", "số_điện_thoại"]
            },
            {
                "entity_id": "B",
                "entity_label": "bên_mua",
                "entity_type": "person",
                "description": "Người mua xe",
                "attributes": ["tên", "năm_sinh", "CMND", "ngày_cấp_CMND", "nơi_cấp_CMND", "hộ_khẩu", "địa_chỉ_hiện_tại", "số_điện_thoại"]
            }
        ],
        "relationships": [
            {
                "from_entity": "A",
                "to_entity": "B",
                "relation_type": "mua_bán",
                "description": "A bán xe cho B"
            }
        ]
    }

print("\n=== META OUTPUT (CẤU TRÚC ĐƯỢC XÁC ĐỊNH) ===")
print(json.dumps(meta_json, indent=2, ensure_ascii=False))

# =========================
# STEP 2: NER PROMPT - TRÍCH XUẤT THEO CẤU TRÚC LINH HOẠT
# =========================

# Xây dựng prompt động dựa trên meta_json
entities_description = ""
for entity in meta_json.get("entities", []):
    entities_description += f"\n- Thực thể {entity['entity_id']} ({entity['entity_label']}):\n"
    entities_description += f"  Loại: {entity['entity_type']}\n"
    entities_description += f"  Mô tả: {entity['description']}\n"
    entities_description += f"  Các thuộc tính cần trích xuất: {', '.join(entity.get('attributes', []))}\n"

relationships_description = ""
for rel in meta_json.get("relationships", []):
    relationships_description += f"\n- {rel['from_entity']} --({rel['relation_type']})--> {rel['to_entity']}: {rel['description']}"

# Tạo schema động cho output
output_schema = {
    "entities": []
}

for entity in meta_json.get("entities", []):
    entity_schema = {
        "entity_id": entity["entity_id"],
        "entity_label": entity["entity_label"],
        "attributes": {}
    }
    # Thêm các thuộc tính vào schema
    for attr in entity.get("attributes", []):
        # Chuyển snake_case thành tên field phù hợp
        field_name = attr.lower().replace(" ", "_").replace("-", "_")
        entity_schema["attributes"][field_name] = "string | null"
    
    output_schema["entities"].append(entity_schema)

# Thêm relationships vào schema
output_schema["relationships"] = []
for rel in meta_json.get("relationships", []):
    output_schema["relationships"].append({
        "from_entity": rel["from_entity"],
        "to_entity": rel["to_entity"],
        "relation_type": "string",
        "description": "string | null",
        "evidence": "string | null"  # câu văn thể hiện quan hệ này
    })

# Thêm metadata
output_schema["metadata"] = {
    "document_type": meta_json.get("domain", "unknown"),
    "extraction_date": "2026-03-09"
}

NER_PROMPT = f"""
Bạn là hệ thống trích xuất thông tin chính xác từ văn bản.

THÔNG TIN VĂN BẢN:
- Domain: {meta_json.get('domain', 'unknown')}
- Các thực thể cần tìm: {', '.join([f"{e['entity_id']}({e['entity_label']})" for e in meta_json.get('entities', [])])}

CHI TIẾT CÁC THỰC THỂ VÀ THUỘC TÍNH:
{entities_description}

MỐI QUAN HỆ CẦN XÁC ĐỊNH:
{relationships_description}

NHIỆM VỤ:
Đọc kỹ văn bản và trích xuất:
1. Tất cả thông tin của từng thực thể theo các thuộc tính đã liệt kê
2. Xác định mối quan hệ giữa các thực thể
3. Với mỗi quan hệ, tìm câu văn (evidence) chứng minh quan hệ đó

QUY TẮC QUAN TRỌNG:
- KHÔNG gán cứng vai trò "bán" hay "mua" - hãy để model tự xác định dựa trên văn bản
- Nếu một thuộc tính không xuất hiện trong văn bản, để giá trị null
- Trích xuất CHÍNH XÁC từ ngữ trong văn bản, không suy diễn thêm
- Chú ý các thông tin có thể bị viết liền hoặc xuống dòng
- Phân biệt rõ thông tin của thực thể A và thực thể B

ĐỊNH DẠNG ĐẦU RA:
CHỈ trả về JSON object duy nhất theo schema sau:
{json.dumps(output_schema, indent=2, ensure_ascii=False)}

Văn bản nguồn:
\"\"\"{TEXT_DOC}\"\"\" 
"""

messages_ner = [
    {"role": "system", "content": "Bạn là chuyên gia trích xuất thông tin chính xác, không suy diễn."},
    {"role": "user", "content": NER_PROMPT}
]

chat_ner = tokenizer.apply_chat_template(
    messages_ner,
    tokenize=False,
    add_generation_prompt=True
)

inputs_ner = tokenizer(chat_ner, return_tensors="pt").to(model.device)

outputs_ner = model.generate(
    **inputs_ner,
    max_new_tokens=1200,
    temperature=0,
    do_sample=False
)

ner_ids = outputs_ner[0][inputs_ner["input_ids"].shape[-1]:]
ner_text = tokenizer.decode(ner_ids, skip_special_tokens=True)

print("\n=== FINAL OUTPUT (CẤU TRÚC LINH HOẠT) ===")

try:
    result_json = extract_json(ner_text)
    print(json.dumps(result_json, indent=2, ensure_ascii=False))
    
    # Phân tích thêm về quan hệ
    if "relationships" in result_json:
        print("\n=== PHÂN TÍCH QUAN HỆ ===")
        for rel in result_json["relationships"]:
            print(f"Thực thể {rel['from_entity']} --({rel['relation_type']})--> Thực thể {rel['to_entity']}")
            if rel.get('evidence'):
                print(f"  Evidence: \"{rel['evidence']}\"")
except Exception as e:
    print("Lỗi parse JSON:", e)
    print("\nRaw output:")
    print(ner_text)

end = time.time()
print(f"\n⏱️ Thời gian chạy: {end - start:.2f} giây")