
###  First , use  regex to identity ID 
## Secondly, use  NER model -- had fine-tune to identity ALL Entity 
## Finally, if text phức tạp dùng LLMs để nhận dạng 


## FUNCTION REGEX 




string  =  """ HỢP ĐỒNG MUA BÁN XE MÁY Biển kiểm soát: 29 Hôm nay, ngày 19 tháng 12 năm 2020, tại C.AN Hà Nội, chúng tôi gồm có: BÊN BÁN (BÊN A) Ông(Bà): Đỗ Văn Hưng Ngày/tháng/năm sinh: 03/05/1974 CMND số: C00N10740K19085N do Công an Hà Nội cấp ngày 06/07/1987 Hộ khẩu thường trú: Hữu Bằng, Thạch Nhất, Hà Nội Nơi ở hiện tại: Ba Mát, Hữu Bằng, Thạch Nhất, Hà nội Số điện thoại: 0363865325 BÊN MUA(BÊN B) Ông (Bà): Nguyễn Thị Trúc Linh sinh năm: 1980 CMND số: 00H11N63120138N do Công an Hà Nam cấp ngày 23/09/1995 Hộ khẩu thường trú: Xã Nhân Nghĩa, huyện Lý Nhân, Hà Nam Nơi ở hiện tại: Nguyễn Khánh Toàn, phường Cầu Giấy, Hà Nội Số điện thoại: 0985771235 Chúng tôi tự nguyện cùng nhau lập và ký bản hợp đồng này để thực hiện việc mua bán xe máy/xe môtô, với những điều khoản đã được hai bên bàn bạc và thoả thuận như sau: ĐIỀU 1: ĐẶC ĐIỂM XE MUA BÁN Bên bán là chủ sở hữu của chiếc xe máy/xe môtô nhãn hiệu: HONDA Loại xe: Hai Bánh, màu sơn: Trắng, số máy: F03E-0057735, số khung: 5A04F-070410, biển số đăng ký: 29P1 – 498.89 do Phòng Cảnh sát Giao thông - Công an Hà Nội cấp ngày 04/08/2000 (đăng ký lần đầu ngày 08/06/1999). ĐIỀU 2: SỰ THỎA THUẬN MUA BÁN 2.1. Bên bán đồng ý bán và Bên mua đồng ý mua chiếc xe nói trên như hiện trạng với giá là: 6.000.000 đồng (Sáu triệu đồng) và không thay đổi vì bất kỳ lý do gì. 2.2. Bên bán đã nhận đủ tiền do Bên mua trả và đã giao xe đúng như hiện trạng cho Bên mua cùng toàn bộ giấy tờ có liên quan đến chiếc xe này. Việc giao nhận không có gì vướng mắc. Việc giao tiền, giao xe được hai bên thực hiện bằng việc ký vào biên bàn bàn giao hoặc thực hiện đồng thời bằng việc ký vào hợp đồng này. 2.3. Hai bên thoả thuận: Bên mua nộp toàn bộ các loại lệ phí, thuế liên quan đến việc mua bán ô tô. ĐIỀU 3: CAM ĐOAN 3.1. Bên bán cam đoan: Khi đem bán theo bản hợp đồng này, chiếc xe nói trên thuộc quyền sở hữu và sử dụng hợp pháp của Bên bán; chưa đem cầm cố, thế chấp hoặc dùng để đảm bảo cho bất kỳ nghĩa vụ tài sản nào. 3.2. Bên mua cam đoan: Bên mua đã tự mình xem xét kỹ, biết rõ về nguồn gốc sở hữu và hiện trạng chiếc xe nói trên của Bên bán, bằng lòng mua và không có điều gì thắc mắc. ĐIỀU 4: ĐIỀU KHOẢN CUỐI CÙNG Hai bên đã tự đọc lại nguyên văn bản hợp đồng này, đều hiểu và chấp thuận toàn bộ nội dung của hợp đồng, không có điều gì vướng mắc. Hai bên cùng ký tên dưới đây để làm bằng chứng. BÊN A (Ký, ghi rõ họ và tên) BÊN B (Ký, ghi rõ họ và tên) Đỗ Văn Hưng
"""

label_here = {}


import re
import spacy
from spacy.language import Language
from spacy.tokens import Span
from typing import Dict, List, Tuple

class VietnameseNER:
    """Vietnamese NER với phân loại theo nhóm"""
    
    # Cấu hình các loại entity
    ENTITY_CONFIG = {
        # ===== THÔNG TIN CÁ NHÂN =====
        "PERSONAL": {
            "PHONE": {
                "patterns": [r"0\d{9}", r"84\d{9,10}", r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}"],
                "triggers": ["sđt", "điện thoại", "số điện thoại", "đt"]
            },
            "ID_CARD": {
                "patterns": [r"\b\d{9}\b", r"\b\d{12}\b"],
                "triggers": ["cccd", "căn cước", "cmnd"]
            },
            "SOCIAL_INSURANCE": {
                "patterns": [r"\b\d{10}\b", r"\b\d{13}\b"],
                "triggers": ["bhxh", "bảo hiểm xã hội"]
            },
            "HEALTH_INSURANCE": {
                "patterns": [r"\b[A-Z]{2}\d{13}\b"],
                "triggers": ["bảo hiểm y tế", "bhyt" , "bhy"]
            }
        },
        

        # ===== PHƯƠNG TIỆN =====
        "VEHICLE": {
            "LICENSE_PLATE": {
                "patterns": [r"\b\d{2}[A-Z]{1,2}-\d{4,5}\b", r"\b\d{2}[A-Z]\d{4,5}\b"],
                "triggers": ["biển số", "biển số xe", "số xe"]
            },
            "VEHICLE_ID": {
                "patterns": [r"\b[A-HJ-NPR-Z0-9]{17}\b"],
                "triggers": ["số khung", "số máy", "vin"]
            }
        },

        # ===== TÀI CHÍNH NGÂN HÀNG =====
        "FINANCE": {
            "BANK_ACCOUNT": {
                "patterns": [r"\b\d{9,14}\b", r"\b\d{3} \d{3} \d{3}\b"],
                "triggers": ["stk", "số tài khoản", "tài khoản ngân hàng"]
            },
            "BANK_CARD": {
                "patterns": [r"\b\d{4} \d{4} \d{4} \d{4}\b", r"\b\d{16}\b"],
                "triggers": ["số thẻ", "thẻ atm", "thẻ tín dụng"]
            },
            "MONEY": {
                "patterns": [r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\s*(?:đ|vnđ|vnd)\b"],
                "triggers": ["số tiền", "tiền", "giá", "thành tiền"]
            },
            "TAX_CODE": {
                "patterns": [r"\b\d{10}\b", r"\b\d{13}\b"],
                "triggers": ["mã số thuế", "mst"]
            }
        },
        
        
        # ===== THƯƠNG MẠI =====
        "COMMERCE": {
            "ORDER_ID": {
                "patterns": [r"\b(?:DH|HD|ORDER|OD)\d{6,10}\b"],
                "triggers": ["mã đơn", "đơn hàng", "order"]
            },
            "TRACKING_CODE": {
                "patterns": [r"\b[A-Z0-9]{10,15}\b"],
                "triggers": ["mã vận đơn", "tracking", "vận đơn"]
            },
            "CONTRACT_NUMBER": {
                "patterns": [r"\b(?:HĐ|HD|CT)\d{6,10}\b"],
                "triggers": ["số hợp đồng", "hợp đồng"]
            }
        }
    }

@Language.factory("vietnamese_ner")
def create_vietnamese_ner(nlp, name):
    return VietnameseNERComponent()

class VietnameseNERComponent:
    def __init__(self):
        self.entity_config = VietnameseNER.ENTITY_CONFIG
    
    def __call__(self, doc):
        text = doc.text
        text_lower = text.lower()
        matches = []
        
        # Test tất cả các loại entity
        for category, entities in self.entity_config.items():
            for label, config in entities.items():
                patterns = config["patterns"]
                triggers = config["triggers"]
                
                # Check triggers
                if not any(trigger in text_lower for trigger in triggers):
                    continue
                
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        
                        start, end = match.start(), match.end()
                        context = text_lower[max(0, start-50):end+50]
                        
                        if any(trigger in context for trigger in triggers):
                            # Tìm token boundaries
                            span_start = None
                            span_end = None
                            
                            for token in doc:
                                if token.idx <= start < token.idx + len(token.text):
                                    span_start = token.i
                                if token.idx < end <= token.idx + len(token.text):
                                    span_end = token.i + 1
                                    break
                            
                            if span_start is not None and span_end is not None:
                                matches.append({
                                    'start': span_start,
                                    'end': span_end,
                                    'label': label,
                                    'category': category,
                                    'text': match.group()
                                })
        
    
        matches.sort(key=lambda x: x['start'])
        
        filtered_matches = []
        for match in matches:
            overlap = False
            for existing in filtered_matches:
                if not (match['end'] <= existing['start'] or match['start'] >= existing['end']):
                    overlap = True
                    # Ưu tiên match dài hơn
                    if len(match['text']) > len(existing['text']):
                        filtered_matches.remove(existing)
                        filtered_matches.append(match)
                    break
            
            if not overlap:
                filtered_matches.append(match)
        
        # Tạo spans
        spans = [Span(doc, m['start'], m['end'], label=m['label']) for m in filtered_matches]
        doc.ents = spans
        
        return doc

# Khởi tạo pipeline
nlp = spacy.blank("vi")
nlp.add_pipe("vietnamese_ner")



doc = nlp(string )


# Nhóm theo category
entities_by_category = {}
for ent in doc.ents:
    category = "UNKNOWN"
    for cat, entities in VietnameseNER.ENTITY_CONFIG.items():
        if ent.label_ in entities:
            category = cat
            break
    
    if category not in entities_by_category:
        entities_by_category[category] = []
    entities_by_category[category].append(ent)

# REGEX step 
for category, entities in entities_by_category.items():
    for ent in entities:
        label_here[ent.label_] = label_here.get(ent.label_, [])
        label_here[ent.label_].append(ent.text)

print( label_here )



## NER step 

import torch
from transformers import AutoTokenizer
from train import PhoBERTForNER, NER_LABELS, predict_ner

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
NER_LABELS = [ 
    'B-AGE', 'I-AGE',
    'B-GENDER', 'I-GENDER',
    'B-SKILL','I-SKILL',
    'B-EMAIL','I-EMAIL',
    'B-PERSON','I-PERSON',
    'B-PHONENUMBER','I-PHONENUMBER',
    'B-QUANTITY','I-QUANTITY',
    'B-JOB','I-JOB',
    'B-ORGANIZATION','I-ORGANIZATION',
    'B-PRODUCT','I-PRODUCT',
    'B-IP','I-IP',
    'B-LOCATION','I-LOCATION',
    'B-DATETIME','I-DATETIME',
    'B-EVENT','I-EVENT',
    'B-URL','I-URL',
    'B-PATIENT_ID', 'I-PATIENT_ID',
    'B-SYMPTOM_AND_DISEASE', 'I-SYMPTOM_AND_DISEASE',
    'B-TRANSPORTATION', 'I-TRANSPORTATION',
    'B-MISCELLANEOUS'  , 'I-MISCELLANEOUS' , 
    'O'
]


# load model
model = PhoBERTForNER(num_labels=len(NER_LABELS))
model.load_state_dict(
    torch.load("models/phobert_ner_model.pth", map_location=device)
)
model.to(device)
model.eval()






test_text = (
 """HỢP ĐỒNG MUA BÁN XE MÁY Biển kiểm soát: 29 Hôm nay, ngày 19 tháng 12 năm 2020, tại C.AN Hà Nội, chúng tôi gồm có: BÊN BÁN (BÊN A) Ông(Bà): Đỗ Văn Hưng Ngày/tháng/năm sinh: 03/05/1974 CMND số: C00N10740K19085N do Công an Hà Nội cấp ngày 06/07/1987 Hộ khẩu thường trú: Hữu Bằng, Thạch Nhất, Hà Nội Nơi ở hiện tại: Ba Mát, Hữu Bằng, Thạch Nhất, Hà nội Số điện thoại: 0363865325 BÊN MUA(BÊN B) Ông (Bà): Nguyễn Thị Trúc Linh sinh năm: 1980 CMND số: 00H11N63120138N do Công an Hà Nam cấp ngày 23/09/1995 Hộ khẩu thường trú: Xã Nhân Nghĩa, huyện Lý Nhân, Hà Nam Nơi ở hiện tại: Nguyễn Khánh Toàn, phường Cầu Giấy, Hà Nội Số điện thoại: 0985771235 Chúng tôi tự nguyện cùng nhau lập và ký bản hợp đồng này để thực hiện việc mua bán xe máy/xe môtô, với những điều khoản đã được hai bên bàn bạc và thoả thuận như sau: ĐIỀU 1: ĐẶC ĐIỂM XE MUA BÁN Bên bán là chủ sở hữu của chiếc xe máy/xe môtô nhãn hiệu: HONDA Loại xe: Hai Bánh, màu sơn: Trắng, số máy: F03E-0057735, số khung: 5A04F-070410, biển số đăng ký: 29P1 – 498.89 do Phòng Cảnh sát Giao thông - Công an Hà Nội cấp ngày 04/08/2000 (đăng ký lần đầu ngày 08/06/1999). ĐIỀU 2: SỰ THỎA THUẬN MUA BÁN 2.1. Bên bán đồng ý bán và Bên mua đồng ý mua chiếc xe nói trên như hiện trạng với giá là: 6.000.000 đồng (Sáu triệu đồng) và không thay đổi vì bất kỳ lý do gì. 2.2. Bên bán đã nhận đủ tiền do Bên mua trả và đã giao xe đúng như hiện trạng cho Bên mua cùng toàn bộ giấy tờ có liên quan đến chiếc xe này. Việc giao nhận không có gì vướng mắc. Việc giao tiền, giao xe được hai bên thực hiện bằng việc ký vào biên bàn bàn giao hoặc thực hiện đồng thời bằng việc ký vào hợp đồng này. 2.3. Hai bên thoả thuận: Bên mua nộp toàn bộ các loại lệ phí, thuế liên quan đến việc mua bán ô tô. ĐIỀU 3: CAM ĐOAN 3.1. Bên bán cam đoan: Khi đem bán theo bản hợp đồng này, chiếc xe nói trên thuộc quyền sở hữu và sử dụng hợp pháp của Bên bán; chưa đem cầm cố, thế chấp hoặc dùng để đảm bảo cho bất kỳ nghĩa vụ tài sản nào. 3.2. Bên mua cam đoan: Bên mua đã tự mình xem xét kỹ, biết rõ về nguồn gốc sở hữu và hiện trạng chiếc xe nói trên của Bên bán, bằng lòng mua và không có điều gì thắc mắc. ĐIỀU 4: ĐIỀU KHOẢN CUỐI CÙNG Hai bên đã tự đọc lại nguyên văn bản hợp đồng này, đều hiểu và chấp thuận toàn bộ nội dung của hợp đồng, không có điều gì vướng mắc. Hai bên cùng ký tên dưới đây để làm bằng chứng. BÊN A (Ký, ghi rõ họ và tên) BÊN B (Ký, ghi rõ họ và tên) Đỗ Văn Hưng """
)

results = predict_ner(
    model=model,
    tokenizer=tokenizer,
    sentence=test_text,
    device=device,
    max_len=256  
)

from collections import defaultdict
import re

def clean_text(text):
    text = re.sub(r'\s+([,.])', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()
    
def compact_bio_ner(bio_results):
    entities = defaultdict(list)

    current_tokens = []
    current_label = None

    for token, tag in bio_results:
        # ===== CASE O =====
        if tag == 'O':
            # cho phép dấu câu nối LOCATION
            if current_label == 'LOCATION' and token in {',', '.', '-'}:
                current_tokens.append(token)
                continue

            if current_label:
                entities[current_label].append(clean_text(" ".join(current_tokens)))
                current_tokens = []
                current_label = None
            continue

        prefix, label = tag.split('-', 1)

        # ===== CASE B =====
        if prefix == 'B':
            # LOCATION mới nhưng đang ở LOCATION → gộp
            if label == 'LOCATION' and current_label == 'LOCATION':
                current_tokens.append(token)
            else:
                if current_label:
                    entities[current_label].append(clean_text(" ".join(current_tokens)))
                current_tokens = [token]
                current_label = label

        # ===== CASE I =====
        elif prefix == 'I' and label == current_label:
            current_tokens.append(token)

    # flush entity cuối
    if current_label:
        entities[current_label].append(clean_text(" ".join(current_tokens)))

    return dict(entities)


compact_results = compact_bio_ner(results)
print(compact_results)
