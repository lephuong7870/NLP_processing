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
            "EMAIL": {
                "patterns": [r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"],
                "triggers": ["email", "thư điện tử", "gmail"]
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
                "triggers": ["bảo hiểm y tế", "bhy"]
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
        
        # ===== ĐỊA CHỈ & LIÊN LẠC =====
        "LOCATION": {
            "ADDRESS": {
                "patterns": [r"\b\d{1,3}[/-]?\d*\s*[a-zA-ZÀ-ỹ\s]+"],
                "triggers": ["địa chỉ", "số nhà", "đường", "phố"]
            },
            "POSTAL_CODE": {
                "patterns": [r"\b\d{5,6}\b"],
                "triggers": ["mã bưu điện", "postal code", "zip code"]
            }
        },
        
        # ===== THỜI GIAN =====
        "TIME": {
            "DATE": {
                "patterns": [r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", r"\b\d{2}/\d{2}/\d{4}\b"],
                "triggers": ["ngày", "date", "ngày sinh", "ngày cấp"]
            },
            "TIME": {
                "patterns": [r"\b\d{1,2}:\d{2}(?::\d{2})?\b"],
                "triggers": ["giờ", "time", "thời gian"]
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
        
        # ===== INTERNET & CÔNG NGHỆ =====
        "TECH": {
            "URL": {
                "patterns": [r"https?://[^\s]+", r"www\.[^\s]+"],
                "triggers": ["website", "trang web", "url", "link"]
            },
            "IP_ADDRESS": {
                "patterns": [r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"],
                "triggers": ["ip", "địa chỉ ip", "ip address"]
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
        
        # Duyệt qua tất cả các loại entity
        for category, entities in self.entity_config.items():
            for label, config in entities.items():
                patterns = config["patterns"]
                triggers = config["triggers"]
                
                # Kiểm tra triggers
                if not any(trigger in text_lower for trigger in triggers):
                    continue
                
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        # Kiểm tra context
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
        
        # Lọc và sắp xếp
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

# Test với dữ liệu phức tạp
complex_document = """
HỢP ĐỒNG MUA BÁN XE Ô TÔ

Bên A: Công ty TNHH Thương Mại XYZ
MST: 0312345678
Địa chỉ: Số 123 Đường Lê Lợi, Quận 1, TP.HCM
Điện thoại: 028 3823 4567
Email: contact@xyzcompany.vn
Website: https://xyzcompany.vn

Bên B: Ông Nguyễn Văn A
CCCD: 025123456789
Địa chỉ: 45/12 Nguyễn Du, Quận 3
SĐT: 0912345678
Email: nguyenvana@gmail.com

THÔNG TIN XE:
Biển số: 51A-12345
Số khung: 1HGCM82633A123456
Số máy: ABC123456789
Ngày đăng ký: 15/08/2020

THANH TOÁN:
Số tiền: 850.000.000 VND
STK: 1234567890 tại Ngân hàng ABC
Số thẻ: 1234 5678 9012 3456

Mã hợp đồng: HD202300123
Ngày ký: 20/10/2023
"""

doc = nlp(complex_document)

print("PHÂN TÍCH VĂN BẢN PHÁP LÝ")
print("=" * 80)
print(complex_document)
print("\n" + "=" * 80)
print("CÁC THỰC THỂ PHÁT HIỆN:")
print("=" * 80)

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

# Hiển thị theo nhóm
for category, entities in entities_by_category.items():
    print(f"\n{category}:")
    print("-" * 40)
    for ent in entities:
        print(f"  {ent.label_:25} → {ent.text}")
