import torch
from transformers import AutoTokenizer
from train import PhoBERTForNER, NER_LABELS, predict_ner
import function.format as format
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
NER_LABELS = [ 
    'B-ADDRESS','I-ADDRESS',
    'B-SKILL','I-SKILL',
    'B-EMAIL','I-EMAIL',
    'B-PERSON','I-PERSON',
    'B-PHONENUMBER','I-PHONENUMBER',
    'B-MISCELLANEOUS','I-MISCELLANEOUS',
    'B-QUANTITY','I-QUANTITY',
    'B-PERSONTYPE','I-PERSONTYPE',
    'B-ORGANIZATION','I-ORGANIZATION',
    'B-PRODUCT','I-PRODUCT',
    'B-IP','I-IP',
    'B-LOCATION','I-LOCATION',
    'B-DATETIME','I-DATETIME',
    'B-EVENT','I-EVENT',
    'B-URL','I-URL',
    'O'
]


LABELS2ID = {
    'B-ADDRESS': 0,
    'I-ADDRESS': 1,
    'B-SKILL': 2,
    'I-SKILL': 3,
    'B-EMAIL': 4,
    'I-EMAIL': 5,
    'B-PERSON': 6,
    'I-PERSON': 7,
    'B-PHONENUMBER': 8,
    'I-PHONENUMBER': 9,
    'B-MISCELLANEOUS': 10,
    'I-MISCELLANEOUS': 11,
    'B-QUANTITY': 12,
    'I-QUANTITY': 13,
    'B-PERSONTYPE': 14,
    'I-PERSONTYPE': 15,
    'B-ORGANIZATION': 16,
    'I-ORGANIZATION': 17,
    'B-PRODUCT': 18,
    'I-PRODUCT': 19,
    'B-IP': 20,
    'I-IP': 21,
    'B-LOCATION': 22,
    'I-LOCATION': 23,
    'B-DATETIME': 24,
    'I-DATETIME': 25,
    'B-EVENT': 26,
    'I-EVENT': 27,
    'B-URL': 28,
    'I-URL': 29,
    'O': 30
}

ID2LABELS = {v: k for k, v in LABELS2ID.items()}


# load model
model = PhoBERTForNER(num_labels=len(NER_LABELS))
model.load_state_dict(
    torch.load("models/phobert_ner_model.pth", map_location=device)
)
model.to(device)
model.eval()



test_text = "Nguyễn Văn A ( Nam ) là giám đốc Công ty TNHH ABC tại Hà Nội vào ngày hôm qua ( tức 10 tháng 5 năm 2023 ), hiện đang là bệnh nhân BN002 tại bệnh viện Ung bướu Trung ương, biểu hiện ho, sốt cao, đang được điều trị tại khoa Hồi sức số 1, được vận chuyển bằng xe cứu thương."
tagged_words = predict_ner(     model=model,
                                tokenizer=tokenizer,
                                sentence=test_text,
                                device=device,
                                max_len=256    )

extractor = format.DateExtractor()
results, changes = extractor.process_ner_results(tagged_words)

print("\nKết quả nhận diện thực thể:")
print("Tagged words:", results)

print("\nSau khi định dạng:")
print(changes)
