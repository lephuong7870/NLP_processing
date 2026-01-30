import torch
from transformers import AutoTokenizer
from train import PhoBERTForNER, NER_LABELS, predict_ner

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

for word, tag in results:
    print(f"{word:15s} -> {tag}")