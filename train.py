import torch
import random
#import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from seqeval.metrics import classification_report
#import re
import os
#import json
from tqdm import tqdm
from pyvi import ViTokenizer
import function.format as format

import re


# Cố định seed cho các thư viện
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Bước 3: Tải và chuẩn bị tokenizer PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

# Bước 4: Định nghĩa các nhãn NER
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

# Ánh xạ nhãn sang id và ngược lại
tag2id = {tag: id for id, tag in enumerate(NER_LABELS)}
id2tag = {id: tag for tag, id in tag2id.items()}

class NERDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_len=128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]

        word_ids = [] # chứa vị trí của từng token - word (hoặc subword)
        tokens = [] # là token được chuyển thành dạng id để làm input cho model
        labels = [] # nhãn dưới dạng id

        tokens.append(self.tokenizer.cls_token_id) #CLS là token đặc biệt đánh dấu bắt đầu câu, với PHOBERT là 0
        word_ids.append(None) # vì là đánh dấu đầu câu nên sẽ không thuộc từ nào trong câu
        labels.append(-100)

        for word_idx, (word, tag) in enumerate(zip(text, tags)): # duyệt qua từng word trong câu text, với danh sách tags của câu
            word_tokens = self.tokenizer.tokenize(word) # tách từng từ
            if not word_tokens: # rỗng
                continue

            token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens) # chuyển token thành id
            tokens.extend(token_ids) # tương tự như append nhưng đẩy từng phần tử trong list

            word_ids.append(word_idx) # thêm vị trí của từ vừa tách
            labels.append(tag2id[tag]) # thêm id tag tương ứng

            for _ in range(1, len(word_tokens)): # nếu 1 token được tách thành subword dạng "Vi" "ệt"
                word_ids.append(word_idx) # đánh dấu từ subword thứ hai, tức là bỏ qua sub đầu
                if tag.startswith('B-') or tag.startswith('I-'):
                    entity_type = tag[2:] # bỏ qua hai kí tự đầu -> LOC or PER, ...
                    labels.append(tag2id[f"I-{entity_type}"]) # gán thêm tiền tố I-
                else:
                    labels.append(tag2id[tag]) # là O(outside)

        tokens.append(self.tokenizer.sep_token_id) # SEP là kí hiệu kết thúc câu. PHOBERT là 2
        word_ids.append(None)
        labels.append(-100)

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            word_ids = word_ids[:self.max_len]
            labels = labels[:self.max_len]
        else:
            padding_length = self.max_len - len(tokens)
            tokens.extend([self.tokenizer.pad_token_id] * padding_length) # thêm các PAD để đúng độ dài PAD = 1
            word_ids.extend([None] * padding_length)
            labels.extend([-100] * padding_length)

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long), # chuyển thanh các tensor( mảng nhiều chiều) để đưa vào mô hình
            'attention_mask': torch.tensor([1 if t != self.tokenizer.pad_token_id else 0 for t in tokens], # đánh dấu các PAD thêm vào cho model
                                           dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def read_conll_data(file_path):
    texts = []
    tags = []

    current_words = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    word, tag = parts[0], parts[-1]
                    current_words.append(word)
                    current_tags.append(tag)
            elif current_words:
                texts.append(current_words)
                tags.append(current_tags)
                current_words = []
                current_tags = []

    if current_words:
        texts.append(current_words)
        tags.append(current_tags)
    
    return texts, tags

def normalize_text(texts, tags): # bước làm sạch, loại bỏ các dấu cách thừa, nhãn O cho word là khoảng trắng
    normalized_texts = []
    normalized_tags = []

    for sentence, sentence_tags in zip(texts, tags):
        norm_sentence = []
        norm_tags = []

        for word, tag in zip(sentence, sentence_tags):
            word = word.strip()
            if word:
                norm_sentence.append(word)
                norm_tags.append(tag)

        normalized_texts.append(norm_sentence)
        normalized_tags.append(norm_tags)
    
    return normalized_texts, normalized_tags

class PhoBERTForNER(torch.nn.Module):
    def __init__(self, num_labels):
        super(PhoBERTForNER, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-large")
        self.dropout = torch.nn.Dropout(0.3) # Dropout là một kỹ thuật điều chuẩn (regularization) giúp ngăn chặn mô hình bị 
        # overfitting (quá khớp) với dữ liệu huấn luyện.
        # Trong quá trình huấn luyện, tại mỗi bước, một số đơn vị (neuron) trong lớp này sẽ bị "tắt"
        #  (đặt giá trị đầu ra bằng 0) một cách ngẫu nhiên với xác suất là 0.3 (30%). 
        # Điều này buộc mô hình phải học các đặc trưng mạnh mẽ hơn và không phụ thuộc quá nhiều vào một số ít neuron cụ thể.
        self.classifier = torch.nn.Linear(self.phobert.config.hidden_size, num_labels)
        # hidden_size: Kích thước của vector đặc trưng đầu ra từ lớp cuối cùng của PhoBERT (thường là 768 cho phiên bản "base")
        # Mỗi token sau khi đi qua PhoBERT sẽ được biểu diễn bằng một vector có kích thước này.
        # out_featured: num_labels - số lượng đặc trưng đầu ra, là số lượng các nhãn
        self.init_weights()

    def init_weights(self): # hàm khởi tạo trọng số
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        # torch.nn.init.xavier_uniform_: Một phương pháp khởi tạo trọng số phổ biến, còn được gọi là Glorot uniform initialization. 
        # Nó giúp giữ cho phương sai của các kích hoạt (activations) và gradient ổn định qua các lớp, 
        # đặc biệt hữu ích cho các mạng sâu. Dấu gạch dưới (_) ở cuối tên hàm trong PyTorch thường chỉ ra rằng hàm này
        # thực hiện thao tác in-place (thay đổi trực tiếp dữ liệu của tensor).
        self.classifier.bias.data.fill_(0) # truy cập trực tiếp vào dữ liệu vector của tenser,
        #Điền tất cả các giá trị của bias bằng 0. Đây là một cách khởi tạo bias phổ biến.

    def forward(self, input_ids, attention_mask, labels=None):
        # input_ids là tensor của token
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # output có dạng ma trận (batch_size, seq_len, 768), batch_size là số câu, seq_size là số token trong câu ( vector ngữ nghĩa của các token)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # ánh xạ từ  vector 768 chiều thành điểm số của từng token trong từng câu cho các nhãn
        # logits có shape = (batch_size, seq_len, num_labels), 
        # trong đó Mỗi logits[i][j] là vector điểm số của token j trong câu i đối với số nhãn.

        # trong quá trình huấn luyện, labels sẽ được truyền vào để tính loss, còn val và test thì ko
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss() # khởi tạo hàm mất mát với input là vector điểm số dự đoán logits, và các nhãn thật labels 
            active_loss = labels.view(-1) != -100 # làm phẳng id labels thành 1D, so sánh với -100 để lấy True, False
            active_logits = logits.view(-1, logits.size(-1))[active_loss] # làm phẳng logit thành 1D, mỗi phần tử là vector điểm số với từng nhãn, so sánh với active_loss chỉ giữ lại các gtri True
            active_labels = labels.view(-1)[active_loss] # làm phẳng, giữ lại các nhãn thật tương ứng với token hợp lệ
            loss = loss_fct(active_logits, active_labels) # tính toán loss dựa trên 2 tham số

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=5):
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Gọi evaluate_model, giờ đây val_report chứa toàn bộ báo cáo từ seqeval
        val_report = evaluate_model(model, val_dataloader, device)
        print("Validation Metrics:")
        # Sử dụng hàm mới để in chi tiết các chỉ số
        print_detailed_metrics(val_report) 

        # Sử dụng F1-score trung bình trọng số từ báo cáo đầy đủ để chọn model tốt nhất
        current_val_f1 = val_report['weighted avg']['f1-score']
        if current_val_f1 > best_f1:
            best_f1 = current_val_f1
            best_model_state = model.state_dict().copy()
            # Thêm thông báo khi tìm thấy model tốt hơn
            print(f"New best validation F1-score: {best_f1:.4f}. Saving model state.")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def evaluate_model(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            predictions = torch.argmax(logits, dim=2)

            predictions = predictions.detach().cpu().numpy()
            true_labels = labels.detach().cpu().numpy()

            for i in range(len(true_labels)):
                pred = [id2tag[p] for p, l in zip(predictions[i], true_labels[i]) if l != -100]
                true = [id2tag[l] for l in true_labels[i] if l != -100]

                all_predictions.append(pred)
                all_true_labels.append(true)

    results = classification_report(all_true_labels, all_predictions, output_dict=True)

    # metrics = {
    #     'precision': results['weighted avg']['precision'],
    #     'recall': results['weighted avg']['recall'],
    #     'f1': results['weighted avg']['f1-score']
    # }

    return results

def tokenize_vi_full(text):
    # Bước 1: Tách từ bằng pyvi
    tokenized = ViTokenizer.tokenize(text)
    
    # Bước 2: Bỏ dấu _ và tách từng từ
    words = []
    for word in tokenized.split():
        if "_" in word:
            words.extend(word.split("_"))
        else:
            words.append(word)
    
    # Bước 3: Tách dấu câu thành token riêng
    tokens = []
    for word in words:
        # Tách các dấu câu ra khỏi từ (giữ lại dấu câu)
        tokens.extend(re.findall(r"\w+|[^\w\s]", word, re.UNICODE))
    
    return tokens

def predict_ner(model, tokenizer, sentence, device, max_len=1024):
    model.eval()

    words = tokenize_vi_full(sentence)

    features = [tokenizer.cls_token_id]
    word_ids = [None]

    for word_idx, word in enumerate(words):
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            print(f"Warning: Word '{word}' could not be tokenized")
            continue
        token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
        features.extend(token_ids)
        word_ids.extend([word_idx] * len(token_ids))

    features.append(tokenizer.sep_token_id)
    word_ids.append(None)

    if len(features) > max_len:
        print(f"Warning: Input sentence truncated to {max_len} tokens")
        features = features[:max_len]
        word_ids = word_ids[:max_len]

    attention_mask = [1] * len(features)
    padding_length = max_len - len(features)
    if padding_length > 0:
        features.extend([tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        word_ids.extend([None] * padding_length)

    input_ids = torch.tensor([features], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]

    results = []
    prev_word_idx = None
    token_tags = []

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            if prev_word_idx is not None:
                tag = token_tags[0]
                results.append((words[prev_word_idx], tag))
            token_tags = [id2tag[predictions[token_idx]]]
            prev_word_idx = word_idx
        else:
            token_tags.append(id2tag[predictions[token_idx]])

    if prev_word_idx is not None:
        tag = token_tags[0]
        results.append((words[prev_word_idx], tag))

    return results

def validate_bio_tags(tags):
    for sentence_tags in tags:
        for i, tag in enumerate(sentence_tags):
            if tag.startswith('I-'):
                if i == 0 or not (sentence_tags[i-1].startswith('B-') or sentence_tags[i-1].startswith('I-')):
                    print(f"Invalid BIO sequence at {sentence_tags}")
                    return False
    return True

def print_detailed_metrics(report_dict):
    """
    In các chỉ số chi tiết bao gồm precision, recall, F1-score, và support cho từng loại thực thể,
    cũng như trung bình trọng số.
    """
    print(f"  Overall (Weighted Avg):")
    print(f"    Precision: {report_dict['weighted avg']['precision']:.4f}")
    print(f"    Recall:    {report_dict['weighted avg']['recall']:.4f}")
    print(f"    F1-score:  {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"    Support:   {report_dict['weighted avg']['support']}")
    print("  -----------------------------------")
    print("  Per Entity Type:")
    # Duyệt qua các nhãn (ví dụ: 'PER', 'LOC') có trong báo cáo
    # Loại trừ các key tổng hợp đã in hoặc không liên quan trực tiếp đến từng loại thực thể
    for label, metrics in report_dict.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']: 
            print(f"  Entity: {label}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-score:  {metrics['f1-score']:.4f}")
            print(f"    Support:   {metrics['support']}")
    print("  -----------------------------------")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_file = "data/train.conll"
    val_file = "data/val.conll"
    test_file = "data/test.conll"

    print("Đọc dữ liệu...")
    train_texts, train_tags = read_conll_data(train_file)
    assert validate_bio_tags(train_tags), "Invalid BIO tags in train data"
    val_texts, val_tags = read_conll_data(val_file)
    test_texts, test_tags = read_conll_data(test_file)

    print(f"Số lượng dữ liệu - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    print("Chuẩn hóa dữ liệu...")
    train_texts, train_tags = normalize_text(train_texts, train_tags)
    val_texts, val_tags = normalize_text(val_texts, val_tags)
    test_texts, test_tags = normalize_text(test_texts, test_tags)

    print("Chuẩn bị dữ liệu...")
    train_dataset = NERDataset(train_texts, train_tags, tokenizer)
    val_dataset = NERDataset(val_texts, val_tags, tokenizer)
    test_dataset = NERDataset(test_texts, test_tags, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    print("Khởi tạo mô hình PhoBERT for NER...")
    model = PhoBERTForNER(num_labels=len(NER_LABELS))
    model.to(device)

    print("Bắt đầu huấn luyện...")
    model = train_model(model, train_dataloader, val_dataloader, device, num_epochs=10)

    print("Đánh giá mô hình trên tập test...")
    # test_metrics = evaluate_model(model, test_dataloader, device)
    # Gọi evaluate_model, giờ đây test_report chứa toàn bộ báo cáo từ seqeval
    test_report = evaluate_model(model, test_dataloader, device) 
    print("Test Results:")
    # Sử dụng hàm mới để in chi tiết các chỉ số
    print_detailed_metrics(test_report) 

    model_save_path = "models/phobert_ner_model.pth"
    print(f"Lưu mô hình vào {model_save_path}...") # Thêm đường dẫn vào thông báo
    torch.save(model.state_dict(), model_save_path)
    print(f"Mô hình đã được lưu vào {model_save_path}") # Thêm thông báo xác nhận


if __name__ == "__main__":
    main() 