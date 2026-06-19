# Mispronunciation Detection & Diagnosis (MDD) for Vietnamese Speech

Dự án phát triển hệ thống phát hiện và chẩn đoán lỗi phát âm (Mispronunciation Detection & Diagnosis - MDD) cho tiếng Việt, dựa trên kiến trúc **Phonetic-Linguistic (PL) Cross-Attention Model** kết hợp với mô hình tự giám sát **Wav2Vec2** tiền huấn luyện.

---

## 📌 Tổng quan dự án

Mục tiêu của bài toán MDD là:
- Nhận đầu vào là một file âm thanh (`.wav`) người đọc phát âm một câu tiếng Việt và câu gốc chuẩn tương ứng (canonical transcription).
- Phát hiện xem người dùng có phát âm sai âm vị (phoneme) nào không, và nếu sai thì họ đã phát âm lệch sang âm vị nào (chẩn đoán lỗi).

Mô hình sử dụng backbone là `nguyenvulebinh/wav2vec2-base-vietnamese-250h` kết hợp với nhánh mã hóa ngôn ngữ (Linguistic Encoder) mã hóa câu chuẩn (canonical), kết hợp chúng thông qua cơ chế Cross-Attention và Gated Fusion để dự đoán chuỗi âm vị thực tế được nói (transcript), từ đó đối chiếu tìm ra các lỗi phát âm.

---

## 📂 Cấu trúc thư mục

```text
├── MDD-Challenge-2025-training-set/  # Thư mục dữ liệu huấn luyện (cần tải về và giải nén tại đây)
│   ├── audio_data/
│   │   └── train/                     # Chứa 3180 file .wav huấn luyện
│   └── metadata/
│       ├── train.csv                  # Metadata text thông thường
│       ├── train_phones.csv           # Metadata dạng chuỗi âm vị (phonemes)
│       └── lexicon_vmd.txt            # Từ điển phát âm từ sang âm vị
│
├── MDD-Challenge-2025-private-test/  # Thư mục dữ liệu test private (dùng để tạo file submission)
│   ├── audio_data/                    # Các file audio test (.wav)
│   └── metadata/
│       ├── private_test_submission.csv
│       └── private_test_submission_example.csv
│
├── MDD-Metrics/                       # Mã nguồn chứa script đánh giá từ BTC
│   ├── evaluate.py                    # Script tính toán F1, DER, PER
│   └── README.md
│
├── src/                               # Mã nguồn chính của dự án
│   ├── config.py                      # Cấu hình siêu tham số (hyperparameters)
│   ├── data.py                        # Pipeline xử lý dữ liệu, augmentation và DataLoader
│   ├── model.py                       # Định nghĩa mô hình PLModel (Cross-Attention + Gated Fusion)
│   ├── train.py                       # Vòng lặp huấn luyện chính và kiểm thử
│   ├── eval.py                        # Dự đoán trên tập kiểm thử và xuất kết quả
│   ├── inference.py                   # Script chạy dự đoán trên dữ liệu test mới (ví dụ tập public/private test)
│   └── main.py                        # File thực thi luồng chính (Train -> Eval)
│
├── requirements.txt                   # Các thư viện Python cần cài đặt
├── AGENTS.md                          # Tài liệu mô tả nhanh về bài toán và luật chơi
└── README.md                          # Tài liệu hướng dẫn này
```

---

## 🛠️ Hướng dẫn cài đặt từ đầu

### 1. Chuẩn bị môi trường
Khuyến nghị sử dụng Python 3.8+ và khởi tạo một môi trường ảo (virtual environment) để tránh xung đột thư viện:

```bash
# Tạo môi trường ảo với venv
python3 -m venv venv
source venv/bin/activate

# Hoặc nếu dùng conda:
# conda create -n mdd python=3.9 -y
# conda activate mdd
```

### 2. Cài đặt các thư viện phụ thuộc
Cài đặt tất cả các gói cần thiết trong `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Lưu ý: Đối với việc huấn luyện bằng GPU (Khuyến nghị dùng GPU T4 trở lên như trên Colab/Kaggle), hãy đảm bảo phiên bản PyTorch của bạn tương thích với phiên bản CUDA hiện tại trên máy.*

---

## 📊 Chuẩn bị dữ liệu

Đảm bảo cấu trúc dữ liệu huấn luyện nằm đúng vị trí quy định trong file cấu hình (`src/config.py`):
1. Đặt thư mục dữ liệu huấn luyện tại `MDD-Challenge-2025-training-set/`.
2. Kiểm tra xem các file âm thanh có nằm trong `MDD-Challenge-2025-training-set/audio_data/train/` hay không.
3. File nhãn chính dạng phoneme phải nằm tại `MDD-Challenge-2025-training-set/metadata/train_phones.csv`.

**Định dạng dữ liệu phoneme:**
Mỗi dòng trong tập dữ liệu chứa:
- `canonical`: Chuỗi âm vị chuẩn mong đợi (ví dụ: `ɓ aː-0 $ ɗ aː-2 ...`).
- `transcript`: Chuỗi âm vị thực tế người đọc phát âm (có thể có lỗi sai).
- Kí tự `$` đại diện cho khoảng trắng phân tách giữa các từ và sẽ được tự động xử lý khi căn hàng (alignment).

---

## 🏗️ Kiến trúc mô hình: PL Model (Phonetic-Linguistic)

Mô hình được hiện thực hóa trong `src/model.py` với các thành phần chính:
1. **Wav2Vec2 Backbone**: Trích xuất đặc trưng âm học thô từ audio (`nguyenvulebinh/wav2vec2-base-vietnamese-250h`).
2. **Phonetic Encoder**: Gồm các lớp Conv1D để nén chiều thời gian (stride=2) và mạng BiLSTM để nắm bắt thông tin ngữ cảnh âm thanh hai chiều.
3. **Linguistic Encoder**: Mã hóa chuỗi canonical phonemes sử dụng Embedding lớp tích hợp Sinsuoidal Positional Encoding cùng BiLSTM để mô hình hóa thứ tự và thông tin của chuỗi âm vị chuẩn.
4. **Cross-Attention Block**: Dùng các đặc trưng âm học làm Query ($Q$) truy vấn thông tin chuẩn hóa từ Linguistic Encoder làm Key/Value ($K, V$).
5. **Gated Fusion**: Cơ chế cổng học cách cân bằng động giữa thông tin âm thanh thực tế nhận được và thông tin chuẩn từ câu mẫu (canonical text) dựa trên từng bước thời gian.
6. **Weighted CTC Loss**: Tăng trọng số loss đối với những phân khúc phát âm lỗi nhằm cải thiện độ nhạy phát hiện sai âm vị.

---

## 🚀 Huấn luyện mô hình

Chạy toàn bộ quy trình xây dựng bộ từ vựng (Vocabulary), chia dữ liệu (Train/Val/Test), huấn luyện và đánh giá bằng cách chạy file `src/main.py`:

```bash
python src/main.py
```

### Quá trình thực hiện của `src/main.py`:
1. **Trích xuất từ vựng**: Tạo file `outputs/vocab.json` chứa danh sách tất cả các phonemes.
2. **Chia dữ liệu**: Chia tập dữ liệu huấn luyện ban đầu thành 3 phần: Train (80%), Validation (10%), Test (10%) dựa trên `seed` được cấu hình.
3. **Huấn luyện**:
   - Sử dụng kỹ thuật Speed Perturbation và Gaussian Noise để tăng cường dữ liệu âm thanh (Audio Augmentation).
   - Huấn luyện mô hình với tối ưu hóa AdamW, cơ chế học tập Warmup và Gradient Accumulation.
   - Mô hình Wav2Vec2 backbone sẽ được khóa (freeze) ở các epoch đầu và tự động mở khóa (unfreeze) để tinh chỉnh toàn diện (fine-tune) theo cấu hình `unfreeze_epoch`.
   - Lưu checkpoint tốt nhất dựa trên Validation Loss về `outputs/checkpoints/model.pt`.
4. **Đánh giá**:
   - Chạy dự đoán trên tập kiểm thử (Test split) bằng thuật toán CTC Beam Search hoặc Greedy.
   - Tự động gọi script đánh giá `MDD-Metrics/evaluate.py` để tính các chỉ số F1, DER, PER.

---

## 📈 Đánh giá mô hình thủ công

Nếu bạn đã có file dự đoán và file ground truth tương ứng, bạn có thể chạy độc lập script đánh giá của ban tổ chức:

```bash
python MDD-Metrics/evaluate.py outputs/ground_truth.csv outputs/predictions.csv
```

**Các chỉ số đo lường:**
- **F1**: Đo lường độ chính xác phát hiện lỗi phát âm.
- **DER (Diagnosis Error Rate)**: Tỉ lệ chẩn đoán sai lỗi (mô hình đoán sai âm vị thay thế khi phát hiện đúng từ đó bị phát âm sai).
- **PER (Phoneme Error Rate)**: Tỉ lệ lỗi âm vị tổng thể giữa thực tế nói và dự đoán từ mô hình.

---

## 🎯 Chạy dự đoán trên tập dữ liệu Private Test và Nộp bài (Submission)

Để tạo dự đoán trên tập Private Test (hoặc Public Test) phục vụ nộp bài lên AIHub, sử dụng script `src/inference.py`:

```bash
python src/inference.py \
  --test_csv MDD-Challenge-2025-private-test/metadata/private_test_submission.csv \
  --audio_dir MDD-Challenge-2025-private-test/audio_data \
  --checkpoint outputs/checkpoints/model.pt \
  --vocab outputs/vocab.json \
  --processor outputs/processor \
  --output results.csv \
  --beam_width 10
```

### Các đối số chính của `inference.py`:
- `--test_csv`: Đường dẫn tới file csv chứa danh sách test (chứa cột `id` và `path`).
- `--audio_dir`: Thư mục gốc chứa các file audio test.
- `--checkpoint`: Đường dẫn tới checkpoint mô hình đã huấn luyện xong.
- `--vocab`: Đường dẫn tới file từ vựng `vocab.json`.
- `--processor`: Thư mục chứa cấu hình processor của Wav2Vec2.
- `--output`: Tên file csv kết quả xuất ra (ví dụ: `results.csv`).
- `--beam_width`: Độ rộng chùm cho CTC Beam Search decoding (thiết lập `=1` để giải mã Greedy nhanh hơn).


## ⚙️ Tùy chỉnh tham số huấn luyện

Bạn có thể chỉnh sửa các tham số huấn luyện trực tiếp trong file `src/config.py` để tối ưu kết quả:
- `learning_rate`: Tốc độ học (mặc định: `1e-4`).
- `batch_size`: Kích thước batch (mặc định: `8` để phù hợp với GPU T4 16GB).
- `gradient_accumulation`: Tích lũy gradient (mặc định: `2`).
- `num_epochs`: Số lượng epochs chạy huấn luyện (mặc định: `20`).
- `oversample_errors`: Bật/tắt chế độ oversampling các mẫu phát âm sai để cân bằng dữ liệu (mặc định: `True`).
- `weighted_ctc`: Bật/tắt tính trọng số loss lớn hơn cho các phân đoạn phát âm sai (mặc định: `True`).
- `beam_width`: Thiết lập độ rộng chùm trong giải mã CTC Beam Search tại thời điểm test/inference (mặc định: `10`).
