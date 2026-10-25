# Quy Trình Các Bước Feature Engineering

Feature Engineering (Kỹ nghệ đặc trưng) là quá trình biến đổi, làm sạch và tạo mới các đặc trưng (features) từ dữ liệu thô nhằm giúp các mô hình Machine Learning học hỏi và dự báo chính xác hơn.

---

## Bước 1. Xử Lý Giá Trị Thiếu (Handling Missing Values)

### Các nội dung chi tiết:
*   **Xóa bỏ dữ liệu (Deletion):** Loại bỏ hoàn toàn hàng hoặc cột có tỷ lệ dữ liệu thiếu quá cao (thường > 50-60%) khi thông tin đó không đóng vai trò cốt lõi và không thể tự khôi phục.
*   **Điền khuyết biến số (Numerical Imputation):**
    *   **Mean Imputation:** Điền bằng giá trị trung bình (chỉ dùng khi dữ liệu phân phối chuẩn, đối xứng và không có outliers).
    *   **Median Imputation:** Điền bằng giá trị trung vị (rất an toàn cho dữ liệu phân phối lệch hoặc chứa nhiều outliers).
    *   **Arbitrary Imputation:** Điền bằng một giá trị cố định đặc biệt (ví dụ: -999 hoặc 9999) để mô hình nhận biết sự khuyết thiếu.
*   **Điền khuyết biến phân loại (Categorical Imputation):**
    *   **Mode Imputation:** Điền bằng giá trị xuất hiện nhiều nhất trong tập dữ liệu.
    *   **Missing Label:** Tạo thêm một nhãn mới hoàn toàn như "Unknown" hoặc "Missing" để thay thế cho ô trống.
*   **Thêm biến chỉ thị khuyết thiếu (Missing Indicator):** Tạo thêm cột phụ kiểu Boolean (True/False) nhằm báo hiệu cho mô hình biết điểm dữ liệu nào từng bị thiếu, giúp giữ lại thông tin về sự thiếu hụt.

### Phân tích 5W1H:
*   **What (Là gì?):** Quá trình điền khuyết (imputation) các giá trị thiếu hoặc loại bỏ các dòng/cột chứa dữ liệu khuyết (`NaN`).
*   **Why (Tại sao cần làm?):** Phần lớn các mô hình học máy (ngoại trừ một số thuật toán dạng cây như XGBoost, LightGBM) không thể hoạt động hoặc sẽ báo lỗi nếu đầu vào chứa giá trị khuyết thiếu.
*   **Where (Thực hiện ở đâu?):** Trên các cột thuộc tính chứa giá trị khuyết thiếu đã được nhận diện ở bước EDA.
*   **When (Khi nào thực hiện?):** Thực hiện ngay đầu giai đoạn Feature Engineering, và bắt buộc phải thực hiện sau khi đã chia tập dữ liệu thành Train/Test.
*   **Which (Sử dụng công cụ/hàm nào?):** Các class tiền xử lý của Scikit-Learn như `SimpleImputer` (điền Mean, Median, Mode), `KNNImputer`, `IterativeImputer` (MICE), hoặc các hàm Pandas như `.fillna()`, `.dropna()`.
*   **How (Thực hiện như thế nào?):** Nếu cột số có phân phối chuẩn, điền bằng Mean; nếu lệch hoặc có outlier, điền bằng Median. Đối với cột phân loại, điền bằng Mode hoặc tạo nhãn mới "Unknown". Thực hiện `.fit()` trên tập Train trước rồi áp dụng `.transform()` lên tập Test để tránh rò rỉ dữ liệu (Data Leakage).

---

## Bước 2. Xử Lý Giá Trị Ngoại Lệ (Handling Outliers)

### Các nội dung chi tiết:
*   **Cắt tỉa (Trimming):** Loại bỏ hoàn toàn các hàng dữ liệu chứa giá trị ngoại lệ nếu số lượng nhỏ và không làm suy giảm đáng kể kích thước tập huấn luyện.
*   **Giới hạn biên (Winsorization/Capping):** Thay thế các giá trị vượt ngoài phạm vi bình thường (thường dựa trên $1.5 \times IQR$ hoặc Z-score) bằng giá trị biên lớn nhất hoặc nhỏ nhất được chấp nhận.
*   **Biến đổi toán học (Mathematical Transformations):** Sử dụng các hàm toán học như Logarithm, Căn bậc hai (Square Root) hoặc Box-Cox để nén các giá trị cực lớn, làm giảm tác động của outliers và đưa phân phối dữ liệu về gần phân phối chuẩn.

### Phân tích 5W1H:
*   **What (Là gì?):** Quá trình xử lý các điểm dữ liệu dị biệt (outliers) bằng cách loại bỏ, giới hạn biên hoặc biến đổi chúng.
*   **Why (Tại sao cần làm?):** Outliers có thể làm lệch nghiêm trọng đường biên quyết định hoặc các hệ số trọng số của các mô hình nhạy cảm với khoảng cách (như Linear Regression, Logistic Regression, KNN, SVM, Mạng nơ-ron).
*   **Where (Thực hiện ở đâu?):** Trên các cột thuộc tính số chứa các điểm dị biệt đã xác định trong bước EDA.
*   **When (Khi nào thực hiện?):** Sau khi đã điền khuyết các giá trị thiếu ở Bước 1.
*   **Which (Sử dụng công cụ/hàm nào?):** Phương pháp cắt tỉa (Trimming), phương pháp giới hạn biên (Winsorization/Capping) dựa trên IQR hoặc Z-score, hoặc sử dụng các hàm biến đổi như Logarithm (`np.log`).
*   **How (Thực hiện như thế nào?):** Nếu số lượng outlier cực kỳ ít, ta xóa dòng chứa outlier. Nếu số lượng nhiều, áp dụng giới hạn biên (capping): thay các giá trị vượt ngoài biên trên bằng giá trị biên trên và vượt biên dưới bằng giá trị biên dưới. Hoặc sử dụng Log transform để nén dữ liệu.

---

## Bước 3. Mã Hóa Biến Phân Loại (Categorical Encoding)

### Các nội dung chi tiết:
*   **Mã hóa One-Hot (One-Hot Encoding):** Chuyển đổi mỗi danh mục thành một cột nhị phân mới (0 hoặc 1). Phù hợp cho các biến không có thứ tự và số lượng danh mục ít.
*   **Mã hóa Thứ tự (Ordinal Encoding):** Gán các giá trị phân loại thành các số nguyên theo thứ tự tăng dần hoặc giảm dần dựa trên một logic thứ cấp (ví dụ: mức độ hài lòng, học vị).
*   **Mã hóa theo Biến mục tiêu (Target Encoding / Mean Encoding):** Thay thế mỗi danh mục bằng giá trị trung bình của biến mục tiêu tương ứng với danh mục đó. Thường dùng cho các biến phân loại có rất nhiều nhóm khác nhau (High Cardinality).
*   **Mã hóa theo Tần suất (Frequency Encoding):** Thay thế danh mục bằng tần suất hoặc tỷ lệ phần trăm xuất hiện của nó trong bộ dữ liệu.

### Phân tích 5W1H:
*   **What (Là gì?):** Quá trình chuyển đổi các cột thuộc tính dạng chữ hoặc danh mục (categorical) thành các con số tương ứng.
*   **Why (Tại sao cần làm?):** Mô hình học máy thực chất là các phép toán số học phức tạp, chúng chỉ hiểu và tính toán được trên các con số, không thể làm việc trực tiếp trên văn bản hay ký tự chuỗi.
*   **Where (Thực hiện ở đâu?):** Trên toàn bộ các cột dữ liệu phân loại (dạng chuỗi văn bản).
*   **When (Khi nào thực hiện?):** Sau khi các biến số và giá trị khuyết thiếu đã được làm sạch cơ bản.
*   **Which (Sử dụng công cụ/hàm nào?):** `OneHotEncoder` (mã hóa một-nóng), `OrdinalEncoder` (mã hóa thứ tự), `TargetEncoder` hoặc `LabelEncoder` của Scikit-Learn.
*   **How (Thực hiện như thế nào?):** Áp dụng One-Hot Encoding cho các biến không có thứ tự và số lượng nhóm ít; áp dụng Ordinal Encoding cho các biến có cấp độ/thứ tự tự nhiên; áp dụng Target Encoding cho các biến có quá nhiều nhóm (như Zipcode, Tên thành phố) để tránh bùng nổ số lượng cột dữ liệu.

---

## Bước 4. Biến Đổi & Chuẩn Hóa Biến Số (Numerical Transformation & Scaling)

### Các nội dung chi tiết:
*   **Chuẩn hóa Z-Score (Standardization):** Biến đổi đặc trưng số để có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1. Thích hợp cho phần lớn các mô hình tuyến tính, SVM, hoặc mạng nơ-ron.
*   **Chuẩn hóa Min-Max (Normalization):** Đưa toàn bộ giá trị đặc trưng về một khoảng cố định (thường là từ 0 đến 1). Giúp bảo toàn các giá trị bằng 0 trong ma trận thưa thớt (sparse matrix).
*   **Chuẩn hóa Robust (Robust Scaling):** Sử dụng trung vị (Median) và khoảng phân vị (IQR) để chuẩn hóa dữ liệu, giúp hạn chế sự ảnh hưởng tiêu cực của các giá trị ngoại lệ.

### Phân tích 5W1H:
*   **What (Là gì?):** Quá trình điều chỉnh thang đo của các đặc trưng số để đưa chúng về cùng một khoảng giá trị tiêu chuẩn.
*   **Why (Tại sao cần làm?):** Tránh việc các thuộc tính có biên độ giá trị lớn (ví dụ: thu nhập hàng triệu) lấn át các thuộc tính có biên độ nhỏ (ví dụ: độ tuổi từ 1-100) trong các mô hình tính khoảng cách hoặc tối ưu hóa độ dốc gradient.
*   **Where (Thực hiện ở đâu?):** Thực hiện trên tất cả các cột dữ liệu dạng số (numerical features).
*   **When (Khi nào thực hiện?):** Sau khi đã thực hiện mã hóa các biến phân loại thành số ở Bước 3.
*   **Which (Sử dụng công cụ/hàm nào?):** `StandardScaler` (chuẩn hóa Z-score), `MinMaxScaler` (chuẩn hóa về khoảng [0, 1]), `RobustScaler` (chuẩn hóa dựa trên IQR).
*   **How (Thực hiện như thế nào?):** Áp dụng `StandardScaler` để đưa dữ liệu về dạng trung bình = 0, độ lệch chuẩn = 1; áp dụng `MinMaxScaler` khi mô hình yêu cầu dữ liệu nằm trong khoảng cố định (như Mạng nơ-ron hoặc xử lý ảnh); áp dụng `RobustScaler` nếu dữ liệu của bạn vẫn còn nhiều outlier chưa loại bỏ hết.

---

## Bước 5. Tạo Đặc Trưng Mới (Feature Creation / Generation)

### Các nội dung chi tiết:
*   **Tương tác số học (Arithmetic Combinations):** Thực hiện cộng, trừ, nhân, chia giữa các cột số để tạo ra chỉ số mới mang tính đại diện cao hơn (ví dụ: tính tỷ lệ nợ trên thu nhập từ hai cột nợ và thu nhập).
*   **Trích xuất thời gian (Datetime Extraction):** Phân rã dữ liệu ngày tháng thành các thông tin chi tiết hơn như năm, tháng, ngày, giờ, thứ trong tuần, hoặc phân biệt ngày làm việc và ngày cuối tuần.
*   **Tạo thuộc tính tích (Interaction Features):** Nhân hai hoặc nhiều thuộc tính độc lập để tạo ra biến tương tác thể hiện hiệu ứng kết hợp đối với biến mục tiêu.
*   **Gom nhóm và thống kê (Aggregation Features):** Tính toán giá trị trung bình, lớn nhất, nhỏ nhất của một thuộc tính theo từng nhóm phân loại cụ thể.

### Phân tích 5W1H:
*   **What (Là gì?):** Quá trình thiết kế và tạo thêm các thuộc tính mới bằng cách kết hợp toán học, logic nghiệp vụ hoặc phân rã các thuộc tính cũ.
*   **Why (Tại sao cần làm?):** Cung cấp các thông tin cô đọng, giúp mô hình học được các mối liên hệ phức tạp hoặc phi tuyến tính mà các đặc trưng đơn lẻ thô ban đầu chưa thể hiện rõ.
*   **Where (Thực hiện ở đâu?):** Trên các cột dữ liệu hiện tại có tiềm năng kết hợp hoặc trích xuất (ngày tháng, vị trí địa lý, các chỉ số tài chính).
*   **When (Khi nào thực hiện?):** Sau khi dữ liệu thô ban đầu đã được làm sạch và chuẩn hóa.
*   **Which (Sử dụng công cụ/hàm nào?):** Các phép toán số học cộng trừ nhân chia, trích xuất chuỗi thời gian, gom nhóm và tính toán thống kê (Aggregation).
*   **How (Thực hiện như thế nào?):** Tạo cột mới bằng cách chia tỷ lệ (ví dụ: diện tích chia cho số phòng); phân rã cột ngày tháng thành ngày trong tuần, tháng, quý; tạo biến tương tác giữa hai thuộc tính có mối liên kết mạnh.

---

## Bước 6. Lựa Chọn Đặc Trưng (Feature Selection)

### Các nội dung chi tiết:
*   **Phương pháp lọc (Filter Methods):** Đánh giá độc lập từng đặc trưng bằng các chỉ số thống kê (hệ số tương quan, kiểm định Chi-Square, ANOVA, Mutual Information) và loại bỏ những biến ít liên quan hoặc bị trùng lặp.
*   **Phương pháp bọc (Wrapper Methods):** Lựa chọn đặc trưng thông qua việc thử nghiệm huấn luyện mô hình nhiều lần và loại bỏ/thêm dần các đặc trưng (RFE - loại bỏ đặc trưng đệ quy, Forward/Backward Selection).
*   **Phương pháp nhúng (Embedded Methods):** Sử dụng các mô hình có khả năng tự đánh giá độ quan trọng của đặc trưng trong quá trình huấn luyện như hồi quy Lasso (L1 regularization), hoặc các mô hình dạng cây quyết định (Random Forest, XGBoost).

### Phân tích 5W1H:
*   **What (Là gì?):** Quá trình đánh giá và chỉ giữ lại những đặc trưng quan trọng nhất đối với việc dự báo, loại bỏ các đặc trưng thừa hoặc gây nhiễu cho mô hình.
*   **Why (Tại sao cần làm?):** Giảm thiểu hiện tượng quá khớp (overfitting), rút ngắn thời gian huấn luyện mô hình, tiết kiệm tài nguyên tính toán và làm cho mô hình dễ giải thích hơn đối với con người.
*   **Where (Thực hiện ở đâu?):** Thực hiện trên toàn bộ danh sách các đặc trưng hiện có (sau khi đã thực hiện xong các bước tạo mới).
*   **When (Khi nào thực hiện?):** Là bước cuối cùng của quy trình Feature Engineering ngay trước khi đưa dữ liệu vào huấn luyện mô hình chính thức.
*   **Which (Sử dụng công cụ/hàm nào?):** Các kiểm định thống kê (Chi-Square, ANOVA, Mutual Information), ma trận tương quan để loại đa cộng tuyến, phương pháp bọc `RFE` (Recursive Feature Elimination), regularization L1 (Lasso), hoặc trích xuất độ quan trọng của đặc trưng (Feature Importance).
*   **How (Thực hiện như thế nào?):** Loại bỏ các đặc trưng có phương sai bằng 0; lọc bớt một trong hai đặc trưng nếu chúng có hệ số tương quan chéo quá cao ($>0.85$); sử dụng các mô hình cây để xếp hạng độ quan trọng và lọc lấy top $K$ đặc trưng tốt nhất.
