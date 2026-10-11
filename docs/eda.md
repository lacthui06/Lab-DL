# Quy Trình Các Bước Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) - Phân tích khám phá dữ liệu là bước bắt buộc nhằm hiểu rõ cấu trúc dữ liệu, phát hiện các mẫu, điểm bất thường (outliers), kiểm tra giả định và tìm kiếm các mối quan hệ giữa các biến trước khi xây dựng mô hình học máy.

---

## Bước 1. Tổng Quan Dữ Liệu & Kiểu Dữ Liệu (Data Overview & Data Types)

### Các nội dung chi tiết:
*   **Xác định kích thước dữ liệu:** Kiểm tra tổng số hàng (mẫu) và số cột (thuộc tính) của bộ dữ liệu để đánh giá quy mô dữ liệu.
*   **Định dạng và kiểu dữ liệu:** Xác định kiểu dữ liệu của từng cột (số nguyên `int64`, số thực `float64`, chuỗi văn bản `object`, ngày tháng `datetime64[ns]`, kiểu logic `bool`) để chọn phương án xử lý phù hợp.
*   **Xem trước dữ liệu mẫu:** Kiểm tra một số dòng đầu tiên (`.head()`), dòng cuối cùng (`.tail()`) hoặc dòng ngẫu nhiên (`.sample()`) để có cái nhìn trực quan ban đầu về giá trị thực tế của các thuộc tính.
*   **Why :** Để hiểu quy mô của bộ dữ liệu và định dạng của các cột nhằm chọn phương án xử lý thích hợp. Phát hiện các trường hợp định dạng sai (ví dụ: cột ngày tháng hoặc cột số bị lưu thành chuỗi văn bản do chứa ký tự lạ).

---

## Bước 2. Thống Kê Tổng Hợp (Summary Statistics)

### Các nội dung chi tiết:
*   **Đối với thuộc tính số (Numerical Features):**
    *   Tính toán xu hướng trung tâm: Trung bình (Mean), trung vị (Median / Percentile 50%) và yếu vị (Mode).
    *   Đo lường mức độ phân tán: Độ lệch chuẩn (Standard Deviation), giá trị lớn nhất (Max), nhỏ nhất (Min) và các mức phân vị (25%, 75%).
*   **Đối với thuộc tính phân loại (Categorical Features):**
    *   Đếm số lượng giá trị duy nhất (Unique values) trong từng cột.
    *   Xác định tần suất xuất hiện và tỷ lệ phần trăm của từng nhóm phân loại (nhãn xuất hiện nhiều nhất - Mode).
*   **What :** Việc tính toán các chỉ số toán học mô tả độ tập trung và độ phân tán của dữ liệu số cũng như dữ liệu phân loại.
*   **Why :** Để hiểu nhanh các giá trị trung bình, trung vị, giá trị lớn nhất, nhỏ nhất, độ lệch chuẩn của dữ liệu mà chưa cần vẽ biểu đồ, giúp phát hiện sớm các bất thường cực đoan hoặc các giá trị không hợp lý về mặt logic.
*   **Which (Sử dụng công cụ/hàm nào?):** Sử dụng hàm `.describe()` cho cột số và `.describe(include=['O'])` cho cột chữ, hoặc các hàm tính toán riêng lẻ như `.mean()`, `.median()`, `.std()`, `.value_counts()`.
---

## Bước 3. Phân Tích Đơn Biến & Phân Phối (Univariate Analysis & Distributions)

### Các nội dung chi tiết:
*   **Phân tích biến số (Data Distribution):**
    *   Đánh giá hình dạng phân phối (phân phối chuẩn - Normal Distribution, phân phối lệch trái/phải - Skewed Distribution, phân phối nhiều đỉnh).
    *   Xác định độ tập trung của dữ liệu và khoanh vùng sơ bộ các khoảng giá trị bất thường.
*   **Phân tích biến phân loại (Class Distribution):**
    *   Xem xét mức độ phân bố của các nhãn.
    *   Kiểm tra tính cân bằng của dữ liệu (đặc biệt quan trọng đối với biến mục tiêu - target trong bài toán phân loại để phát hiện mất cân bằng nhãn - Imbalanced Class).
*   **What :** Khảo sát phân phối của từng thuộc tính đơn lẻ một cách độc lập, bao gồm phân phối dữ liệu số và phân phối các lớp nhãn.
*   **Why :** Để hiểu rõ hình dạng phân phối (chuẩn hay lệch), độ nhọn dẹt của dữ liệu số, tần suất xuất hiện của các nhãn phân loại và phát hiện xem dữ liệu nhãn mục tiêu có bị mất cân bằng nghiêm trọng không.
*   **Which (Sử dụng công cụ/hàm nào?):** Thư viện trực quan hóa dữ liệu (Seaborn, Matplotlib) với biểu đồ Histogram, biểu đồ mật độ KDE plot (cho biến số) và biểu đồ cột Count plot (cho biến phân loại).
*   **How :** Vẽ biểu đồ tần suất cho biến phân loại để so sánh tỷ lệ giữa các class; vẽ biểu đồ phân phối tần suất cho biến số để đánh giá hình dáng đồ thị có đối xứng hay không.

---

## Bước 4. Phân Tích Song Biến & Tương Quan (Bivariate Analysis, Correlation & Patterns)

### Các nội dung chi tiết:
*   **Mối quan hệ Số - Số (Numerical vs Numerical):**
    *   Xem xét xu hướng tương quan (tuyến tính, phi tuyến, cùng chiều hay ngược chiều).
    *   Đo lường mức độ tương quan tuyến tính thông qua các hệ số tương quan (Pearson, Spearman).
*   **Mối quan hệ Số - Phân loại (Numerical vs Categorical):**
    *   So sánh sự khác biệt về phân phối (trung bình, trung vị, độ phân tán) của biến số giữa các nhóm phân loại khác nhau.
*   **Mối quan hệ Phân loại - Phân loại (Categorical vs Categorical):**
    *   Phân tích tỷ lệ phân bố chéo giữa các thuộc tính phân loại để tìm quy luật kết hợp.
*   **Patterns (Các mẫu dữ liệu):** Nhận diện các quy luật biến động đặc biệt, xu hướng (trend) hoặc tính chu kỳ của dữ liệu.
*   **What :** Khảo sát mối quan hệ giữa cặp hai biến với nhau (đặc biệt là giữa các biến độc lập với biến mục tiêu) để tìm quy luật và hệ số tương quan.
*   **Why :** Để lọc ra những thuộc tính có mối liên kết chặt chẽ với biến mục tiêu và nhận diện các biến độc lập bị tương quan quá mạnh với nhau gây nhiễu (đa cộng tuyến).
*   **Which (Sử dụng công cụ/hàm nào?):** Biểu đồ phân tán Scatter plot (Số vs Số), biểu đồ hộp Box plot / Violin plot (Số vs Phân loại), ma trận tương quan `.corr()` trực quan bằng Heatmap, hoặc Stacked Bar chart (Phân loại vs Phân loại).
*   **How :** Tính toán hệ số tương quan Pearson hoặc Spearman; vẽ biểu đồ phân tán để xem xu hướng tăng/giảm đồng thời; vẽ Box plot so sánh phân phối của một đặc trưng số trên từng nhãn phân loại của biến mục tiêu.

---

## Bước 5. Nhận Diện Dữ Liệu Thiếu, Trùng Lặp & Ngoại Lệ (Missing Data, Duplicate Data & Outliers)

### Các nội dung chi tiết:
*   **Nhận diện dữ liệu thiếu (Missing Data):** Tính toán số lượng và tỷ lệ % dữ liệu bị khuyết thiếu ở mỗi thuộc tính, phân tích xem dữ liệu bị thiếu ngẫu nhiên hay có hệ thống.
*   **Nhận diện dữ liệu trùng lặp (Duplicate Data):** Kiểm tra xem có tồn tại các dòng bản ghi giống nhau hoàn toàn hay trùng lặp khóa chính trong dataset để tránh gây nhiễu cho mô hình học máy.
*   **Nhận diện dữ liệu ngoại lệ (Outliers):** 
    *   Sử dụng khoảng giá trị IQR để xác định các điểm dữ liệu nằm ngoài biên trên và biên dưới ($Q1 - 1.5 \times IQR$ và $Q3 + 1.5 \times IQR$).
    *   Sử dụng phương pháp độ lệch chuẩn (Z-score) để tìm các điểm dữ liệu có giá trị biến động quá xa so với giá trị trung bình ($|Z-score| > 3$).
*   **Why :** Để phát hiện các lỗi về chất lượng dữ liệu, đánh giá mức độ nghiêm trọng của chúng và khoanh vùng các thuộc tính cần làm sạch trước khi đưa vào huấn luyện mô hình.
*   **Which (Sử dụng công cụ/hàm nào?):** Các hàm `.isnull().sum()`, `.duplicated().sum()`, phương pháp khoảng IQR hoặc Z-score.
*   **How :** Tính tỷ lệ phần trăm thiếu trên từng cột; đếm số hàng trùng lặp hoàn toàn; tính toán biên trên/dưới để khoanh vùng các dòng chứa giá trị ngoại lệ bất thường.

---

## Bước 6. Đánh Giá Chất Lượng Dữ Liệu & Trực Quan Hóa (Data Quality & Data Visualization)

### Các nội dung chi tiết:
*   **Data Visualization (Trực quan hóa):** Tổng hợp việc sử dụng biểu đồ trực quan (như Box plot, Scatter plot, Bar chart) để hiển thị cấu trúc dữ liệu một cách trực quan, giúp dễ dàng giải thích kết quả EDA cho các bên liên quan.
*   **Data Quality (Đánh giá chất lượng):** Đánh giá độ sạch, độ bao phủ và tính nhất quán của bộ dữ liệu thô để kết luận xem dữ liệu đã sẵn sàng cho bước huấn luyện mô hình hay chưa.
*   **What :** Là bước tổng kết quá trình EDA trước khi chuyển sang giai đoạn tiền xử lý.
*   **Why :** Để có bức tranh tổng thể cuối cùng về độ sạch của dữ liệu, từ đó đưa ra quyết định xem dữ liệu đã đủ điều kiện huấn luyện mô hình chưa và lên kế hoạch cụ thể cho các bước biến đổi dữ liệu tiếp theo.
*   **How (Thực hiện như thế nào?):** Lập bảng tổng kết số lượng dòng lỗi, tỷ lệ dữ liệu khuyết, danh sách các biến có nhiều outlier và ghi chú các giải pháp đề xuất cho bước Feature Engineering kế tiếp.
